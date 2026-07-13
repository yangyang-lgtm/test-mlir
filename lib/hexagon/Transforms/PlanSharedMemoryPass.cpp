//===- PlanSharedMemoryPass.cpp -------------------------------------------===//
//
// Lifetime-based memory planning for memory space 128.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "triton-shared/Conversion/MemorySpaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_PLANSHAREDMEMORY
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct Allocation {
  memref::AllocOp alloc;
  MemRefType type;
  int64_t size = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t offset = 0;
  SmallVector<Operation *> deallocs;
};

struct ActiveRange {
  int64_t end;
  int64_t offset;
  int64_t size;
};

int64_t getStaticSizeInBytes(MemRefType type) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity())
    return -1;
  int64_t bitWidth = type.getElementTypeBitWidth();
  if (bitWidth <= 0 || bitWidth % 8 != 0)
    return -1;
  int64_t elements =
      std::accumulate(type.getShape().begin(), type.getShape().end(),
                      int64_t{1}, std::multiplies<int64_t>());
  return elements * (bitWidth / 8);
}

bool isAliasProducer(Operation *op, Value source) {
  if (auto subview = dyn_cast<memref::SubViewOp>(op))
    return subview.getSource() == source;
  if (auto cast = dyn_cast<memref::CastOp>(op))
    return cast.getSource() == source;
  if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op))
    return reinterpret.getSource() == source;
  if (auto view = dyn_cast<memref::ViewOp>(op))
    return view.getSource() == source;
  return false;
}

void collectAliasUses(Value root, DenseMap<Operation *, int64_t> &order,
                      int64_t functionEnd, Allocation &info) {
  SmallVector<Value> worklist{root};
  llvm::SmallDenseSet<Value> seenValues;
  llvm::SmallPtrSet<Operation *, 4> seenDeallocs;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seenValues.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (isa<memref::DeallocOp>(user)) {
        if (seenDeallocs.insert(user).second)
          info.deallocs.push_back(user);
        continue;
      }

      auto it = order.find(user);
      if (it != order.end())
        info.end = std::max(info.end, it->second);

      // A value escaping through a region/function terminator is conservatively
      // live until function end.
      if (user->hasTrait<OpTrait::IsTerminator>())
        info.end = functionEnd;

      if (!isAliasProducer(user, value))
        continue;
      for (Value result : user->getResults())
        if (isa<BaseMemRefType>(result.getType()))
          worklist.push_back(result);
    }
  }
}

FailureOr<SmallVector<Allocation>> collectAllocations(func::FuncOp func) {
  DenseMap<Operation *, int64_t> order;
  int64_t nextOrder = 0;
  func.walk([&](Operation *op) { order[op] = nextOrder++; });

  SmallVector<Allocation> allocations;
  bool unsupported = false;
  func.walk([&](memref::AllocOp alloc) {
    auto type = cast<MemRefType>(alloc.getType());
    if (type.getMemorySpaceAsInt() != triton::kSharedMemorySpace)
      return;

    int64_t size = getStaticSizeInBytes(type);
    if (size <= 0) {
      alloc.emitError("plan-shared-memory requires a static identity-layout "
                      "memref with byte-addressable element type");
      unsupported = true;
      return;
    }

    Allocation &info = allocations.emplace_back();
    info.alloc = alloc;
    info.type = type;
    info.size = size;
    info.start = order.lookup(alloc);
    info.end = info.start;
    collectAliasUses(alloc.getResult(), order, nextOrder, info);
  });

  if (unsupported)
    return failure();
  return allocations;
}

int64_t assignOffsets(SmallVectorImpl<Allocation> &allocations,
                      int64_t alignment) {
  llvm::sort(allocations, [](const Allocation &lhs, const Allocation &rhs) {
    if (lhs.start != rhs.start)
      return lhs.start < rhs.start;
    return lhs.alloc->isBeforeInBlock(rhs.alloc);
  });

  SmallVector<ActiveRange> active;
  int64_t peak = 0;
  for (Allocation &allocation : allocations) {
    llvm::erase_if(active, [&](const ActiveRange &range) {
      return range.end < allocation.start;
    });
    llvm::sort(active, [](const ActiveRange &lhs, const ActiveRange &rhs) {
      return lhs.offset < rhs.offset;
    });

    int64_t candidate = 0;
    for (const ActiveRange &range : active) {
      candidate = llvm::alignTo(candidate, alignment);
      if (candidate + allocation.size <= range.offset)
        break;
      candidate = std::max(candidate, range.offset + range.size);
    }
    candidate = llvm::alignTo(candidate, alignment);
    allocation.offset = candidate;
    peak = std::max(peak, candidate + allocation.size);
    active.push_back(
        ActiveRange{allocation.end, allocation.offset, allocation.size});
  }
  return peak;
}

void rewriteAllocations(func::FuncOp func,
                        SmallVectorImpl<Allocation> &allocations, int64_t peak,
                        int64_t alignment) {
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(&func.front());
  auto backingType =
      MemRefType::get({peak}, builder.getI8Type(), AffineMap(),
                      triton::kSharedMemorySpace);
  Value backing = memref::AllocOp::create(
      builder, func.getLoc(), backingType, ValueRange{},
      builder.getI64IntegerAttr(alignment));

  llvm::SmallPtrSet<Operation *, 8> erasedDeallocs;
  for (Allocation &allocation : allocations) {
    for (Operation *dealloc : allocation.deallocs)
      if (erasedDeallocs.insert(dealloc).second)
        dealloc->erase();

    builder.setInsertionPoint(allocation.alloc);
    Value offset = arith::ConstantIndexOp::create(
        builder, allocation.alloc.getLoc(), allocation.offset);
    Value view = memref::ViewOp::create(
        builder, allocation.alloc.getLoc(), allocation.type, backing, offset,
        ValueRange{});
    allocation.alloc.replaceAllUsesWith(view);
    allocation.alloc.erase();
  }

  func.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    memref::DeallocOp::create(builder, returnOp.getLoc(), backing);
  });
}

struct PlanSharedMemoryPass
    : public ::impl::PlanSharedMemoryBase<PlanSharedMemoryPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    if (alignment <= 0 || bufferSize <= 0) {
      getOperation().emitError(
          "plan-shared-memory requires positive alignment and buffer size");
      return signalPassFailure();
    }

    FailureOr<SmallVector<Allocation>> allocations =
        collectAllocations(getOperation());
    if (failed(allocations))
      return signalPassFailure();
    if (allocations->empty())
      return;

    int64_t peak = assignOffsets(*allocations, alignment);
    if (peak > bufferSize) {
      getOperation().emitError()
          << "peak shared-memory usage " << peak
          << " bytes exceeds configured capacity " << bufferSize << " bytes";
      return signalPassFailure();
    }

    rewriteAllocations(getOperation(), *allocations, peak, alignment);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createPlanSharedMemoryPass() {
  return std::make_unique<PlanSharedMemoryPass>();
}
