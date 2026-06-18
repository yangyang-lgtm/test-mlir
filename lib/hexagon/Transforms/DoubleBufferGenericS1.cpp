//===- DoubleBufferGenericS1.cpp - Double Buffer Generic Pass : Stage 1 ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass transforms single-buffered tiled linalg-generic loops into
// software-pipelined double-buffered loops using ping-pong buffers to overlap
// DMA transfers with computation. It allocates two buffer sets, creates a
// prologue to prefetch the first iteration, and generates alternating ping/pong
// sub-kernels that prefetch the next iteration while computing the current one.
//
// This implementation is mainly in two main passes. First is this one and
// the next is in DoubleBufferGenericS2.cpp
//
// Future extensions will support additional cases including >2D DMA transfers,
// reduction loops, multi-buffering strategies, and broader pattern coverage.
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "double-buffer-generic-s1"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERGENERICS1
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Structs to parse and store the schedule.
struct ScheduleTriplet {
  memref::AllocOp alloc;
  memref::CopyOp load;
};

struct SingleBufferSchedule {
  scf::ForOp forOp;
  SmallVector<ScheduleTriplet> triplets;
  SmallVector<Operation *> computeOps;
  SmallVector<memref::CopyOp> stores;
};

bool isSupportedCompute(Operation *op) {
  return isa<linalg::LinalgOp, scf::ForOp, scf::ForallOp>(op);
}

memref::AllocOp findBaseAlloc(Value value) {
  while (true) {
    if (auto alloc = value.getDefiningOp<memref::AllocOp>())
      return alloc;
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      value = reinterpret.getSource();
      continue;
    }
    return nullptr;
  }
}

bool isSameCopy(memref::CopyOp lhs, memref::CopyOp rhs) {
  return lhs.getOperation() == rhs.getOperation();
}

bool isAvailableBeforeCopy(memref::AllocOp alloc, scf::ForOp forOp,
                           memref::CopyOp copy) {
  Block *forBody = forOp.getBody();
  if (alloc->getBlock() == forBody)
    return alloc->isBeforeInBlock(copy);
  return alloc->getBlock() == forOp->getBlock() &&
         alloc->isBeforeInBlock(forOp);
}

bool isDefinedInBlockBefore(Operation *op, Block *block, Operation *limit) {
  return op && op->getBlock() == block && op->isBeforeInBlock(limit);
}

Value cloneValueSlice(Value value, IRRewriter &rewriter, IRMapping &mapping,
                      Block *sourceBlock, Operation *limit) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;

  auto *def = value.getDefiningOp();
  if (!isDefinedInBlockBefore(def, sourceBlock, limit))
    return value;

  for (Value operand : def->getOperands())
    cloneValueSlice(operand, rewriter, mapping, sourceBlock, limit);

  Operation *cloned = def->clone(mapping);
  rewriter.insert(cloned);
  mapping.map(def->getResults(), cloned->getResults());
  return mapping.lookup(value);
}

memref::CopyOp cloneCopyWithMappedSlices(memref::CopyOp copy,
                                         IRRewriter &rewriter,
                                         IRMapping &mapping,
                                         Block *sourceBlock) {
  Value source = cloneValueSlice(copy.getSource(), rewriter, mapping,
                                 sourceBlock, copy);
  Value target = cloneValueSlice(copy.getTarget(), rewriter, mapping,
                                 sourceBlock, copy);
  auto clonedCopy =
      memref::CopyOp::create(rewriter, copy.getLoc(), source, target);
  if (Attribute direction = copy->getAttr("copy_direction"))
    clonedCopy->setAttr("copy_direction", direction);
  return clonedCopy;
}

Operation *cloneOpWithMappedSlices(Operation *op, IRRewriter &rewriter,
                                   IRMapping &mapping, Block *sourceBlock) {
  for (Value operand : op->getOperands())
    cloneValueSlice(operand, rewriter, mapping, sourceBlock, op);

  Operation *cloned = op->clone(mapping);
  rewriter.insert(cloned);
  mapping.map(op->getResults(), cloned->getResults());
  return cloned;
}

struct TileDataflowAnalysis {
  AliasAnalysis &aliasAnalysis;
  scf::ForOp forOp;
  SingleBufferSchedule &schedule;
  bool verifiedTiledGenericParallel;
  llvm::SmallDenseSet<Operation *> tileAllocs;
  llvm::SmallDenseSet<Operation *> preloadCopies;
  SetVector<Operation *> storeBackwardSlice;

  bool aliasesTileBuffer(Value value) {
    if (!isa<BaseMemRefType>(value.getType()))
      return false;

    for (Operation *op : tileAllocs) {
      Value tileBuffer = cast<memref::AllocOp>(op).getMemref();
      if (aliasAnalysis.alias(value, tileBuffer))
        return true;
    }
    return false;
  }

  bool opUsesTileBufferValue(Operation *op) {
    bool touches = false;
    op->walk([&](Operation *nestedOp) {
      if (touches)
        return WalkResult::interrupt();
      for (Value operand : nestedOp->getOperands()) {
        if (aliasesTileBuffer(operand)) {
          touches = true;
          return WalkResult::interrupt();
        }
      }
      for (Value result : nestedOp->getResults()) {
        if (aliasesTileBuffer(result)) {
          touches = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return touches;
  }

  ModRefResult getTileModRef(Operation *op) {
    ModRefResult result = ModRefResult::getNoModRef();
    for (Operation *allocOp : tileAllocs) {
      Value tileBuffer = cast<memref::AllocOp>(allocOp).getMemref();
      result = result.merge(aliasAnalysis.getModRef(op, tileBuffer));
    }
    return result;
  }

  bool opTouchesTileBuffer(Operation *op) {
    if (opUsesTileBufferValue(op))
      return true;
    if (isMemoryEffectFree(op))
      return false;
    return getTileModRef(op).isModOrRef();
  }

  bool isAllowedLocalUtilityOp(Operation *op) {
    if (isa<memref::AllocOp, memref::DeallocOp, memref::SubViewOp,
            memref::CastOp, memref::ReinterpretCastOp>(op))
      return true;
    return isMemoryEffectFree(op);
  }

  bool computeSliceTouchesAlloc(memref::AllocOp alloc) {
    Value tileBuffer = alloc.getMemref();
    for (Operation *compute : schedule.computeOps) {
      bool touches = false;
      compute->walk([&](Operation *nestedOp) {
        if (touches)
          return WalkResult::interrupt();
        for (Value operand : nestedOp->getOperands()) {
          if (isa<BaseMemRefType>(operand.getType()) &&
              aliasAnalysis.alias(operand, tileBuffer)) {
            touches = true;
            return WalkResult::interrupt();
          }
        }
        for (Value result : nestedOp->getResults()) {
          if (isa<BaseMemRefType>(result.getType()) &&
              aliasAnalysis.alias(result, tileBuffer)) {
            touches = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (touches)
        return true;
    }
    return false;
  }

  bool collectCopyIns() {
    Block *forBody = forOp.getBody();
    for (Operation &op : forBody->without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy)
        continue;
      auto alloc = findBaseAlloc(copy.getTarget());
      if (!alloc || !isAvailableBeforeCopy(alloc, forOp, copy))
        continue;
      schedule.triplets.push_back({alloc, copy});
      tileAllocs.insert(alloc.getOperation());
      preloadCopies.insert(copy.getOperation());
    }

    return !schedule.triplets.empty();
  }

  bool refineCopyInsToComputeUsedBuffers() {
    SmallVector<ScheduleTriplet> refinedTriplets;
    llvm::SmallDenseSet<Operation *> refinedAllocs;
    llvm::SmallDenseSet<Operation *> refinedPreloadCopies;
    for (ScheduleTriplet triplet : schedule.triplets) {
      if (!computeSliceTouchesAlloc(triplet.alloc))
        continue;
      refinedTriplets.push_back(triplet);
      refinedAllocs.insert(triplet.alloc.getOperation());
      refinedPreloadCopies.insert(triplet.load.getOperation());
    }
    if (refinedTriplets.empty())
      return false;
    schedule.triplets = std::move(refinedTriplets);
    tileAllocs = std::move(refinedAllocs);
    preloadCopies = std::move(refinedPreloadCopies);
    return true;
  }

  bool collectCopyOuts() {
    Block *forBody = forOp.getBody();
    for (Operation &op : forBody->without_terminator()) {
      auto copy = dyn_cast<memref::CopyOp>(&op);
      if (!copy)
        continue;
      auto sourceAlloc = findBaseAlloc(copy.getSource());
      if (!sourceAlloc || !tileAllocs.contains(sourceAlloc.getOperation()))
        continue;
      auto targetAlloc = findBaseAlloc(copy.getTarget());
      if (targetAlloc && tileAllocs.contains(targetAlloc.getOperation()))
        continue;
      schedule.stores.push_back(copy);
    }

    return !schedule.stores.empty();
  }

  bool collectStoreSlices() {
    BackwardSliceOptions options;
    options.omitUsesFromAbove = false;
    options.filter = [&](Operation *op) {
      return op->getBlock() == forOp.getBody() || forOp->isAncestor(op);
    };

    for (memref::CopyOp store : schedule.stores) {
      if (failed(getBackwardSlice(store.getSource(), &storeBackwardSlice,
                                  options)))
        return false;
      if (failed(getBackwardSlice(store.getTarget(), &storeBackwardSlice,
                                  options)))
        return false;
    }
    return true;
  }

  bool collectComputeSlice() {
    Block *forBody = forOp.getBody();
    for (Operation &op : forBody->without_terminator()) {
      bool inStoreSlice = storeBackwardSlice.contains(&op);
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) {
        if (preloadCopies.contains(copy.getOperation()) ||
            llvm::any_of(schedule.stores, [&](memref::CopyOp store) {
              return isSameCopy(store, copy);
            }))
          continue;
        auto sourceAlloc = findBaseAlloc(copy.getSource());
        auto targetAlloc = findBaseAlloc(copy.getTarget());
        if (sourceAlloc && tileAllocs.contains(sourceAlloc.getOperation()) &&
            (!targetAlloc || !tileAllocs.contains(targetAlloc.getOperation())))
          continue;
        if (opTouchesTileBuffer(&op))
          return false;
        continue;
      }

      bool touchesTile = opTouchesTileBuffer(&op);
      if (!touchesTile && !inStoreSlice)
        continue;

      if (!isSupportedCompute(&op)) {
        if (isAllowedLocalUtilityOp(&op))
          continue;
        return false;
      }

      if (isa<scf::ForOp>(&op) && !verifiedTiledGenericParallel)
        return false;
      schedule.computeOps.push_back(&op);
    }

    return !schedule.computeOps.empty();
  }

  bool validateScheduleOrder() {
    Operation *firstCompute = schedule.computeOps.front();
    Operation *lastCompute = schedule.computeOps.back();
    for (auto triplet : schedule.triplets) {
      if (!triplet.load->isBeforeInBlock(firstCompute))
        return false;
    }
    for (memref::CopyOp store : schedule.stores) {
      if (!lastCompute->isBeforeInBlock(store))
        return false;
    }
    return true;
  }

  bool run() {
    return collectCopyIns() && collectComputeSlice() &&
           refineCopyInsToComputeUsedBuffers() && collectCopyOuts() &&
           collectStoreSlices() && validateScheduleOrder();
  }
};

// Generate the ping or pong region IR of the double-buffered loop.
void generatePingPongSubKernel(IRRewriter &rewriter,
                               SingleBufferSchedule &schedule,
                               scf::LoopNest &dbLoopNest, scf::IfOp ppongIfOp,
                               Value nextExists, Value nextIdx,
                               ArrayRef<Value> thisBuffers,
                               ArrayRef<Value> nextBuffers, Value toggle,
                               Value toggleNextStoreValue) {
  auto forOp = schedule.forOp;
  auto loc = forOp.getLoc();
  auto context = forOp.getContext();
  auto indVar = forOp.getInductionVar();

  SmallVector<Value> ivs = llvm::map_to_vector(
      dbLoopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  assert(ivs.size() == 1 && "expecting a single loop at this point");
  Block *forBody = dbLoopNest.loops.back().getBody();
  Value newDBLoopIndVar = dbLoopNest.loops.back().getInductionVar();

  // Create prefetch section: executes if this is not the last iteration.
  rewriter.setInsertionPointToStart(&ppongIfOp.getThenRegion().front());
  auto ifNotLastIter =
      scf::IfOp::create(rewriter, loc, TypeRange(), nextExists, false);
  ifNotLastIter->setAttr("db_prefetch", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifNotLastIter.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    IRMapping mapping;
    mapping.map(indVar, nextIdx);
    mapping.map(schedule.triplets[i].alloc, nextBuffers[i]);
    cloneCopyWithMappedSlices(schedule.triplets[i].load, rewriter, mapping,
                              forOp.getBody());
  }

  // Re-map the compute slice.
  rewriter.setInsertionPointAfter(ifNotLastIter);
  IRMapping mapping2;
  mapping2.map(indVar, newDBLoopIndVar);
  for (auto i = 0; i < schedule.triplets.size(); ++i)
    mapping2.map(schedule.triplets[i].alloc, thisBuffers[i]);
  for (Operation *compute : schedule.computeOps)
    cloneOpWithMappedSlices(compute, rewriter, mapping2, forOp.getBody());

  // Store the results.
  for (auto i = 0; i < schedule.stores.size(); ++i) {
    cloneCopyWithMappedSlices(schedule.stores[i], rewriter, mapping2,
                              forOp.getBody());
  }
  memref::StoreOp::create(rewriter, loc, toggleNextStoreValue, toggle,
                          ValueRange{});
}

/// Rewrite the single-buffered loop as double buffered.
void rewriteAsDoubleBuffered(IRRewriter &rewriter, scf::ForOp sbForOp,
                             SingleBufferSchedule &schedule, int &uid) {
  auto loc = sbForOp.getLoc();
  auto context = sbForOp.getContext();

  // Define some general types.
  auto boolType = rewriter.getI1Type();
  auto boolMemrefType = MemRefType::get({}, boolType);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sbForOp);

  // Define some general constants.
  Value trueVal = arith::ConstantOp::create(rewriter, loc, boolType,
                                            rewriter.getBoolAttr(true));
  Value falseVal = arith::ConstantOp::create(rewriter, loc, boolType,
                                             rewriter.getBoolAttr(false));

  // Allocate `toggle` for ping-pong state and initialize it to true.
  Value toggle = memref::AllocOp::create(rewriter, loc, boolMemrefType);
  auto store =
      memref::StoreOp::create(rewriter, loc, trueVal, toggle, ValueRange{});

  // Allocate Ping-Pong Buffers.
  int64_t alignment = 2048;
  auto alignmentAttr = rewriter.getI64IntegerAttr(alignment);
  SmallVector<Value, 3> pingBuffers;
  SmallVector<Value, 3> pongBuffers;
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    auto alloc = schedule.triplets[i].alloc;
    Value ping = memref::AllocOp::create(rewriter, loc, alloc.getType(),
                                         mlir::ValueRange{}, alignmentAttr);
    Value pong = memref::AllocOp::create(rewriter, loc, alloc.getType(),
                                         mlir::ValueRange{}, alignmentAttr);
    pingBuffers.push_back(ping);
    pongBuffers.push_back(pong);
  }

  // Get the bounds from the single-buffer original for-loop.
  Value lowerBound = sbForOp.getLowerBound();
  Value upperBound = sbForOp.getUpperBound();
  Value step = sbForOp.getStep();
  auto indVar = sbForOp.getInductionVar();
  auto idAttr = mlir::IntegerAttr::get(rewriter.getI64Type(), uid++);

  // Prologue: executes iff not 0-iteration loop.
  Value mayLoop =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            lowerBound, upperBound)
          .getResult();
  Value mayLoopVar = memref::AllocOp::create(rewriter, loc, boolMemrefType);
  memref::StoreOp::create(rewriter, loc, mayLoop, mayLoopVar, ValueRange{});
  Value mayLoopReFetch =
      memref::LoadOp::create(rewriter, loc, boolType, mayLoopVar);
  auto ifMayLoop =
      scf::IfOp::create(rewriter, loc, TypeRange(), mayLoopReFetch, false);
  ifMayLoop->setAttr("db_generic", idAttr);
  ifMayLoop->setAttr("db_prologue", UnitAttr::get(context));
  rewriter.setInsertionPointToStart(&ifMayLoop.getThenRegion().front());
  for (auto i = 0; i < schedule.triplets.size(); ++i) {
    IRMapping mapping;
    mapping.map(indVar, lowerBound);
    mapping.map(schedule.triplets[i].alloc, pingBuffers[i]);
    cloneCopyWithMappedSlices(schedule.triplets[i].load, rewriter, mapping,
                              sbForOp.getBody());
  }

  // Kernel: create the new double-buffered top-loop.
  rewriter.setInsertionPoint(sbForOp);
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, SmallVector<Value>{lowerBound},
      SmallVector<Value>{upperBound}, SmallVector<Value>{step});
  loopNest.loops.back()->setAttr("db_generic", idAttr);
  Block *forBody = loopNest.loops.back().getBody();
  Value dbIndVar = loopNest.loops.back().getInductionVar();

  // Toggle decides whether this is ping-or-pong-stage.
  rewriter.setInsertionPoint(forBody->getTerminator());
  Value toggleVal = memref::LoadOp::create(rewriter, loc, boolType, toggle);

  // Check if next preload should happen (or this is last iteration).
  Value nextIdx =
      arith::AddIOp::create(rewriter, loc, dbIndVar, step).getResult();
  Value nextExists =
      arith::CmpIOp::create(rewriter, loc, mlir::arith::CmpIPredicate::slt,
                            nextIdx, upperBound)
          .getResult();

  // Ping sub-kernel.
  auto pingIfOp =
      scf::IfOp::create(rewriter, loc, TypeRange(), toggleVal, false);
  pingIfOp->setAttr("db_ping_kernel", UnitAttr::get(context));
  generatePingPongSubKernel(rewriter, schedule, loopNest, pingIfOp, nextExists,
                            nextIdx, pingBuffers, pongBuffers, toggle,
                            falseVal /*toggleNextStoreValue*/);

  // Pong sub-kernel.
  rewriter.setInsertionPointAfter(pingIfOp);
  Value invertedToggleVal =
      arith::XOrIOp::create(rewriter, loc, toggleVal, trueVal);
  auto pongIfOp =
      scf::IfOp::create(rewriter, loc, TypeRange(), invertedToggleVal, false);
  pongIfOp->setAttr("db_pong_kernel", UnitAttr::get(context));
  generatePingPongSubKernel(rewriter, schedule, loopNest, pongIfOp, nextExists,
                            nextIdx, pongBuffers, pingBuffers, toggle,
                            trueVal /*toggleNextStoreValue*/);
  rewriter.eraseOp(sbForOp);
}

// State machine to parse the `tiled_generic`.
bool generateSchedule(IRRewriter &rewriter, scf::ForOp forOp,
                      SingleBufferSchedule &schedule,
                      AliasAnalysis &aliasAnalysis) {
  if (!forOp.getInitArgs().empty())
    return false;
  schedule.forOp = forOp;

  bool verifiedTiledGenericParallel = false;
  if (forOp->hasAttr("tiled_generic") && forOp->hasAttr("all_parallel"))
    verifiedTiledGenericParallel = true;

  TileDataflowAnalysis analysis{aliasAnalysis, forOp, schedule,
                                verifiedTiledGenericParallel};
  return analysis.run();
}

// The concrete implementation of the DoubleBufferGenericS1 pass.
struct HexagonDoubleBufferGenericS1Pass
    : public ::impl::HexagonDoubleBufferGenericS1Base<
          HexagonDoubleBufferGenericS1Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    int uniqueID = 0; // for each double-buffered linalg-generic.
    auto func = getOperation();
    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    func.walk([&uniqueID, &aliasAnalysis](scf::ForOp forOp) {
      SingleBufferSchedule schedule;
      IRRewriter rewriter(forOp.getContext());
      bool viableCandidate =
          generateSchedule(rewriter, forOp, schedule, aliasAnalysis);
      if (viableCandidate)
        rewriteAsDoubleBuffered(rewriter, forOp, schedule, uniqueID);
      return WalkResult::advance();
    });
  }
};
} // namespace

/// Creates an instance of the DoubleBufferGenericS1 pass.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS1Pass() {
  return std::make_unique<HexagonDoubleBufferGenericS1Pass>();
}
