//===- DoubleBufferDMALoweringPass.cpp -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "hexagon/Common/Common.h"
#include "hexagon/Common/DMATransferUtil.h"
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
#include "hexagon/Transforms/CopyDirection.h"
#include "hexagon/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERDMALOWERING
#include "hexagon/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role";
constexpr StringLiteral kPlanIdAttr = "db_plan_id";
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue";
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch";
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute";

bool hasCopyRoleAndDirection(memref::CopyOp copy, StringRef role,
                             StringRef direction) {
  auto roleAttr = copy->getAttrOfType<StringAttr>(kPlanCopyRoleAttr);
  auto directionAttr = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return roleAttr && roleAttr.getValue() == role && directionAttr &&
         directionAttr.getValue() == direction;
}

Value computeNumElements(Location loc, IRRewriter &rewriter, Value memref) {
  auto type = cast<BaseMemRefType>(memref.getType());
  Value result = arith::ConstantIndexOp::create(rewriter, loc, 1);
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    Value size =
        type.isDynamicDim(dim)
            ? memref::DimOp::create(rewriter, loc, memref, dim).getResult()
            : arith::ConstantIndexOp::create(rewriter, loc,
                                             type.getDimSize(dim)).getResult();
    result = arith::MulIOp::create(rewriter, loc, result, size);
  }
  return result;
}

bool createDMAStartFromCopy(IRRewriter &rewriter, memref::CopyOp copy,
                            Value handle) {
  Operation *dmaStart = nullptr;
  if (!createDMAStartOp(copy.getLoc(), rewriter, copy.getSource(),
                        copy.getTarget(), handle, &dmaStart))
    return false;
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    dmaStart->setAttr(kCopyDirectionAttrName, direction);
  return true;
}

Value createDMAHandle(Location loc, IRRewriter &rewriter) {
  OperationState state(loc, "memref_ext.dma_handle");
  state.addTypes(memref_ext::DmaHandleType::get(rewriter.getContext()));
  return rewriter.create(state)->getResult(0);
}

Value createPositivePredicate(IRRewriter &rewriter, Location loc,
                              Value numElements) {
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                               numElements, zero);
}

bool replaceCopyWithRecordedDMAStart(IRRewriter &rewriter, memref::CopyOp copy,
                                     Value handle, Value numElementSlot,
                                     ArrayRef<Value> tagIndex) {
  rewriter.setInsertionPoint(copy);
  Value numElements =
      computeNumElements(copy.getLoc(), rewriter, copy.getSource());
  memref::StoreOp::create(rewriter, copy.getLoc(), numElements, numElementSlot,
                          tagIndex);
  Value hasElements =
      createPositivePredicate(rewriter, copy.getLoc(), numElements);
  auto guardedStart = scf::IfOp::create(rewriter, copy.getLoc(), hasElements,
                                        /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(guardedStart.thenBlock());
    if (!createDMAStartFromCopy(rewriter, copy, handle))
      return false;
  }
  rewriter.setInsertionPointAfter(guardedStart);
  if (copy->use_empty())
    rewriter.eraseOp(copy);
  else
    return false;
  return true;
}

void insertDMAWait(IRRewriter &rewriter, Location loc, Value handle,
                   Value numElementSlot, ArrayRef<Value> tagIndex) {
  Value numElements =
      memref::LoadOp::create(rewriter, loc, numElementSlot, tagIndex);
  Value hasElements = createPositivePredicate(rewriter, loc, numElements);
  auto guardedWait =
      scf::IfOp::create(rewriter, loc, hasElements, /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(guardedWait.thenBlock());
    OperationState state(loc, "memref_ext.dma_wait");
    state.addOperands(handle);
    rewriter.create(state);
  }
}

Block &thenBlock(scf::IfOp ifOp) { return ifOp.getThenRegion().front(); }

void collectRoleCopies(Block &block, StringRef role, StringRef direction,
                       SmallVectorImpl<memref::CopyOp> &copies) {
  for (Operation &op : block.without_terminator()) {
    auto copy = dyn_cast<memref::CopyOp>(&op);
    if (copy && hasCopyRoleAndDirection(copy, role, direction))
      copies.push_back(copy);
  }
}

struct PlannedSchedule {
  scf::IfOp prologue;
  Operation *prologueAnchor = nullptr;
  scf::ForOp kernel;
  scf::IfOp prefetch;
  SmallVector<memref::CopyOp> prologueLoads;
  SmallVector<memref::CopyOp> nextLoads;
  SmallVector<memref::CopyOp> stores;
};

bool extractSchedule(scf::ForOp kernel, PlannedSchedule &schedule) {
  auto idAttr = kernel->getAttrOfType<IntegerAttr>(kPlanIdAttr);
  if (!idAttr)
    return false;
  if (kernel.getRegionIterArgs().size() != 1 ||
      !kernel.getRegionIterArgs().front().getType().isInteger(1))
    return false;

  int64_t id = idAttr.getInt();
  schedule.kernel = kernel;

  for (Operation *prev = kernel->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    auto ifOp = dyn_cast<scf::IfOp>(prev);
    if (!ifOp || !ifOp->hasAttr(kPlanPrologueAttr))
      continue;
    auto prologueId = ifOp->getAttrOfType<IntegerAttr>(kPlanIdAttr);
    if (prologueId && prologueId.getInt() == id) {
      schedule.prologue = ifOp;
      schedule.prologueAnchor = ifOp;
      break;
    }
  }
  if (!schedule.prologue) {
    for (Operation *prev = kernel->getPrevNode(); prev;
         prev = prev->getPrevNode()) {
      auto copy = dyn_cast<memref::CopyOp>(prev);
      if (!copy || !hasCopyRoleAndDirection(copy, kPrefetchRole,
                                            kGlobalToShared))
        continue;
      schedule.prologueLoads.push_back(copy);
      schedule.prologueAnchor = copy;
    }
    std::reverse(schedule.prologueLoads.begin(), schedule.prologueLoads.end());
  }
  if (!schedule.prologue && schedule.prologueLoads.empty())
    return false;

  for (Operation &op : kernel.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (ifOp && ifOp->hasAttr(kPlanPrefetchAttr)) {
      schedule.prefetch = ifOp;
      break;
    }
  }
  if (!schedule.prefetch)
    return false;

  if (schedule.prologue)
    collectRoleCopies(thenBlock(schedule.prologue), kPrefetchRole,
                      kGlobalToShared, schedule.prologueLoads);
  collectRoleCopies(thenBlock(schedule.prefetch), kPrefetchRole,
                    kGlobalToShared,
                    schedule.nextLoads);
  collectRoleCopies(*kernel.getBody(), kDB2StoreRole, kSharedToGlobal,
                    schedule.stores);

  return schedule.prologueAnchor && !schedule.prologueLoads.empty() &&
         schedule.prologueLoads.size() == schedule.nextLoads.size();
}

bool lowerSchedule(IRRewriter &rewriter, PlannedSchedule &schedule) {
  RewriterBase::InsertionGuard guard(rewriter);
  Location loc = schedule.kernel.getLoc();

  rewriter.setInsertionPoint(schedule.prologueAnchor);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> tagIndex{zero};
  auto numType = MemRefType::get({1}, rewriter.getIndexType());

  SmallVector<Value> pingLoadHandles;
  SmallVector<Value> pongLoadHandles;
  SmallVector<Value> pingNumElements;
  SmallVector<Value> pongNumElements;
  for (unsigned i = 0; i < schedule.prologueLoads.size(); ++i) {
    pingLoadHandles.push_back(createDMAHandle(loc, rewriter));
    pongLoadHandles.push_back(createDMAHandle(loc, rewriter));
    pingNumElements.push_back(memref::AllocOp::create(rewriter, loc, numType));
    pongNumElements.push_back(memref::AllocOp::create(rewriter, loc, numType));
  }

  for (auto [copy, handle, numSlot] :
       llvm::zip(schedule.prologueLoads, pingLoadHandles, pingNumElements)) {
    if (!replaceCopyWithRecordedDMAStart(rewriter, copy, handle, numSlot,
                                         tagIndex))
      return false;
  }

  Value current = schedule.kernel.getRegionIterArgs().front();
  rewriter.setInsertionPoint(schedule.prefetch);
  SmallVector<Value> currentLoadHandles;
  SmallVector<Value> nextLoadHandles;
  SmallVector<Value> currentNumElements;
  SmallVector<Value> nextNumElements;
  for (auto [pingHandle, pongHandle, pingNum, pongNum] :
       llvm::zip(pingLoadHandles, pongLoadHandles, pingNumElements,
                 pongNumElements)) {
    currentLoadHandles.push_back(arith::SelectOp::create(
        rewriter, loc, current, pingHandle, pongHandle));
    nextLoadHandles.push_back(arith::SelectOp::create(
        rewriter, loc, current, pongHandle, pingHandle));
    currentNumElements.push_back(
        arith::SelectOp::create(rewriter, loc, current, pingNum, pongNum));
    nextNumElements.push_back(
        arith::SelectOp::create(rewriter, loc, current, pongNum, pingNum));
  }

  for (auto [copy, handle, numSlot] :
       llvm::zip(schedule.nextLoads, nextLoadHandles, nextNumElements)) {
    if (!replaceCopyWithRecordedDMAStart(rewriter, copy, handle, numSlot,
                                         tagIndex))
      return false;
  }

  rewriter.setInsertionPointAfter(schedule.prefetch);
  for (auto [handle, numSlot] :
       llvm::zip(currentLoadHandles, currentNumElements))
    insertDMAWait(rewriter, loc, handle, numSlot, tagIndex);

  rewriter.setInsertionPointAfter(schedule.kernel);
  for (Value value : pingNumElements)
    memref::DeallocOp::create(rewriter, loc, value);
  for (Value value : pongNumElements)
    memref::DeallocOp::create(rewriter, loc, value);

  return true;
}

void cleanPlanAttrs(Operation *root) {
  root->walk([](Operation *op) {
    op->removeAttr(kPlanIdAttr);
    op->removeAttr(kPlanPrologueAttr);
    op->removeAttr(kPlanPrefetchAttr);
    op->removeAttr(kPlanComputeAttr);
    op->removeAttr(kPlanCopyRoleAttr);
  });
}

struct HexagonDoubleBufferDMALoweringPass
    : public ::impl::HexagonDoubleBufferDMALoweringBase<
          HexagonDoubleBufferDMALoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    memref_ext::MemRefExtDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> kernels;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kPlanIdAttr))
        kernels.push_back(forOp);
    });

    IRRewriter rewriter(getOperation()->getContext());
    for (scf::ForOp kernel : kernels) {
      PlannedSchedule schedule;
      if (!extractSchedule(kernel, schedule))
        continue;
      if (!lowerSchedule(rewriter, schedule)) {
        signalPassFailure();
        return;
      }
    }

    cleanPlanAttrs(getOperation());
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferDMALoweringPass() {
  return std::make_unique<HexagonDoubleBufferDMALoweringPass>();
}
