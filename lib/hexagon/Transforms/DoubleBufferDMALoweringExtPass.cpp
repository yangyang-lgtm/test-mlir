//===- DoubleBufferDMALoweringExtPass.cpp -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERDMALOWERINGEXT
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// These attributes are private contracts with
// HexagonDoubleBufferPlanRewriteExtPass. They describe the planned
// prologue/kernel/prefetch structure and are removed after this lowering.
constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role";
constexpr StringLiteral kPlanIdAttr = "db_plan_id";
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue";
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch";
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute";

bool hasRole(Operation *op, StringRef role) {
  auto attr = op->getAttrOfType<StringAttr>(kPlanCopyRoleAttr);
  return attr && attr.getValue() == role;
}

Block &thenBlock(scf::IfOp ifOp) { return ifOp.getThenRegion().front(); }

Value createDMAHandle(Location loc, IRRewriter &rewriter) {
  // One handle tracks the lifetime of one in-flight DMA stream. The lowered
  // double-buffer loop uses one handle for ping loads and one for pong loads.
  return memref_ext::DmaHandleOp::create(
             rewriter, loc, memref_ext::DmaHandleType::get(rewriter.getContext()))
      .getHandle();
}

Value castValidSizeToIndex(Location loc, IRRewriter &rewriter, Value validSize) {
  // memref_ext.load_ex carries valid size as either index or an integer.
  // The positive-size guard uses index comparisons, so normalize here.
  if (validSize.getType().isIndex())
    return validSize;
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                    validSize);
}

Value createPositivePredicate(IRRewriter &rewriter, Location loc,
                              Value numElements) {
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                               numElements, zero);
}

LogicalResult createIfPositive(IRRewriter &rewriter, Location loc,
                               Value numElements,
                               function_ref<LogicalResult()> buildThenBody) {
  // Avoid issuing zero-length DMA requests. The original load_ex accepted a
  // valid-size operand; after lowering, the DMA start is guarded explicitly.
  Value hasElements = createPositivePredicate(rewriter, loc, numElements);
  auto ifOp =
      scf::IfOp::create(rewriter, loc, hasElements, /*withElseRegion=*/false);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  return buildThenBody();
}

LogicalResult replaceLoadWithDMAStart(IRRewriter &rewriter,
                                      memref_ext::LoadExOp load,
                                      Value handle) {
  // Keep the load_ex addressing contract but lower the transfer trigger:
  //   load_ex(ptr, valid, other, target)
  // becomes
  //   if valid > 0 { dma_start_ex(ptr, valid, other, target, handle) }.
  // The wait is inserted separately at the schedule level.
  rewriter.setInsertionPoint(load);
  Value numElements =
      castValidSizeToIndex(load.getLoc(), rewriter, load.getValidSize());
  if (failed(createIfPositive(rewriter, load.getLoc(), numElements, [&] {
        memref_ext::DmaStartExOp::create(
            rewriter, load.getLoc(), load.getPtr(), load.getTensorSizeAttr(),
            load.getValidSize(), load.getOther(), load.getTarget(),
            load.getIsOtherValidAttr(), handle);
        return success();
      })))
    return failure();
  rewriter.eraseOp(load);
  return success();
}

void collectRoleLoads(Block &block, SmallVectorImpl<memref_ext::LoadExOp> &ops) {
  // Only loads cloned by PlanRewriteExt as prefetches participate in DMA
  // lowering. Any unrelated load_ex in the same block is left untouched.
  for (Operation &op : block.without_terminator())
    if (auto load = dyn_cast<memref_ext::LoadExOp>(&op))
      if (hasRole(load, kPrefetchRole))
        ops.push_back(load);
}

void collectRoleStores(Block &block,
                       SmallVectorImpl<memref_ext::StoreExOp> &ops) {
  // Stores are collected to validate that the planned loop still contains a
  // writeback phase. This pass currently does not lower store_ex to DMA.
  for (Operation &op : block.without_terminator())
    if (auto store = dyn_cast<memref_ext::StoreExOp>(&op))
      if (hasRole(store, kDB2StoreRole))
        ops.push_back(store);
}

struct PlannedExtSchedule {
  // The pieces emitted by PlanRewriteExt that must agree before this pass can
  // safely replace prefetch load_ex operations with DMA starts.
  scf::ForOp kernel;
  scf::IfOp prologue;
  scf::IfOp prefetch;
  SmallVector<memref_ext::LoadExOp> prologueLoads;
  SmallVector<memref_ext::LoadExOp> nextLoads;
  SmallVector<memref_ext::StoreExOp> stores;
};

bool extractSchedule(scf::ForOp kernel, PlannedExtSchedule &schedule) {
  if (!kernel->hasAttr(kPlanIdAttr))
    return false;

  auto idAttr = kernel->getAttrOfType<IntegerAttr>(kPlanIdAttr);
  // The kernel loop carries a single i1 ping/pong state. Without that state,
  // the pass cannot choose between ping and pong DMA handles.
  if (!idAttr || kernel.getRegionIterArgs().size() != 1 ||
      !kernel.getRegionIterArgs().front().getType().isInteger(1))
    return false;
  int64_t id = idAttr.getInt();
  schedule.kernel = kernel;

  for (Operation *prev = kernel->getPrevNode(); prev; prev = prev->getPrevNode()) {
    auto ifOp = dyn_cast<scf::IfOp>(prev);
    if (!ifOp || !ifOp->hasAttr(kPlanPrologueAttr))
      continue;
    auto prologueId = ifOp->getAttrOfType<IntegerAttr>(kPlanIdAttr);
    // Match by plan id rather than position alone; multiple planned loops may
    // appear in the same function.
    if (prologueId && prologueId.getInt() == id) {
      schedule.prologue = ifOp;
      break;
    }
  }
  if (!schedule.prologue)
    return false;

  for (Operation &op : kernel.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    // PlanRewriteExt emits exactly one prefetch if near the start of the
    // kernel body. This pass consumes the first marked region it finds.
    if (ifOp && ifOp->hasAttr(kPlanPrefetchAttr)) {
      schedule.prefetch = ifOp;
      break;
    }
  }
  if (!schedule.prefetch)
    return false;

  collectRoleLoads(thenBlock(schedule.prologue), schedule.prologueLoads);
  collectRoleLoads(thenBlock(schedule.prefetch), schedule.nextLoads);
  collectRoleStores(*kernel.getBody(), schedule.stores);

  // Prologue and steady-state prefetches are paired by tile. The store check
  // keeps this pass from consuming a partial plan that has no writeback phase.
  return !schedule.prologueLoads.empty() && !schedule.nextLoads.empty() &&
         schedule.prologueLoads.size() == schedule.nextLoads.size() &&
         !schedule.stores.empty();
}

LogicalResult lowerSchedule(IRRewriter &rewriter,
                            PlannedExtSchedule &schedule) {
  RewriterBase::InsertionGuard guard(rewriter);
  Location loc = schedule.kernel.getLoc();

  rewriter.setInsertionPoint(schedule.prologue);
  Value pingLoadHandle = createDMAHandle(loc, rewriter);
  Value pongLoadHandle = createDMAHandle(loc, rewriter);

  // The prologue always fills the initial ping buffers. The first kernel
  // iteration starts with the loop state selecting ping as current.
  for (memref_ext::LoadExOp load : schedule.prologueLoads)
    if (failed(replaceLoadWithDMAStart(rewriter, load, pingLoadHandle)))
      return failure();

  Value current = schedule.kernel.getRegionIterArgs().front();
  rewriter.setInsertionPoint(schedule.prefetch);
  // current chooses the buffer consumed by this iteration; next chooses the
  // opposite buffer that receives the prefetch for the next iteration.
  Value currentLoadHandle =
      arith::SelectOp::create(rewriter, loc, current, pingLoadHandle,
                              pongLoadHandle);
  Value nextLoadHandle =
      arith::SelectOp::create(rewriter, loc, current, pongLoadHandle,
                              pingLoadHandle);

  for (memref_ext::LoadExOp load : schedule.nextLoads)
    if (failed(replaceLoadWithDMAStart(rewriter, load, nextLoadHandle)))
      return failure();

  // Wait for the DMA that produced the current buffers. The next prefetch was
  // just started above and will be waited on when it becomes current.
  rewriter.setInsertionPointAfter(schedule.prefetch);
  memref_ext::DmaWaitOp::create(rewriter, loc, currentLoadHandle);

  return success();
}

void cleanPlanAttrs(Operation *root) {
  // After DMA lowering, the plan markers are no longer part of the public IR
  // contract and should not leak into later passes.
  root->walk([](Operation *op) {
    op->removeAttr(kPlanIdAttr);
    op->removeAttr(kPlanPrologueAttr);
    op->removeAttr(kPlanPrefetchAttr);
    op->removeAttr(kPlanComputeAttr);
    op->removeAttr(kPlanCopyRoleAttr);
  });
}

struct HexagonDoubleBufferDMALoweringExtPass
    : public ::impl::HexagonDoubleBufferDMALoweringExtBase<
          HexagonDoubleBufferDMALoweringExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    memref_ext::MemRefExtDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> kernels;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Only loops produced by PlanRewriteExt carry a plan id.
      if (!forOp->hasAttr(kPlanIdAttr))
        return;
      kernels.push_back(forOp);
    });

    IRRewriter rewriter(getOperation()->getContext());
    for (scf::ForOp kernel : kernels) {
      PlannedExtSchedule schedule;
      // A missing or partial schedule is treated as a non-applicable loop, not
      // as a hard failure. A lowering failure after a schedule is accepted is a
      // real pass failure.
      if (!extractSchedule(kernel, schedule))
        continue;
      if (failed(lowerSchedule(rewriter, schedule))) {
        signalPassFailure();
        return;
      }
    }
    cleanPlanAttrs(getOperation());
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferDMALoweringExtPass() {
  return std::make_unique<HexagonDoubleBufferDMALoweringExtPass>();
}
