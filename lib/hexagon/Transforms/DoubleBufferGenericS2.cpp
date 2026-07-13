//===- DoubleBufferGenericS2.cpp - double buffer stage 2 implementation ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass implements stage 2 of the double buffering in Hexagon-MLIR.
// It takes the double-buffered IR and replaces the `memref.copy`s  with
// `memref.dma_start` and `memref.dma_wait` at the appropriate insertion points.
// See DoubleBufferGenericS1.cpp for details on stage-1.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Common/DMATransferUtil.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/CopyDirection.h"
#include "hexagon/Transforms/Passes.h"
#include "hexagon/Transforms/Transforms.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "double-buffer-generic-s2"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERGENERICS2
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// To parse the state of double-buffered graph IR.
struct KernelSchedule {
  scf::IfOp prefetch;
  SmallVector<Operation *> computeOps;
  SmallVector<memref::CopyOp> stores;
};
struct DBSchedule {
  scf::IfOp prologue;
  scf::ForOp kernel;
  KernelSchedule body;
};

/// 同时校验 S1 写入的调度角色和原始 copy_direction。
///
/// prefetch 必须对应 global_to_shared，store 必须对应
/// shared_to_global，使 S2 不需要再根据 copy 的位置猜测角色。
bool hasCopyRoleAndDirection(memref::CopyOp copy, StringRef role,
                             StringRef direction) {
  auto roleAttr = copy->getAttrOfType<StringAttr>("db_copy_role");
  auto directionAttr = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  return roleAttr && roleAttr.getValue() == role && directionAttr &&
         directionAttr.getValue() == direction;
}

/// Create a DMA start for `copy` and preserve its transfer direction metadata.
bool createDMAStartFromCopy(Location loc, IRRewriter &rewriter,
                            memref::CopyOp copy, Value tag) {
  Operation *dmaStart = nullptr;
  if (!createDMAStartOp(loc, rewriter, copy.getSource(), copy.getTarget(), tag,
                        &dmaStart))
    return false;
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    dmaStart->setAttr(kCopyDirectionAttrName, direction);
  return true;
}

/// Insert dma_waits using provided tags and num-elements.
void insertDMAWaits(Location loc, IRRewriter &rewriter, ArrayRef<Value> tags,
                    ArrayRef<Value> tagIndex, ArrayRef<Value> numElementSlots) {
  for (int i = 0; i < tags.size(); ++i) {
    Value numElements =
        memref::LoadOp::create(rewriter, loc, numElementSlots[i], tagIndex);
    memref::DmaWaitOp::create(rewriter, loc, tags[i], tagIndex, numElements);
  }
}

/// Replace `memref.copy` with `dma_start` using provided tags.
void replacePrefetchWithDMA(Location loc, IRRewriter &rewriter,
                            scf::IfOp schedule, ArrayRef<Value> tags,
                            ArrayRef<Value> numElementSlots,
                            ArrayRef<Value> tagIndex) {
  SmallVector<memref::CopyOp, 3> copies;
  Region &thenRegion = schedule.getThenRegion();
  assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
  Block &block = thenRegion.front();

  // warning: Don't be tempted to do replaceOp within this loop.
  for (Operation &op : block) {
    auto copyOp = dyn_cast<memref::CopyOp>(&op);
    // 只将 S1 标记的 global -> shared 预取转换成异步 DMA。
    if (copyOp && hasCopyRoleAndDirection(copyOp, "prefetch", kGlobalToShared))
      copies.push_back(copyOp);
  }

  for (int i = 0; i < copies.size(); ++i) {
    auto copy = copies[i];
    rewriter.setInsertionPoint(copy);
    Value numElements = createNumElements(loc, rewriter, copy.getSource());
    memref::StoreOp::create(rewriter, loc, numElements, numElementSlots[i],
                            tagIndex);
    bool created = createDMAStartFromCopy(loc, rewriter, copy, tags[i]);
    assert(created && "unable to create dma_start");
    rewriter.eraseOp(copy);
  }
}

// Replace stores with a DMA start followed by its completion wait.
void replaceStoresWithDMA(Location loc, IRRewriter &rewriter,
                          KernelSchedule &schedule, ArrayRef<Value> tags,
                          ArrayRef<Value> tagIndex) {
  for (int i = 0; i < schedule.stores.size(); ++i) {
    auto copy = schedule.stores[i];
    rewriter.setInsertionPoint(copy);
    Value numElements = createNumElements(loc, rewriter, copy.getSource());
    bool created = createDMAStartFromCopy(loc, rewriter, copy, tags[i]);
    memref::DmaWaitOp::create(rewriter, loc, tags[i], tagIndex, numElements);
    rewriter.eraseOp(schedule.stores[i]);
  }
}

/// Rewrite the double-buffered loop operating on memref.copy as
/// dma_start and dma_wait sequence.
void rewriteAsDMATransfers(IRRewriter &rewriter, DBSchedule schedule) {
  RewriterBase::InsertionGuard guard(rewriter);
  Location loc = schedule.kernel.getLoc();

  // Tabulate preloads.
  SmallVector<memref::CopyOp, 3> preloads;
  Region &thenRegion = schedule.prologue.getThenRegion();
  assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
  Block &block = thenRegion.front();
  for (Operation &op : block) {
    auto copyOp = dyn_cast<memref::CopyOp>(&op);
    // prologue 中的 copy 使用 ping tag，并且必须是输入预取。
    if (copyOp && hasCopyRoleAndDirection(copyOp, "prefetch", kGlobalToShared))
      preloads.push_back(copyOp);
  }

  // Create tag-index
  rewriter.setInsertionPoint(schedule.prologue);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> tagIndex = {zero};

  auto numStores = schedule.body.stores.size();

  // Allocate tags.
  SmallVector<Value> pingWaits;
  SmallVector<Value> pongWaits;
  SmallVector<Value> pingStoreWaits;
  SmallVector<Value> pongStoreWaits;
  SmallVector<Value> pingNumElements;
  SmallVector<Value> pongNumElements;
  auto tagType = rewriter.getI32Type();
  auto waitMemrefType = MemRefType::get({1}, tagType);
  auto numElementsMemrefType = MemRefType::get({1}, rewriter.getIndexType());
  for (auto i = 0; i < preloads.size(); i++) {
    Value pingWait = memref::AllocOp::create(rewriter, loc, waitMemrefType);
    Value pongWait = memref::AllocOp::create(rewriter, loc, waitMemrefType);
    pingWaits.push_back(pingWait);
    pongWaits.push_back(pongWait);
    pingNumElements.push_back(
        memref::AllocOp::create(rewriter, loc, numElementsMemrefType));
    pongNumElements.push_back(
        memref::AllocOp::create(rewriter, loc, numElementsMemrefType));
  }
  for (auto i = 0; i < numStores; ++i) {
    Value ping = memref::AllocOp::create(rewriter, loc, waitMemrefType);
    Value pong = memref::AllocOp::create(rewriter, loc, waitMemrefType);
    pingStoreWaits.push_back(ping);
    pongStoreWaits.push_back(pong);
  }

  // Deallocte tags.
  rewriter.setInsertionPointAfter(schedule.kernel);
  for (auto i = 0; i < preloads.size(); i++) {
    memref::DeallocOp::create(rewriter, loc, pingWaits[i]);
    memref::DeallocOp::create(rewriter, loc, pongWaits[i]);
    memref::DeallocOp::create(rewriter, loc, pingNumElements[i]);
    memref::DeallocOp::create(rewriter, loc, pongNumElements[i]);
  }
  for (auto i = 0; i < numStores; ++i) {
    memref::DeallocOp::create(rewriter, loc, pingStoreWaits[i]);
    memref::DeallocOp::create(rewriter, loc, pongStoreWaits[i]);
  }

  // Rewrite preloads to dma-starts
  for (auto i = 0; i < preloads.size(); i++) {
    memref::CopyOp op = preloads[i];
    rewriter.setInsertionPoint(op);
    Value numElements = createNumElements(loc, rewriter, op.getSource());
    memref::StoreOp::create(rewriter, loc, numElements, pingNumElements[i],
                            tagIndex);
    bool created = createDMAStartFromCopy(loc, rewriter, op, pingWaits[i]);
    assert(created && "unable to create dma_start");
    rewriter.eraseOp(op);
  }

  // Select tags with the same loop-carried `cur` value that S1 uses to select
  // the physical tile buffers. true means current=ping and next=pong.
  if (schedule.kernel.getRegionIterArgs().size() != 1)
    return;
  Value cur = schedule.kernel.getRegionIterArgs().front();
  rewriter.setInsertionPoint(schedule.body.prefetch);
  SmallVector<Value> currentWaits;
  SmallVector<Value> nextWaits;
  SmallVector<Value> currentNumElements;
  SmallVector<Value> nextNumElements;
  for (auto [ping, pong, pingNum, pongNum] :
       llvm::zip(pingWaits, pongWaits, pingNumElements, pongNumElements)) {
    currentWaits.push_back(
        arith::SelectOp::create(rewriter, loc, cur, ping, pong));
    nextWaits.push_back(
        arith::SelectOp::create(rewriter, loc, cur, pong, ping));
    currentNumElements.push_back(
        arith::SelectOp::create(rewriter, loc, cur, pingNum, pongNum));
    nextNumElements.push_back(
        arith::SelectOp::create(rewriter, loc, cur, pongNum, pingNum));
  }

  SmallVector<Value> currentStoreWaits;
  for (auto [ping, pong] : llvm::zip(pingStoreWaits, pongStoreWaits))
    currentStoreWaits.push_back(
        arith::SelectOp::create(rewriter, loc, cur, ping, pong));

  // Start loading the next tile, then wait for the current tile before
  // entering its compute slice.
  replacePrefetchWithDMA(loc, rewriter, schedule.body.prefetch, nextWaits,
                         nextNumElements, tagIndex);
  rewriter.setInsertionPointAfter(schedule.body.prefetch);
  insertDMAWaits(loc, rewriter, currentWaits, tagIndex, currentNumElements);

  replaceStoresWithDMA(loc, rewriter, schedule.body, currentStoreWaits,
                       tagIndex);

  // db_compute 只用于 S1/S2 之间传递调度信息，DMA 改写完成后清理。
  for (Operation *compute : schedule.body.computeOps)
    compute->removeAttr("db_compute");
}

// Extract the single-copy loop body produced by S1.
bool extractKernelBody(scf::ForOp kernel, KernelSchedule &schedule) {
  for (Operation &op : kernel.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (ifOp && ifOp->hasAttr("db_prefetch")) {
      if (schedule.prefetch)
        return false;
      schedule.prefetch = ifOp;
    }
  }

  // schedule.prefetch is already set by extractPreFetch
  if (!schedule.prefetch)
    return false;

  Operation *currentOp = schedule.prefetch->getNextNode();
  while (currentOp) {
    if (auto copyOp = dyn_cast<memref::CopyOp>(currentOp)) {
      // 只收集 S1 标记的 shared -> global 结果回写。
      if (hasCopyRoleAndDirection(copyOp, "store", kSharedToGlobal))
        schedule.stores.push_back(copyOp);
    } else if (currentOp->hasAttr("db_compute")) {
      schedule.computeOps.push_back(currentOp);
    }
    currentOp = currentOp->getNextNode();
  }
  if (schedule.computeOps.empty())
    return false;
  return true;
}

/// State machine to parse the double buffered graph.
bool extractSchedule(scf::ForOp forOp, DBSchedule &schedule) {
  auto attr = forOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
  if (!attr)
    return false;
  int64_t id = attr.getInt();
  schedule.kernel = forOp;

  // Extract the prologue.
  bool foundPrologue = false;
  for (Operation *prev = forOp->getPrevNode(); prev != nullptr;
       prev = prev->getPrevNode()) {
    auto ifOp = dyn_cast<scf::IfOp>(prev);
    if (!ifOp)
      continue;
    auto attr = ifOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
    if (attr && id == attr.getInt() && ifOp->hasAttr("db_prologue")) {
      foundPrologue = true;
      schedule.prologue = ifOp;
      break;
    }
  }
  if (!foundPrologue)
    return false;
  DBG("Prologue = " << schedule.prologue);

  if (forOp.getRegionIterArgs().size() != 1 ||
      !forOp.getRegionIterArgs().front().getType().isInteger(1) ||
      !extractKernelBody(forOp, schedule.body))
    return false;
  return true;
}

// The concrete implementation of the Double-Buffer-Generic-Stage-2 pass.
struct HexagonDoubleBufferGenericS2Pass
    : public ::impl::HexagonDoubleBufferGenericS2Base<
          HexagonDoubleBufferGenericS2Pass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    // registry.insert<mlir::hexagonmem::HexagonMemDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    func.walk([](scf::ForOp forOp) {
      DBSchedule schedule;
      bool res = extractSchedule(forOp, schedule);
      if (res) {
        IRRewriter rewriter(forOp.getContext());
        rewriteAsDMATransfers(rewriter, schedule);
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

/// Creates an instance of the DoubleBufferGenericS2 pass.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS2Pass() {
  return std::make_unique<HexagonDoubleBufferGenericS2Pass>();
}
