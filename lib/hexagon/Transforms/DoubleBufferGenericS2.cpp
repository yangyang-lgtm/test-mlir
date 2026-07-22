//===- DoubleBufferGenericS2.cpp - double buffer stage 2 implementation ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// 本 pass 实现 Hexagon-MLIR 双缓冲的第 2 阶段。
// 它消费 S1 已经生成好的双缓冲 IR，将其中承担预取/回写角色的 memref.copy
// 替换成合适位置上的 memref.dma_start 和 memref.dma_wait。
// S1 的结构化改写逻辑见 DoubleBufferGenericS1.cpp。
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Common/DMATransferUtil.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
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

/// 用于描述 S1 生成的 kernel loop body 调度结构。
struct KernelSchedule {
  // kernel 中负责下一轮预取的 scf.if，带有 db_prefetch 属性。
  scf::IfOp prefetch;
  // prefetch 之后、store 之前的计算操作，S1 用 db_compute 标记。
  SmallVector<Operation *> computeOps;
  // 当前轮结果回写 copy，必须是 db_copy_role=store 且 shared_to_global。
  SmallVector<memref::CopyOp> stores;
};
struct DBSchedule {
  // kernel 之前的 prologue if，用于预取第一轮 tile。
  scf::IfOp prologue;
  // prologue 被 canonicalizer 折叠为直线代码时，记录第一轮 preload copy。
  SmallVector<memref::CopyOp> directProloguePreloads;
  // tag/numElements 临时 buffer 的插入锚点。
  Operation *prologueAnchor = nullptr;
  // S1 生成的双缓冲 kernel loop，带有 db_generic 属性。
  scf::ForOp kernel;
  // kernel body 内部的 prefetch/compute/store 调度。
  KernelSchedule body;
};

/// 同时校验 S1 写入的调度角色和原始 copy_direction。
///
/// prefetch 必须对应 global_to_shared，store 必须对应
/// shared_to_global，使 S2 不需要再根据 copy 的位置猜测角色。
bool hasCopyRoleAndDirection(memref::CopyOp copy, StringRef role,
                             StringRef direction) {
  // S1 为克隆出的 copy 写入 db_copy_role，区分 prefetch 和 store。
  auto roleAttr = copy->getAttrOfType<StringAttr>("db_copy_role");
  // copy_direction 来自前置标注 pass，记录 global_to_shared/shared_to_global。
  auto directionAttr = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  // S2 同时校验角色和方向，避免把错误 copy 转成 DMA。
  return roleAttr && roleAttr.getValue() == role && directionAttr &&
         directionAttr.getValue() == direction;
}

/// 根据一个 memref.copy 创建 dma_start，并保留传输方向元数据。
bool createDMAStartFromCopy(Location loc, IRRewriter &rewriter,
                            memref::CopyOp copy, Value tag, Value fetchData) {
  // createDMAStartOp 通过输出参数返回新建的 dma_start operation。
  Operation *dmaStart = nullptr;
  // 使用原 copy 的 source/target 和给定 tag 创建异步 DMA start。
  if (!createDMAStartOp(loc, rewriter, copy.getSource(), copy.getTarget(), tag,
                        fetchData, &dmaStart))
    return false;
  // 保留 copy_direction，方便后续 pass 或调试继续区分传输方向。
  if (Attribute direction = copy->getAttr(kCopyDirectionAttrName))
    dmaStart->setAttr(kCopyDirectionAttrName, direction);
  // DMA start 创建成功。
  return true;
}

Value createDoubleBufferHandles(Location loc, IRRewriter &rewriter) {
  OperationState state(loc, "memref_ext.create_and_init_handles");
  state.addTypes(
      memref_ext::DoubleBufferDmaHandlesType::get(rewriter.getContext(), 2));
  return rewriter.create(state)->getResult(0);
}

Value createDoubleBufferFetchData(Location loc, IRRewriter &rewriter) {
  OperationState state(loc, "memref_ext.create_fetch_data");
  state.addTypes(
      memref_ext::DoubleBufferDmaFetchDataType::get(rewriter.getContext(), 2));
  return rewriter.create(state)->getResult(0);
}

Value selectDMAHandle(IRRewriter &rewriter, Location loc, Value handles,
                      Value condition) {
  return memref_ext::SelectDmaHandleOp::create(
             rewriter, loc,
             memref_ext::DmaHandleType::get(rewriter.getContext()), condition,
             handles)
      .getResult();
}

Value selectDMAFetchData(IRRewriter &rewriter, Location loc, Value fetchData,
                         Value condition) {
  return memref_ext::SelectDmaFetchDataOp::create(
             rewriter, loc,
             memref_ext::DmaFetchDataType::get(rewriter.getContext()),
             condition, fetchData)
      .getResult();
}

/// 使用给定 handle 插入 dma_wait。
void insertDMAWaits(Location loc, IRRewriter &rewriter,
                    ArrayRef<Value> handles) {
  for (Value handle : handles) {
    OperationState state(loc, "memref_ext.dma_wait");
    state.addOperands(handle);
    rewriter.create(state);
  }
}

/// 将一个 prefetch scf.if 中的 memref.copy 替换为 dma_start。
void replacePrefetchWithDMA(Location loc, IRRewriter &rewriter,
                            scf::IfOp schedule, ArrayRef<Value> tags,
                            ArrayRef<Value> fetchData) {
  // 先收集 copy，再统一改写，避免遍历 block 时边删边改。
  SmallVector<memref::CopyOp, 3> copies;
  // prefetch if 只使用 then region，且 S1 生成时应只有一个 block。
  Region &thenRegion = schedule.getThenRegion();
  assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
  // 取 then block。
  Block &block = thenRegion.front();

  // 不要在这个遍历中直接 replace/erase op，否则会破坏迭代器。
  for (Operation &op : block) {
    // 尝试识别 memref.copy。
    auto copyOp = dyn_cast<memref::CopyOp>(&op);
    // 只将 S1 标记的 global -> shared 预取转换成异步 DMA。
    if (copyOp &&
        hasCopyRoleAndDirection(copyOp, kPrefetchRole, kGlobalToShared))
      copies.push_back(copyOp);
  }

  // 按 copy 顺序使用对应的 tag/numElement slot 改写。
  for (int i = 0; i < copies.size(); ++i) {
    // 当前要替换的 copy。
    auto copy = copies[i];
    // 在 copy 原位置插入 numElements store 和 dma_start。
    rewriter.setInsertionPoint(copy);
    // 用相同 source/target 创建 dma_start。
    bool created =
        createDMAStartFromCopy(loc, rewriter, copy, tags[i], fetchData[i]);
    assert(created && "unable to create dma_start");
    // 删除原同步 copy。
    rewriter.eraseOp(copy);
  }
}

// 将 store copy 替换为 dma_start，并紧跟一个 dma_wait 等待回写完成。
void replaceStoresWithDMA(Location loc, IRRewriter &rewriter,
                          KernelSchedule &schedule, ArrayRef<Value> tags,
                          ArrayRef<Value> fetchData) {
  // 每个 store 对应一个 store tag。
  for (int i = 0; i < schedule.stores.size(); ++i) {
    // 当前 shared_to_global copy。
    auto copy = schedule.stores[i];
    // 在 store copy 原位置插入 DMA。
    rewriter.setInsertionPoint(copy);
    // 启动异步回写。
    bool created =
        createDMAStartFromCopy(loc, rewriter, copy, tags[i], fetchData[i]);
    // 当前实现立即等待 store 完成，保证回写生命周期不越过下一步不安全边界。
    insertDMAWaits(loc, rewriter, tags.slice(i, 1));
    // 删除原同步 store copy。
    rewriter.eraseOp(schedule.stores[i]);
  }
}

/// 将 S1 生成的 memref.copy 双缓冲结构改写为 dma_start/dma_wait 序列。
void rewriteAsDMATransfers(IRRewriter &rewriter, DBSchedule schedule) {
  // 保护调用方 insertion point，函数结束自动恢复。
  RewriterBase::InsertionGuard guard(rewriter);
  // 使用 kernel loop 的 loc 作为新 DMA 相关 op 的位置。
  Location loc = schedule.kernel.getLoc();

  // 收集 prologue 中的第一轮 preload copy。
  SmallVector<memref::CopyOp, 3> preloads;
  if (schedule.prologue) {
    // S1 prologue 是一个 scf.if，第一轮 preload 位于 then region。
    Region &thenRegion = schedule.prologue.getThenRegion();
    assert(!thenRegion.empty() && thenRegion.getBlocks().size() == 1);
    // 取 prologue then block。
    Block &block = thenRegion.front();
    // 遍历 prologue 中的所有顶层操作。
    for (Operation &op : block) {
      // 只关心 memref.copy。
      auto copyOp = dyn_cast<memref::CopyOp>(&op);
      // prologue 中的 copy 使用 ping tag，并且必须是输入预取。
      if (copyOp &&
          hasCopyRoleAndDirection(copyOp, kPrefetchRole, kGlobalToShared))
        preloads.push_back(copyOp);
    }
  } else {
    preloads.append(schedule.directProloguePreloads.begin(),
                    schedule.directProloguePreloads.end());
  }

  // 创建 ping/pong handle 的插入点。
  rewriter.setInsertionPoint(schedule.prologueAnchor);

  // store handle 数量等于 kernel body 中 shared_to_global store 数量。
  auto numStores = schedule.body.stores.size();

  SmallVector<Value> preloadHandles;
  SmallVector<Value> preloadFetchData;
  SmallVector<Value> storeHandles;
  SmallVector<Value> storeFetchData;
  // 每个 preload tile 一套 ping/pong handle。
  for (auto i = 0; i < preloads.size(); i++) {
    Value handles = createDoubleBufferHandles(loc, rewriter);
    Value fetchData = createDoubleBufferFetchData(loc, rewriter);
    preloadHandles.push_back(handles);
    preloadFetchData.push_back(fetchData);
  }
  // 每个 store 也需要 ping/pong handle，因为 current slot 会交替变化。
  for (auto i = 0; i < numStores; ++i) {
    Value handles = createDoubleBufferHandles(loc, rewriter);
    Value fetchData = createDoubleBufferFetchData(loc, rewriter);
    storeHandles.push_back(handles);
    storeFetchData.push_back(fetchData);
  }

  Value trueValue = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getBoolAttr(true));

  // 将 prologue 的第一轮 preload copy 改写成 ping dma_start。
  for (auto i = 0; i < preloads.size(); i++) {
    // 当前 prologue copy。
    memref::CopyOp op = preloads[i];
    // 在原 copy 位置插入 DMA。
    rewriter.setInsertionPoint(op);
    // prologue 总是启动 ping 方向的第一轮 preload。
    Value pingWait = selectDMAHandle(rewriter, loc, preloadHandles[i],
                                     trueValue);
    Value pingFetch =
        selectDMAFetchData(rewriter, loc, preloadFetchData[i], trueValue);
    bool created = createDMAStartFromCopy(loc, rewriter, op, pingWait,
                                          pingFetch);
    assert(created && "unable to create dma_start");
    // 删除原同步 copy。
    rewriter.eraseOp(op);
  }

  // 用 S1 loop-carried 的 cur selector 同步选择 handle。S1 用 cur 选择物理 tile
  // buffer；S2 必须用同一个 cur 选择对应 DMA handle。true 表示 current=ping、
  // next=pong。
  // S1 生成的 kernel 应该只有一个 i1 iter_arg：cur selector。
  if (schedule.kernel.getRegionIterArgs().size() != 1)
    return;
  // 取得 cur selector。
  Value cur = schedule.kernel.getRegionIterArgs().front();
  // handle select 应插在 kernel prefetch if 之前，供后续 wait/prefetch/store 使用。
  rewriter.setInsertionPoint(schedule.body.prefetch);
  Value next = arith::XOrIOp::create(rewriter, loc, cur, trueValue);
  // 当前轮 preload wait 使用的 handle。
  SmallVector<Value> currentWaits;
  // 下一轮 preload dma_start 使用的 handle。
  SmallVector<Value> nextWaits;
  SmallVector<Value> nextFetchData;
  // 根据 cur 在 ping/pong 之间选择 current/next。
  for (Value handles : preloadHandles) {
    // cur=true 时当前等待 ping，否则等待 pong。
    currentWaits.push_back(selectDMAHandle(rewriter, loc, handles, cur));
    // cur=true 时下一轮写 pong，否则写 ping。
    nextWaits.push_back(selectDMAHandle(rewriter, loc, handles, next));
  }
  for (Value fetchData : preloadFetchData)
    nextFetchData.push_back(
        selectDMAFetchData(rewriter, loc, fetchData, next));

  // 当前轮 store 使用的 tag；store 只需要 current，不需要 next。
  SmallVector<Value> currentStoreWaits;
  SmallVector<Value> currentStoreFetchData;
  // 根据 cur 选择 store ping/pong tag。
  for (Value handles : storeHandles)
    currentStoreWaits.push_back(selectDMAHandle(rewriter, loc, handles, cur));
  for (Value fetchData : storeFetchData)
    currentStoreFetchData.push_back(
        selectDMAFetchData(rewriter, loc, fetchData, cur));

  // 先启动下一轮 tile 的 DMA 预取，再在进入当前 compute slice 前等待当前 tile。
  // 这就是 S1 结构化排布和 S2 DMA 化共同实现的 overlap。
  replacePrefetchWithDMA(loc, rewriter, schedule.body.prefetch, nextWaits,
                         nextFetchData);
  // 在 prefetch if 之后、compute 之前等待当前 tile 的 preload DMA 完成。
  rewriter.setInsertionPointAfter(schedule.body.prefetch);
  insertDMAWaits(loc, rewriter, currentWaits);

  // 将当前轮 store copy 转成 DMA start + wait。
  replaceStoresWithDMA(loc, rewriter, schedule.body, currentStoreWaits,
                       currentStoreFetchData);

  // db_compute 只用于 S1/S2 之间传递调度信息，DMA 改写完成后清理。
  for (Operation *compute : schedule.body.computeOps)
    compute->removeAttr("db_compute");
}

// 从 S1 生成的单份 kernel loop body 中恢复 prefetch/compute/store 调度。
bool extractKernelBody(scf::ForOp kernel, KernelSchedule &schedule) {
  // 先在 kernel body 顶层查找唯一的 db_prefetch if。
  for (Operation &op : kernel.getBody()->without_terminator()) {
    // kernel 内的下一轮预取由 S1 标记为 db_prefetch。
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (ifOp && ifOp->hasAttr("db_prefetch")) {
      // 一个 kernel 只能有一个 prefetch if；多个说明结构不符合 S1 约定。
      if (schedule.prefetch)
        return false;
      // 记录 prefetch if。
      schedule.prefetch = ifOp;
    }
  }

  // 没找到 prefetch if，说明不是 S1 生成的双缓冲 kernel。
  if (!schedule.prefetch)
    return false;

  // S1 将 compute/store 放在 prefetch if 之后。
  Operation *currentOp = schedule.prefetch->getNextNode();
  // 从 prefetch 后一路扫描到 loop body terminator 前。
  while (currentOp) {
    // store 是 S1 标记过角色的 memref.copy。
    if (auto copyOp = dyn_cast<memref::CopyOp>(currentOp)) {
      // 只收集 S1 标记的 shared -> global 结果回写。
      if (hasCopyRoleAndDirection(copyOp, "store", kSharedToGlobal))
        schedule.stores.push_back(copyOp);
    // compute 由 S1 克隆时打上 db_compute。
    } else if (currentOp->hasAttr("db_compute")) {
      schedule.computeOps.push_back(currentOp);
    }
    // 扫描下一个顶层操作。
    currentOp = currentOp->getNextNode();
  }
  // 至少要有 compute；否则没有需要等待 preload 的计算区域。
  if (schedule.computeOps.empty())
    return false;
  // kernel body 调度恢复成功。
  return true;
}

/// 从 S1 生成的双缓冲 IR 中解析完整 DBSchedule。
bool extractSchedule(scf::ForOp forOp, DBSchedule &schedule) {
  // S1 给 kernel loop 写入 db_generic id；没有该属性就不是候选。
  auto attr = forOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
  if (!attr)
    return false;
  // 记录 kernel/prologue 配对使用的 id。
  int64_t id = attr.getInt();
  // 当前 forOp 就是 kernel。
  schedule.kernel = forOp;

  // 向 kernel 前方查找同 id 的 prologue if。
  bool foundPrologue = false;
  // prologue 由 S1 插在 kernel 之前，因此向前扫描 sibling operations。
  for (Operation *prev = forOp->getPrevNode(); prev != nullptr;
       prev = prev->getPrevNode()) {
    // 只关心 scf.if。
    auto ifOp = dyn_cast<scf::IfOp>(prev);
    if (!ifOp)
      continue;
    // prologue 必须带同一个 db_generic id。
    auto attr = ifOp->getAttrOfType<mlir::IntegerAttr>("db_generic");
    // 同 id 且带 db_prologue 的 if 才是本 kernel 的 prologue。
    if (attr && id == attr.getInt() && ifOp->hasAttr("db_prologue")) {
      // 找到匹配 prologue。
      foundPrologue = true;
      // 保存 prologue。
      schedule.prologue = ifOp;
      schedule.prologueAnchor = ifOp;
      break;
    }
  }
  if (!foundPrologue) {
    for (Operation *prev = forOp->getPrevNode(); prev != nullptr;
         prev = prev->getPrevNode()) {
      auto copyOp = dyn_cast<memref::CopyOp>(prev);
      if (!copyOp ||
          !hasCopyRoleAndDirection(copyOp, kPrefetchRole, kGlobalToShared))
        continue;
      schedule.directProloguePreloads.push_back(copyOp);
      schedule.prologueAnchor = copyOp;
    }
    std::reverse(schedule.directProloguePreloads.begin(),
                 schedule.directProloguePreloads.end());
  }
  // 找不到 guarded 或 direct prologue，不能安全 DMA 化。
  if (!schedule.prologue && schedule.directProloguePreloads.empty())
    return false;
  // debug 输出找到的 prologue。
  DBG("Prologue = " << schedule.prologue);

  // kernel 必须携带一个 i1 cur selector，并且 body 必须能解析出 prefetch/compute/store。
  if (forOp.getRegionIterArgs().size() != 1 ||
      !forOp.getRegionIterArgs().front().getType().isInteger(1) ||
      !extractKernelBody(forOp, schedule.body))
    return false;
  // 完整调度解析成功。
  return true;
}

// Double-Buffer Generic Stage-2 pass 的具体实现。
struct HexagonDoubleBufferGenericS2Pass
    : public ::impl::HexagonDoubleBufferGenericS2Base<
          HexagonDoubleBufferGenericS2Pass> {
  // 声明本 pass 会创建/依赖 SCF 操作。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref_ext::MemRefExtDialect>();
  }

  // pass 入口：遍历函数内所有 S1 kernel loop，并尝试 DMA 化。
  void runOnOperation() override {
    // 当前 FunctionOpInterface。
    auto func = getOperation();
    // 遍历所有 scf.for；只有带 db_generic 且结构匹配的 loop 会被处理。
    func.walk([](scf::ForOp forOp) {
      // 保存从 S1 IR 中恢复出的调度。
      DBSchedule schedule;
      // 尝试解析当前 loop 是否为 S1 生成的 double-buffer kernel。
      bool res = extractSchedule(forOp, schedule);
      // 解析成功才进行 DMA 改写。
      if (res) {
        // 创建 rewriter。
        IRRewriter rewriter(forOp.getContext());
        // 将 memref.copy 调度替换为 dma_start/dma_wait。
        rewriteAsDMATransfers(rewriter, schedule);
      }
      // 继续遍历其它 loop。
      return WalkResult::advance();
    });
  }
};
} // namespace

/// 创建 DoubleBufferGenericS2 pass 实例。
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonDoubleBufferGenericS2Pass() {
  // 返回 S2 pass，用于把 S1 的 memref.copy 双缓冲结构 DMA 化。
  return std::make_unique<HexagonDoubleBufferGenericS2Pass>();
}
