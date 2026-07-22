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

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERDMALOWERING
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// 以下属性名由 plan rewrite pass 写入，用来把计划结构传给 DMA lowering。
constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role";
constexpr StringLiteral kPlanIdAttr = "db_plan_id";
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue";
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch";
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute";

// 同时检查 copy 的计划角色和拷贝方向，确保只 lowering 目标 copy。
bool hasCopyRoleAndDirection(memref::CopyOp copy, StringRef role,
                             StringRef direction) {
  // 读取计划阶段写入的 copy 角色属性。
  auto roleAttr = copy->getAttrOfType<StringAttr>(kPlanCopyRoleAttr);
  // 读取 copy direction pass 写入的方向属性。
  auto directionAttr = copy->getAttrOfType<StringAttr>(kCopyDirectionAttrName);
  // 两个属性都存在且都匹配时才返回 true。
  return roleAttr && roleAttr.getValue() == role && directionAttr &&
         directionAttr.getValue() == direction;
}

// 计算 memref 的元素总数，用于空拷贝保护和 wait 条件。
Value computeNumElements(Location loc, IRRewriter &rewriter, Value memref) {
  // copy 的 source/target 必须是 memref 类型或其基类。
  auto type = cast<BaseMemRefType>(memref.getType());
  // 乘法累积初始值为 1。
  Value result = arith::ConstantIndexOp::create(rewriter, loc, 1);
  // 逐维乘上静态或动态维度大小。
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    // 动态维度用 memref.dim 读取，静态维度直接创建 index 常量。
    Value size =
        type.isDynamicDim(dim)
            ? memref::DimOp::create(rewriter, loc, memref, dim).getResult()
            : arith::ConstantIndexOp::create(rewriter, loc,
                                             type.getDimSize(dim)).getResult();
    // 将当前维度大小乘入元素总数。
    result = arith::MulIOp::create(rewriter, loc, result, size);
  }
  // 返回所有维度大小的乘积。
  return result;
}

// 根据一个 memref.copy 创建对应的 DMA start，并保留 copy direction 属性。
bool createDMAStartFromCopy(IRRewriter &rewriter, memref::CopyOp copy,
                            Value handle, Value fetchData) {
  // createDMAStartOp 会根据 source/target 构造具体 DMA start 操作。
  Operation *dmaStart = nullptr;
  if (!createDMAStartOp(copy.getLoc(), rewriter, copy.getSource(),
                        copy.getTarget(), handle, fetchData, &dmaStart))
    return false;
  // 将原 copy 的方向属性复制到 DMA start，方便后续 pass 或调试识别。
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

// 生成 numElements > 0 的谓词，避免对空 memref 发起 DMA 或 wait。
Value createPositivePredicate(IRRewriter &rewriter, Location loc,
                              Value numElements) {
  // 创建 index 类型的 0。
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  // 用 signed greater-than 判断元素个数是否大于 0。
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                               numElements, zero);
}

// 将记录元素数的槽位初始化为 0。
void initializeNumElementSlot(IRRewriter &rewriter, Location loc,
                              Value numElementSlot,
                              ArrayRef<Value> tagIndex) {
  // 初始化值为 index 0。
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  // 把 0 写入一维计数 memref 的指定 tag 位置。
  memref::StoreOp::create(rewriter, loc, zero, numElementSlot, tagIndex);
}

// 记录本轮 DMA 覆盖的元素数；多条 copy 共用同一槽位时保留最大值。
void recordNumElements(IRRewriter &rewriter, Location loc, Value numElements,
                       Value numElementSlot, ArrayRef<Value> tagIndex) {
  // 读取槽位里当前已经记录的元素数。
  Value previous =
      memref::LoadOp::create(rewriter, loc, numElementSlot, tagIndex);
  // 用最大值合并多条 copy 的元素数，保证 wait 条件覆盖任一非空 copy。
  Value recorded = arith::MaxSIOp::create(rewriter, loc, previous, numElements);
  // 将合并后的元素数写回槽位。
  memref::StoreOp::create(rewriter, loc, recorded, numElementSlot, tagIndex);
}

// 创建 if (numElements > 0) 并在 then 区域里调用回调生成操作。
LogicalResult createIfPositive(IRRewriter &rewriter, Location loc,
                               Value numElements,
                               function_ref<LogicalResult()> buildThenBody) {
  // 先构造大于 0 的布尔条件。
  Value hasElements = createPositivePredicate(rewriter, loc, numElements);
  // 创建没有 else 分支的 scf.if。
  auto ifOp =
      scf::IfOp::create(rewriter, loc, hasElements, /*withElseRegion=*/false);

  // 暂存插入点，避免回调结束后污染调用者的 rewriter 状态。
  OpBuilder::InsertionGuard guard(rewriter);
  // 后续操作插入到 then block 的开头。
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  // 由调用者生成 then block 的具体内容。
  return buildThenBody();
}

// 用带元素数记录和空拷贝保护的 DMA start 替换一个 memref.copy。
bool replaceCopyWithRecordedDMAStart(IRRewriter &rewriter, memref::CopyOp copy,
                                     Value handle, Value fetchData,
                                     Value numElementSlot,
                                     ArrayRef<Value> tagIndex) {
  // 新操作插入到原 copy 的位置。
  rewriter.setInsertionPoint(copy);
  // 使用 source 形状计算本次 copy 的元素数。
  Value numElements =
      computeNumElements(copy.getLoc(), rewriter, copy.getSource());
  // 把本次元素数合并记录到对应 ping/pong 槽。
  recordNumElements(rewriter, copy.getLoc(), numElements, numElementSlot,
                    tagIndex);
  // 只有元素数大于 0 时才发起 DMA start。
  if (failed(createIfPositive(rewriter, copy.getLoc(), numElements, [&] {
    // 在 then block 内创建 DMA start。
    if (!createDMAStartFromCopy(rewriter, copy, handle, fetchData))
      return failure();
    // 回调成功表示 then block 构造完成。
    return success();
  })))
    return false;
  // 原 copy 没有结果使用时可以安全删除。
  if (copy->use_empty())
    rewriter.eraseOp(copy);
  else
    return false;
  // 替换成功。
  return true;
}

// 在当前插入点生成受元素数保护的 DMA wait。
void insertDMAWait(IRRewriter &rewriter, Location loc, Value handle,
                   Value numElementSlot, ArrayRef<Value> tagIndex) {
  // 读取该 ping/pong 槽记录的元素数。
  Value numElements =
      memref::LoadOp::create(rewriter, loc, numElementSlot, tagIndex);
  // 只有确实发起过非空 DMA 时才等待。
  (void)createIfPositive(rewriter, loc, numElements, [&] {
    // 生成 memref_ext.dma_wait(handle)。
    memref_ext::DmaWaitOp::create(rewriter, loc, handle);
    return success();
  });
}

// 取 scf.if 的 then block，简化后续拷贝收集代码。
Block &thenBlock(scf::IfOp ifOp) { return ifOp.getThenRegion().front(); }

// 在一个 block 的非 terminator 操作中收集指定角色和方向的 memref.copy。
void collectRoleCopies(Block &block, StringRef role, StringRef direction,
                       SmallVectorImpl<memref::CopyOp> &copies) {
  // 遍历 block 内所有非 terminator 操作。
  for (Operation &op : block.without_terminator()) {
    // 只处理 memref.copy。
    auto copy = dyn_cast<memref::CopyOp>(&op);
    // 角色和方向匹配时加入输出列表。
    if (copy && hasCopyRoleAndDirection(copy, role, direction))
      copies.push_back(copy);
  }
}

// 保存从 plan rewrite 结构中抽取出的 double-buffer 调度组件。
struct PlannedSchedule {
  // 可选的 prologue if，用来做第一轮预取。
  scf::IfOp prologue;
  // prologue 插入锚点；没有 prologue if 时可能是原始 copy。
  Operation *prologueAnchor = nullptr;
  // 重写后的主循环。
  scf::ForOp kernel;
  // 循环体内的 next-tile prefetch if。
  scf::IfOp prefetch;
  // 第一轮 load copy。
  SmallVector<memref::CopyOp> prologueLoads;
  // 循环内下一轮 load copy。
  SmallVector<memref::CopyOp> nextLoads;
  // 当前计划里的 store copy；当前 lowering 不改写 store，只保留结构信息。
  SmallVector<memref::CopyOp> stores;
};

// 从带 plan 属性的 scf.for 周围抽取 prologue/prefetch/load/store 信息。
bool extractSchedule(scf::ForOp kernel, PlannedSchedule &schedule) {
  // 只有带 plan id 的循环才是本 pass 的目标。
  auto idAttr = kernel->getAttrOfType<IntegerAttr>(kPlanIdAttr);
  if (!idAttr)
    return false;
  // 当前 lowering 假设循环只有一个 i1 ping/pong iter_arg。
  if (kernel.getRegionIterArgs().size() != 1 ||
      !kernel.getRegionIterArgs().front().getType().isInteger(1))
    return false;

  // 记录计划 id，并保存 kernel。
  int64_t id = idAttr.getInt();
  schedule.kernel = kernel;

  // 优先在 kernel 前方查找同 id 的 prologue if。
  for (Operation *prev = kernel->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    auto ifOp = dyn_cast<scf::IfOp>(prev);
    if (!ifOp || !ifOp->hasAttr(kPlanPrologueAttr))
      continue;
    // 只有 plan id 相同的 prologue 才属于当前 kernel。
    auto prologueId = ifOp->getAttrOfType<IntegerAttr>(kPlanIdAttr);
    if (prologueId && prologueId.getInt() == id) {
      schedule.prologue = ifOp;
      schedule.prologueAnchor = ifOp;
      break;
    }
  }
  // 兼容没有 prologue if 的形态：直接从 kernel 前面的 copy 收集首轮预取。
  if (!schedule.prologue) {
    for (Operation *prev = kernel->getPrevNode(); prev;
         prev = prev->getPrevNode()) {
      auto copy = dyn_cast<memref::CopyOp>(prev);
      if (!copy || !hasCopyRoleAndDirection(copy, kPrefetchRole,
                                            kGlobalToShared))
        continue;
      // 逆向扫描时先 push，后面再 reverse 恢复原顺序。
      schedule.prologueLoads.push_back(copy);
      schedule.prologueAnchor = copy;
    }
    std::reverse(schedule.prologueLoads.begin(), schedule.prologueLoads.end());
  }
  // 没有首轮预取就无法构建 double-buffer DMA。
  if (!schedule.prologue && schedule.prologueLoads.empty())
    return false;

  // 在 kernel body 中查找预取下一轮数据的 if。
  for (Operation &op : kernel.getBody()->without_terminator()) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (ifOp && ifOp->hasAttr(kPlanPrefetchAttr)) {
      schedule.prefetch = ifOp;
      break;
    }
  }
  // 没有循环内 prefetch，说明计划结构不完整。
  if (!schedule.prefetch)
    return false;

  // 如果首轮预取在 prologue if 中，就从 then block 收集 load copy。
  if (schedule.prologue)
    collectRoleCopies(thenBlock(schedule.prologue), kPrefetchRole,
                      kGlobalToShared, schedule.prologueLoads);
  // 从 prefetch if 的 then block 收集下一轮 load copy。
  collectRoleCopies(thenBlock(schedule.prefetch), kPrefetchRole,
                    kGlobalToShared,
                    schedule.nextLoads);
  // 从 kernel body 收集 store copy，供结构完整性和调试使用。
  collectRoleCopies(*kernel.getBody(), kDB2StoreRole, kSharedToGlobal,
                    schedule.stores);

  // 首轮和下一轮 load 数量必须一致，才能共享同一 ping/pong 同步结构。
  return schedule.prologueAnchor && !schedule.prologueLoads.empty() &&
         schedule.prologueLoads.size() == schedule.nextLoads.size();
}

// 将抽取出的 double-buffer copy 计划 lower 成 DMA start/wait。
bool lowerSchedule(IRRewriter &rewriter, PlannedSchedule &schedule) {
  // 用 guard 保证本函数结束后恢复调用者插入点。
  RewriterBase::InsertionGuard guard(rewriter);
  // 后续新建操作沿用 kernel 的位置。
  Location loc = schedule.kernel.getLoc();

  // 在 prologue 之前创建 DMA handle 和计数槽。
  rewriter.setInsertionPoint(schedule.prologueAnchor);
  // 计数槽是一维 memref，索引固定为 0。
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> tagIndex{zero};
  auto numType = MemRefType::get({1}, rewriter.getIndexType());

  Value loadHandles = createDoubleBufferHandles(loc, rewriter);
  SmallVector<Value> loadFetchData;
  loadFetchData.reserve(schedule.prologueLoads.size());
  for (size_t index = 0; index < schedule.prologueLoads.size(); ++index) {
    loadFetchData.push_back(createDoubleBufferFetchData(loc, rewriter));
  }
  // ping/pong 分别记录最近一次 DMA 的元素数，用来保护 wait。
  Value pingNumElements = memref::AllocOp::create(rewriter, loc, numType);
  Value pongNumElements = memref::AllocOp::create(rewriter, loc, numType);
  // 初始化两个计数槽，避免未写入时读取未定义值。
  initializeNumElementSlot(rewriter, loc, pingNumElements, tagIndex);
  initializeNumElementSlot(rewriter, loc, pongNumElements, tagIndex);

  // 首轮预取写入 ping buffer，并记录 ping 的元素数。
  Value trueValue = arith::ConstantOp::create(rewriter, loc,
                                              rewriter.getBoolAttr(true));
  Value pingLoadHandle = selectDMAHandle(rewriter, loc, loadHandles, trueValue);
  SmallVector<Value> pingLoadFetchData;
  pingLoadFetchData.reserve(schedule.prologueLoads.size());
  for (Value pair : loadFetchData)
    pingLoadFetchData.push_back(
        selectDMAFetchData(rewriter, loc, pair, trueValue));
  size_t fetchIndex = 0;
  for (memref::CopyOp copy : schedule.prologueLoads) {
    if (!replaceCopyWithRecordedDMAStart(rewriter, copy, pingLoadHandle,
                                         pingLoadFetchData[fetchIndex++],
                                         pingNumElements,
                                         tagIndex))
      return false;
  }

  // kernel 的 i1 iter_arg 表示当前使用 ping 还是 pong。
  Value current = schedule.kernel.getRegionIterArgs().front();
  // 在 prefetch 之前计算当前轮 wait 和下一轮 start 要用的 handle/计数槽。
  rewriter.setInsertionPoint(schedule.prefetch);
  Value next = arith::XOrIOp::create(rewriter, loc, current, trueValue);
  // currentLoadHandle 对应当前计算正在消费的 buffer。
  Value currentLoadHandle = selectDMAHandle(rewriter, loc, loadHandles, current);
  // nextLoadHandle 对应下一轮预取要填充的 buffer。
  Value nextLoadHandle = selectDMAHandle(rewriter, loc, loadHandles, next);
  SmallVector<Value> nextLoadFetchData;
  nextLoadFetchData.reserve(schedule.nextLoads.size());
  for (Value pair : loadFetchData)
    nextLoadFetchData.push_back(selectDMAFetchData(rewriter, loc, pair, next));
  // currentNumElements 是当前 buffer 上一次 DMA 记录的元素数。
  Value currentNumElements =
      arith::SelectOp::create(rewriter, loc, current, pingNumElements,
                              pongNumElements);
  // nextNumElements 是下一轮预取要写入的计数槽。
  Value nextNumElements =
      arith::SelectOp::create(rewriter, loc, current, pongNumElements,
                              pingNumElements);
  // 下一轮计数槽先清零，再由每条 prefetch copy 合并记录。
  initializeNumElementSlot(rewriter, loc, nextNumElements, tagIndex);

  // 将循环内下一轮预取 copy 替换成 DMA start。
  fetchIndex = 0;
  for (memref::CopyOp copy : schedule.nextLoads) {
    if (!replaceCopyWithRecordedDMAStart(rewriter, copy, nextLoadHandle,
                                         nextLoadFetchData[fetchIndex++],
                                         nextNumElements,
                                         tagIndex))
      return false;
  }

  // prefetch 发起之后，等待当前轮要消费的 load DMA 完成。
  rewriter.setInsertionPointAfter(schedule.prefetch);
  insertDMAWait(rewriter, loc, currentLoadHandle, currentNumElements, tagIndex);

  // kernel 之后释放元素数计数槽。
  rewriter.setInsertionPointAfter(schedule.kernel);
  memref::DeallocOp::create(rewriter, loc, pingNumElements);
  memref::DeallocOp::create(rewriter, loc, pongNumElements);

  // 当前计划 lowering 成功。
  return true;
}

// 清理 plan rewrite 阶段留下的内部属性，避免泄漏到后续 IR。
void cleanPlanAttrs(Operation *root) {
  // 遍历 root 下所有操作，移除本组 pass 的私有属性。
  root->walk([](Operation *op) {
    op->removeAttr(kPlanIdAttr);
    op->removeAttr(kPlanPrologueAttr);
    op->removeAttr(kPlanPrefetchAttr);
    op->removeAttr(kPlanComputeAttr);
    op->removeAttr(kPlanCopyRoleAttr);
  });
}

// Pass 主体：把 double-buffer 计划中的 memref.copy lowering 为 DMA。
struct HexagonDoubleBufferDMALoweringPass
    : public ::impl::HexagonDoubleBufferDMALoweringBase<
          HexagonDoubleBufferDMALoweringPass> {
  // 声明本 pass 会用到的 dialect。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    memref_ext::MemRefExtDialect, scf::SCFDialect>();
  }

  // 在当前 FunctionOpInterface 上执行 DMA lowering。
  void runOnOperation() override {
    // 先收集所有带 plan id 的 kernel，避免遍历时改写 IR 干扰 walk。
    SmallVector<scf::ForOp> kernels;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kPlanIdAttr))
        kernels.push_back(forOp);
    });

    // 使用同一个 rewriter 逐个 lowering kernel。
    IRRewriter rewriter(getOperation()->getContext());
    for (scf::ForOp kernel : kernels) {
      // 从当前 kernel 周围抽取计划结构。
      PlannedSchedule schedule;
      if (!extractSchedule(kernel, schedule))
        continue;
      // 计划结构合法时执行 lowering；失败则标记 pass 失败。
      if (!lowerSchedule(rewriter, schedule)) {
        signalPassFailure();
        return;
      }
    }

    // 所有 lowering 完成后清理内部标记属性。
    cleanPlanAttrs(getOperation());
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferDMALoweringPass() {
  // 返回 pass 实例，供 pipeline 注册和创建。
  return std::make_unique<HexagonDoubleBufferDMALoweringPass>();
}
