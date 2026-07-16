//===- DoubleBufferDMALoweringExtPass.cpp -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h" // 使用 kPrefetchRole/kDB2StoreRole 等双缓冲公共角色常量。
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h" // 生成 memref_ext.dma_handle/dma_start_ex/dma_wait。
#include "hexagon/Transforms/Passes.h" // 引入 TableGen 生成的 pass 声明。

#include "mlir/Dialect/Arith/IR/Arith.h" // 生成 index_cast、cmpi、select 等算术操作。
#include "mlir/Dialect/MemRef/IR/MemRef.h" // 注册 memref dialect，保持 pass 依赖完整。
#include "mlir/Dialect/SCF/IR/SCF.h" // 识别并生成 scf.if/scf.for。
#include "mlir/Interfaces/FunctionInterfaces.h" // 当前 pass 运行在 FunctionOpInterface 上。
#include "mlir/Pass/Pass.h" // 引入 MLIR pass 基础设施。

using namespace mlir; // 简化 MLIR 类型和 builder API 的命名空间书写。
using namespace mlir::hexagon; // 简化 Hexagon 常量、方言和 pass 工厂函数命名空间书写。

#define GEN_PASS_DEF_HEXAGONDOUBLEBUFFERDMALOWERINGEXT // 选择本文件要展开的 TableGen pass 基类。
#include "hexagon/Transforms/Passes.h.inc" // 展开 HexagonDoubleBufferDMALoweringExtBase。

namespace { // 文件内 helper 不需要导出到链接符号表。

// 这些属性是 PlanRewriteExt 与 DMALoweringExt 之间的私有约定。
// lowering 完成后会删除它们，避免泄漏到后续 pipeline。
constexpr StringLiteral kPlanCopyRoleAttr = "db_plan_copy_role"; // 标记 transfer 是 prefetch 还是 store。
constexpr StringLiteral kPlanIdAttr = "db_plan_id"; // 关联同一个 prologue 和 kernel loop。
constexpr StringLiteral kPlanPrologueAttr = "db_plan_prologue"; // 标记首轮预取的 scf.if。
constexpr StringLiteral kPlanPrefetchAttr = "db_plan_prefetch"; // 标记 loop 内下一轮预取的 scf.if。
constexpr StringLiteral kPlanComputeAttr = "db_plan_compute"; // 标记 PlanRewriteExt 克隆出的 compute。

bool hasRole(Operation *op, StringRef role) { // 判断操作是否带有指定的计划 transfer 角色。
  auto attr = op->getAttrOfType<StringAttr>(kPlanCopyRoleAttr); // 读取 db_plan_copy_role 字符串属性。
  return attr && attr.getValue() == role; // 只有属性存在且值匹配时才认为命中角色。
}

Block &thenBlock(scf::IfOp ifOp) { return ifOp.getThenRegion().front(); } // 取无 else 的 scf.if then block。

Value createDMAHandle(Location loc, IRRewriter &rewriter) { // 创建一个 memref_ext DMA handle。
  // 一个 handle 跟踪一条 in-flight DMA 流。双缓冲 load 需要 ping/pong 两个 handle。
  return memref_ext::DmaHandleOp::create( // 在当前位置插入 dma_handle op。
             rewriter, loc, memref_ext::DmaHandleType::get(rewriter.getContext())) // 使用 memref_ext 的 handle 类型。
      .getHandle(); // 返回 handle SSA value。
}

Value castValidSizeToIndex(Location loc, IRRewriter &rewriter, Value validSize) { // 把 valid size 规约到 index 类型。
  // load_ex 的 valid size 可能是 index，也可能是整数；正数 guard 使用 index 比较。
  if (validSize.getType().isIndex()) // 已经是 index 时不需要生成 cast。
    return validSize; // 直接复用原 valid size。
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), // 插入 index_cast。
                                    validSize); // 把整数 valid size 转成 index。
}

Value createPositivePredicate(IRRewriter &rewriter, Location loc,
                              Value numElements) { // 生成 numElements > 0 的谓词。
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0); // 构造 index 类型常量 0。
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt, // 使用 signed greater-than 比较。
                               numElements, zero); // 返回 numElements 是否大于 0。
}

LogicalResult createIfPositive(IRRewriter &rewriter, Location loc,
                               Value numElements,
                               function_ref<LogicalResult()> buildThenBody) { // 用 if 包住只在正元素数时执行的 lowering。
  // 避免发起 0 长度 DMA。原 load_ex 接受 valid size，lower 后显式用 guard 表达。
  Value hasElements = createPositivePredicate(rewriter, loc, numElements); // 计算 valid size 是否为正。
  auto ifOp = // 创建只有 then region 的 scf.if。
      scf::IfOp::create(rewriter, loc, hasElements, /*withElseRegion=*/false);

  OpBuilder::InsertionGuard guard(rewriter); // 保护调用者的插入点。
  rewriter.setInsertionPointToStart(ifOp.thenBlock()); // 在 then block 开头构造 DMA start。
  return buildThenBody(); // 让调用者填充 then block，并把构造结果向上传递。
}

LogicalResult replaceLoadWithDMAStart(IRRewriter &rewriter,
                                      memref_ext::LoadExOp load,
                                      Value handle) { // 把一个 prefetch load_ex 替换为 dma_start_ex。
  // 保留 load_ex 的地址语义，只替换传输触发方式：
  //   load_ex(ptr, valid, other, target)
  // 变成：
  //   if valid > 0 { dma_start_ex(ptr, valid, other, target, handle) }
  // 对应的 dma_wait 在 schedule 层统一插入。
  rewriter.setInsertionPoint(load); // 新的 guard 和 dma_start_ex 放在原 load_ex 位置。
  Value numElements = // 准备用于正数 guard 的元素数。
      castValidSizeToIndex(load.getLoc(), rewriter, load.getValidSize());
  if (failed(createIfPositive(rewriter, load.getLoc(), numElements, [&] { // 只在 valid > 0 时发起 DMA。
        memref_ext::DmaStartExOp::create( // 创建异步 DMA start。
            rewriter, load.getLoc(), load.getPtr(), load.getTensorSizeAttr(), // 复用原 load 的源指针和 tensor size。
            load.getValidSize(), load.getOther(), load.getTarget(), // 复用 valid size、填充值和目标 tile。
            load.getIsOtherValidAttr(), handle); // 复用 other-valid 属性，并绑定选中的 DMA handle。
        return success(); // then body 构造成功。
      })))
    return failure(); // guard 或 dma_start_ex 构造失败时向上传递失败。
  rewriter.eraseOp(load); // 原 load_ex 已被 guarded dma_start_ex 替代。
  return success(); // 当前 load lowering 完成。
}

void collectRoleLoads(Block &block, SmallVectorImpl<memref_ext::LoadExOp> &ops) { // 收集指定 block 内的 planned prefetch load。
  // 只有 PlanRewriteExt 克隆并标记为 prefetch 的 load_ex 参与 DMA lowering。
  // 同一 block 中没有该 role 的 load_ex 保持原样。
  for (Operation &op : block.without_terminator()) // terminator 不可能是 transfer，跳过。
    if (auto load = dyn_cast<memref_ext::LoadExOp>(&op)) // 只关心 memref_ext.load_ex。
      if (hasRole(load, kPrefetchRole)) // 只收集 db_plan_copy_role = prefetch 的 load。
        ops.push_back(load); // 保留原 IR 顺序，后续按同样顺序 lowering。
}

void collectRoleStores(Block &block,
                       SmallVectorImpl<memref_ext::StoreExOp> &ops) { // 收集 planned store 写回。
  // store 当前只用于校验计划结构中仍有写回阶段；本 pass 不把 store_ex lower 成 DMA。
  for (Operation &op : block.without_terminator()) // 遍历 kernel body 的实际操作。
    if (auto store = dyn_cast<memref_ext::StoreExOp>(&op)) // 只关心 memref_ext.store_ex。
      if (hasRole(store, kDB2StoreRole)) // 只收集 db_plan_copy_role = store 的写回。
        ops.push_back(store); // 记录 store，作为 planned schedule 完整性的校验依据。
}

struct PlannedExtSchedule { // 保存一个 PlanRewriteExt 生成的完整 double-buffer 计划。
  // 这些 IR 片段必须能互相匹配，才能安全把 prefetch load_ex 替换成 DMA start。
  scf::ForOp kernel; // 新生成的 steady-state kernel loop。
  scf::IfOp prologue; // kernel 前面的首轮预取 if。
  scf::IfOp prefetch; // kernel body 内的下一轮预取 if。
  SmallVector<memref_ext::LoadExOp> prologueLoads; // prologue 中写 ping buffer 的首轮 load。
  SmallVector<memref_ext::LoadExOp> nextLoads; // prefetch 中写下一轮 buffer 的 load。
  SmallVector<memref_ext::StoreExOp> stores; // kernel body 中的写回 store。
};

bool extractSchedule(scf::ForOp kernel, PlannedExtSchedule &schedule) { // 从一个候选 kernel 中提取 planned schedule。
  if (!kernel->hasAttr(kPlanIdAttr)) // 没有 plan id 的 loop 不是 PlanRewriteExt 产物。
    return false; // 非适用对象，调用方会跳过。

  auto idAttr = kernel->getAttrOfType<IntegerAttr>(kPlanIdAttr); // 读取 plan id。
  // kernel loop 必须携带单个 i1 ping/pong 状态；否则无法选择 ping/pong DMA handle。
  if (!idAttr || kernel.getRegionIterArgs().size() != 1 ||
      !kernel.getRegionIterArgs().front().getType().isInteger(1))
    return false; // 结构不是当前 lowering 支持的 double-buffer kernel。
  int64_t id = idAttr.getInt(); // 保存当前 plan id，后续用它匹配 prologue。
  schedule.kernel = kernel; // 记录 kernel loop。

  for (Operation *prev = kernel->getPrevNode(); prev; prev = prev->getPrevNode()) { // 向前查找匹配的 prologue if。
    auto ifOp = dyn_cast<scf::IfOp>(prev); // prologue 由 PlanRewriteExt 生成为 scf.if。
    if (!ifOp || !ifOp->hasAttr(kPlanPrologueAttr)) // 不是 prologue 标记则继续向前找。
      continue; // 允许 kernel 前还有其他操作。
    auto prologueId = ifOp->getAttrOfType<IntegerAttr>(kPlanIdAttr); // 读取 prologue 的 plan id。
    // 必须按 plan id 匹配，而不是只靠位置；同一函数里可能有多个 planned loop。
    if (prologueId && prologueId.getInt() == id) {
      schedule.prologue = ifOp; // 找到与当前 kernel 成对的 prologue。
      break; // 一个 kernel 只消费一个 prologue。
    }
  }
  if (!schedule.prologue) // 找不到匹配 prologue 时计划不完整。
    return false; // 跳过该 kernel。

  for (Operation &op : kernel.getBody()->without_terminator()) { // 在 kernel body 中查找 prefetch if。
    auto ifOp = dyn_cast<scf::IfOp>(&op); // prefetch 也是 scf.if。
    // PlanRewriteExt 在 kernel body 靠前位置生成一个 db_plan_prefetch if。
    if (ifOp && ifOp->hasAttr(kPlanPrefetchAttr)) {
      schedule.prefetch = ifOp; // 记录下一轮预取区域。
      break; // 当前实现消费第一个标记的 prefetch 区域。
    }
  }
  if (!schedule.prefetch) // 没有 prefetch 区域就无法建立 steady-state DMA。
    return false; // 跳过该 kernel。

  collectRoleLoads(thenBlock(schedule.prologue), schedule.prologueLoads); // 收集首轮预取 load。
  collectRoleLoads(thenBlock(schedule.prefetch), schedule.nextLoads); // 收集下一轮预取 load。
  collectRoleStores(*kernel.getBody(), schedule.stores); // 收集写回 store，用于确认计划完整。

  // prologue loads 和 steady-state loads 按 tile 成对。store check 防止消费没有写回阶段的半成品计划。
  return !schedule.prologueLoads.empty() && !schedule.nextLoads.empty() &&
         schedule.prologueLoads.size() == schedule.nextLoads.size() &&
         !schedule.stores.empty();
}

LogicalResult lowerSchedule(IRRewriter &rewriter,
                            PlannedExtSchedule &schedule) { // 对一个完整 planned schedule 执行 DMA lowering。
  RewriterBase::InsertionGuard guard(rewriter); // 保护调用者插入点。
  Location loc = schedule.kernel.getLoc(); // 使用 kernel 位置作为新 handle/select/wait 的位置。

  rewriter.setInsertionPoint(schedule.prologue); // DMA handle 放在 prologue 前，供 prologue 和 kernel 同时使用。
  Value pingLoadHandle = createDMAHandle(loc, rewriter); // ping buffer 对应的 load DMA handle。
  Value pongLoadHandle = createDMAHandle(loc, rewriter); // pong buffer 对应的 load DMA handle。

  // prologue 总是填充初始 ping buffer；kernel 第一轮的 i1 状态也选择 ping 为 current。
  for (memref_ext::LoadExOp load : schedule.prologueLoads) // 逐个 lowering 首轮预取 load。
    if (failed(replaceLoadWithDMAStart(rewriter, load, pingLoadHandle))) // prologue load 绑定 ping handle。
      return failure(); // 任一 load lowering 失败则整个 schedule lowering 失败。

  Value current = schedule.kernel.getRegionIterArgs().front(); // kernel 的 i1 iter_arg 表示当前使用 ping 还是 pong。
  rewriter.setInsertionPoint(schedule.prefetch); // handle select 放在 prefetch if 前，供 prefetch 和 wait 使用。
  // currentLoadHandle 对应本轮 compute 要读取的 buffer；nextLoadHandle 对应本轮启动的下一轮预取 buffer。
  Value currentLoadHandle =
      arith::SelectOp::create(rewriter, loc, current, pingLoadHandle,
                              pongLoadHandle); // cur=true 选 ping，否则选 pong。
  Value nextLoadHandle =
      arith::SelectOp::create(rewriter, loc, current, pongLoadHandle,
                              pingLoadHandle); // 下一轮写入当前相反的 buffer。

  for (memref_ext::LoadExOp load : schedule.nextLoads) // lowering loop 内下一轮预取 load。
    if (failed(replaceLoadWithDMAStart(rewriter, load, nextLoadHandle))) // next prefetch 绑定 next handle。
      return failure(); // 任一 load lowering 失败则整个 schedule lowering 失败。

  // 等待当前 buffer 对应的 DMA 完成。刚启动的 next prefetch 会在下一轮变成 current 后再等待。
  rewriter.setInsertionPointAfter(schedule.prefetch); // wait 放在 prefetch if 后、compute 前。
  memref_ext::DmaWaitOp::create(rewriter, loc, currentLoadHandle); // 保证当前轮 compute 读到已完成的 DMA 数据。

  return success(); // 当前 schedule lowering 成功。
}

void cleanPlanAttrs(Operation *root) { // 清理 PlanRewriteExt 和本 pass 之间的私有属性。
  // DMA lowering 完成后，这些 marker 不再是公开 IR 契约，后续 pass 不应继续依赖它们。
  root->walk([](Operation *op) { // 遍历整个 function/module 子树。
    op->removeAttr(kPlanIdAttr); // 删除 plan id。
    op->removeAttr(kPlanPrologueAttr); // 删除 prologue 标记。
    op->removeAttr(kPlanPrefetchAttr); // 删除 prefetch 标记。
    op->removeAttr(kPlanComputeAttr); // 删除 compute 标记。
    op->removeAttr(kPlanCopyRoleAttr); // 删除 transfer role 标记。
  });
}

struct HexagonDoubleBufferDMALoweringExtPass // pass 实现类。
    : public ::impl::HexagonDoubleBufferDMALoweringExtBase<
          HexagonDoubleBufferDMALoweringExtPass> { // 继承 TableGen 生成的 pass 基类。
  void getDependentDialects(DialectRegistry &registry) const override { // 注册本 pass 会读取或生成的 dialect。
    registry.insert<arith::ArithDialect, memref::MemRefDialect, // arith 用于 guard/select，memref 保持依赖完整。
                    memref_ext::MemRefExtDialect, scf::SCFDialect>(); // memref_ext/scf 是主要 lowering 目标。
  }

  void runOnOperation() override { // pass 主入口。
    SmallVector<scf::ForOp> kernels; // 先收集候选 kernel，避免 walk 时改写 IR 影响遍历。
    getOperation()->walk([&](scf::ForOp forOp) { // 遍历当前函数内所有 scf.for。
      // 只有 PlanRewriteExt 生成的 loop 会带 plan id。
      if (!forOp->hasAttr(kPlanIdAttr)) // 普通 loop 不参与 lowering。
        return; // 继续 walk 其他 loop。
      kernels.push_back(forOp); // 保存 planned kernel 候选。
    });

    IRRewriter rewriter(getOperation()->getContext()); // 创建用于 IR 改写的 rewriter。
    for (scf::ForOp kernel : kernels) { // 按收集顺序处理每个候选 kernel。
      PlannedExtSchedule schedule; // 保存当前 kernel 对应的完整计划。
      // 缺失或不完整的 schedule 视为不适用；提取成功后 lowering 失败才是硬错误。
      if (!extractSchedule(kernel, schedule)) // 校验并提取 prologue/prefetch/load/store。
        continue; // 不完整计划保持原样。
      if (failed(lowerSchedule(rewriter, schedule))) { // 执行 load_ex 到 dma_start_ex 的 lowering。
        signalPassFailure(); // 已接受的计划 lowering 失败，通知 pass manager。
        return; // 停止处理后续 kernel。
      }
    }
    cleanPlanAttrs(getOperation()); // 所有可处理 schedule 完成后，清理私有 plan 属性。
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hexagon::createHexagonDoubleBufferDMALoweringExtPass() { // 创建 pass 实例供 pipeline 使用。
  return std::make_unique<HexagonDoubleBufferDMALoweringExtPass>(); // 交给 MLIR pass pipeline 管理生命周期。
}
