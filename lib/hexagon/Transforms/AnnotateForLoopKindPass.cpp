//===- AnnotateForLoopKindPass.cpp ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "hexagon/Common/Common.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ANNOTATEFORLOOPKIND
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// 判断一个 SSA value 的类型是不是 tensor，用来识别仍处在 tensor 语义上的循环。
bool isTensorValue(Value value) { return isa<TensorType>(value.getType()); }

// 判断 scf.for 是否应该被当成 tensor 相关循环继续分类。
bool isTensorLoopCandidate(scf::ForOp forOp) {
  // 已经带有 tiled generic 或 all_parallel 标记的循环，直接认为是候选循环。
  if (forOp->hasAttr(kTiledGenericAttr) || forOp->hasAttr("all_parallel"))
    return true;
  // 如果循环的 iter_args 或结果里出现 tensor，说明循环携带 tensor 数据流。
  if (llvm::any_of(forOp.getInitArgs(), isTensorValue) ||
      llvm::any_of(forOp.getResults(), isTensorValue))
    return true;

  // 继续检查循环体内部是否有任意操作读写 tensor 值。
  bool hasTensorBodyValue = false;
  forOp.getBody()->walk([&](Operation *op) {
    // 只要操作数或结果中出现 tensor，就可以停止遍历。
    if (llvm::any_of(op->getOperands(), isTensorValue) ||
        llvm::any_of(op->getResults(), isTensorValue)) {
      hasTensorBodyValue = true;
      return WalkResult::interrupt();
    }
    // 当前操作不涉及 tensor，继续检查下一个操作。
    return WalkResult::advance();
  });
  // 返回循环体里是否发现 tensor 相关值。
  return hasTensorBodyValue;
}

// 从 root 沿着定义操作的操作数反向搜索，判断它是否依赖 needle。
bool valueDependsOn(Value root, Value needle, Block *body) {
  // worklist 保存待检查的值，初始从 root 开始。
  SmallVector<Value> worklist{root};
  // seen 避免重复访问同一个 SSA value。
  llvm::SmallPtrSet<Value, 16> seen;

  // 迭代处理所有可达的依赖值。
  while (!worklist.empty()) {
    // 取出一个待检查值。
    Value value = worklist.pop_back_val();
    // 找到目标值，说明 root 依赖 needle。
    if (value == needle)
      return true;
    // 已访问过的值不再重复展开。
    if (!seen.insert(value).second)
      continue;

    // 只追踪在当前循环体内定义的值，避免跨作用域误判。
    Operation *def = value.getDefiningOp();
    if (!def || def->getBlock() != body)
      continue;
    // 将定义操作的所有操作数加入反向依赖搜索。
    worklist.append(def->operand_begin(), def->operand_end());
  }

  // 搜索完仍未命中，说明没有找到依赖关系。
  return false;
}

// 判断循环是否存在 loop-carried tensor reduction。
bool isLoopCarriedTensorReduction(scf::ForOp forOp) {
  // 没有 iter_args 的循环不可能形成 loop-carried reduction。
  if (forOp.getInitArgs().empty())
    return false;

  // scf.for 的终结符必须是 scf.yield，这里读取每个迭代参数的回传值。
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // 后续依赖分析只在循环体 block 内展开。
  Block *body = forOp.getBody();
  // 对每个 init_arg 检查它对应的 region iter_arg 和 yield operand。
  for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    // 只关心 tensor 类型的循环携带值。
    if (!isTensorValue(initArg))
      continue;

    // iterArg 是循环体内代表上一轮结果的 block argument。
    Value iterArg = forOp.getRegionIterArgs()[index];
    // yielded 是本轮迭代 yield 回下一轮或循环结果的值。
    Value yielded = yield.getOperand(index);
    // 如果 yield 的值不是原样回传，并且依赖 iterArg，就视为 reduction。
    if (yielded != iterArg && valueDependsOn(yielded, iterArg, body))
      return true;
  }

  // 所有 tensor iter_arg 都没有形成回环依赖，说明不是 reduction。
  return false;
}

// 给候选 scf.for 分类：reduction 或 pointwise；非候选循环不标注。
std::optional<StringLiteral> classifyForLoop(scf::ForOp forOp) {
  // 没有 tensor 语义的循环跳过。
  if (!isTensorLoopCandidate(forOp))
    return std::nullopt;
  // loop-carried tensor 依赖归类为 reduce。
  if (isLoopCarriedTensorReduction(forOp))
    return kReduce;
  // 其它 tensor 候选循环默认归类为 pointwise。
  return kPointwise;
}

// Pass 主体：遍历函数里的 scf.for 并写入循环类型属性。
struct AnnotateForLoopKindPass
    : public ::impl::AnnotateForLoopKindBase<AnnotateForLoopKindPass> {
  // 声明本 pass 会创建或依赖 SCF dialect 中的操作。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  // 在当前 func::FuncOp 上执行循环标注。
  void runOnOperation() override {
    // 遍历函数内所有 scf.for。
    getOperation().walk([&](scf::ForOp forOp) {
      // 对当前循环做类型分类。
      std::optional<StringLiteral> kind = classifyForLoop(forOp);
      // 分类失败表示该循环不是目标循环，保持原样。
      if (!kind)
        return;
      // 将分类结果写成 kLoopKindAttr，供后续 double-buffer pass 使用。
      forOp->setAttr(kLoopKindAttr,
                     StringAttr::get(forOp.getContext(), *kind));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createAnnotateForLoopKindPass() {
  // 返回 pass 实例，供 pipeline 注册和创建。
  return std::make_unique<AnnotateForLoopKindPass>();
}
