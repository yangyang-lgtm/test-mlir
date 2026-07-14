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

bool isTensorValue(Value value) { return isa<TensorType>(value.getType()); }

bool isTensorLoopCandidate(scf::ForOp forOp) {
  if (forOp->hasAttr(kTiledGenericAttr) || forOp->hasAttr("all_parallel"))
    return true;
  if (llvm::any_of(forOp.getInitArgs(), isTensorValue) ||
      llvm::any_of(forOp.getResults(), isTensorValue))
    return true;

  bool hasTensorBodyValue = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (llvm::any_of(op->getOperands(), isTensorValue) ||
        llvm::any_of(op->getResults(), isTensorValue)) {
      hasTensorBodyValue = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasTensorBodyValue;
}

bool valueDependsOn(Value root, Value needle, Block *body) {
  SmallVector<Value> worklist{root};
  llvm::SmallPtrSet<Value, 16> seen;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (value == needle)
      return true;
    if (!seen.insert(value).second)
      continue;

    Operation *def = value.getDefiningOp();
    if (!def || def->getBlock() != body)
      continue;
    worklist.append(def->operand_begin(), def->operand_end());
  }

  return false;
}

bool isLoopCarriedTensorReduction(scf::ForOp forOp) {
  if (forOp.getInitArgs().empty())
    return false;

  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Block *body = forOp.getBody();
  for (auto [index, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    if (!isTensorValue(initArg))
      continue;

    Value iterArg = forOp.getRegionIterArgs()[index];
    Value yielded = yield.getOperand(index);
    if (yielded != iterArg && valueDependsOn(yielded, iterArg, body))
      return true;
  }

  return false;
}

std::optional<StringLiteral> classifyForLoop(scf::ForOp forOp) {
  if (!isTensorLoopCandidate(forOp))
    return std::nullopt;
  if (isLoopCarriedTensorReduction(forOp))
    return kReduce;
  return kPointwise;
}

struct AnnotateForLoopKindPass
    : public ::impl::AnnotateForLoopKindBase<AnnotateForLoopKindPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    getOperation().walk([&](scf::ForOp forOp) {
      std::optional<StringLiteral> kind = classifyForLoop(forOp);
      if (!kind)
        return;
      forOp->setAttr(kLoopKindAttr,
                     StringAttr::get(forOp.getContext(), *kind));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createAnnotateForLoopKindPass() {
  return std::make_unique<AnnotateForLoopKindPass>();
}
