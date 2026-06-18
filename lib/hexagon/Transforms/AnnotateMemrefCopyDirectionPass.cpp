//===- AnnotateMemrefCopyDirectionPass.cpp -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "triton-shared/Conversion/MemorySpaces.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_ANNOTATEMEMREFCOPYDIRECTION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

std::optional<StringRef> classifyMemorySpace(BaseMemRefType type) {
  int64_t memorySpace = type.getMemorySpaceAsInt();
  if (memorySpace == triton::kGlobalMemorySpace)
    return "global";
  if (memorySpace == triton::kSharedMemorySpace)
    return "shared";
  return std::nullopt;
}

struct AnnotateMemrefCopyDirectionPass
    : public ::impl::AnnotateMemrefCopyDirectionBase<
          AnnotateMemrefCopyDirectionPass> {
  void runOnOperation() override {
    getOperation().walk([](memref::CopyOp copyOp) {
      auto sourceType = dyn_cast<BaseMemRefType>(copyOp.getSource().getType());
      auto targetType = dyn_cast<BaseMemRefType>(copyOp.getTarget().getType());
      if (!sourceType || !targetType)
        return;

      std::optional<StringRef> source = classifyMemorySpace(sourceType);
      std::optional<StringRef> target = classifyMemorySpace(targetType);
      if (!source || !target) {
        copyOp->removeAttr("copy_direction");
        return;
      }

      std::string direction =
          (llvm::Twine(*source) + "_to_" + *target).str();
      copyOp->setAttr("copy_direction",
                      StringAttr::get(copyOp.getContext(), direction));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createAnnotateMemrefCopyDirectionPass() {
  return std::make_unique<AnnotateMemrefCopyDirectionPass>();
}
