//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGEXPERIMENTAL_H
#define TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGEXPERIMENTAL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_TRITONTOLINALGEXPERIMENTAL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
#undef GEN_PASS_DECL_TRITONTOLINALGEXPERIMENTAL

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgExperimentalPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGEXPERIMENTAL_H
