//===- MemRefExtOps.cpp ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::memref_ext;

void MemRefExtDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"
