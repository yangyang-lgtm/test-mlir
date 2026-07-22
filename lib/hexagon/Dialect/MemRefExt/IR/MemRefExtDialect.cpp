//===- MemRefExtDialect.cpp ----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"

using namespace mlir;
using namespace mlir::memref_ext;

void MemRefExtDialect::initialize() {
  registerOperations();
  registerTypes();
}

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.cpp.inc"
