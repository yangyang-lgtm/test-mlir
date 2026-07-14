//===- MemRefExtTypes.cpp -------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::memref_ext;

void MemRefExtDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtTypes.cpp.inc"
