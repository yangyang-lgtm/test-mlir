//===- MemRefExtDialect.td ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_DIALECT_TD
#define HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_DIALECT_TD

include "mlir/IR/OpBase.td"

def MemRefExt_Dialect : Dialect {
  let name = "memref_ext";
  let cppNamespace = "::mlir::memref_ext";
  let summary = "MemRef extension operations";
  let description = [{
    The `memref_ext` dialect contains custom memref-like operations used by the
    Hexagon pipeline when upstream MemRef does not provide the needed operation.
  }];
  let extraClassDeclaration = [{
    void registerOperations();
    void registerTypes();
  }];
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_DIALECT_TD
