//===- MemRefExtTypes.td --------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_TYPES_TD
#define HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.td"

class MemRefExt_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MemRefExt_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MemRefExt_DmaHandleType : MemRefExt_Type<"DmaHandle", "dma_handle"> {
  let summary = "DMA runtime handle";
  let description = [{
    A token storage handle used to synchronize an asynchronous DMA transfer.
    The handle is an SSA value in memref_ext IR and lowers to runtime token
    storage in LLVM IR.
  }];
}

def MemRefExt_DmaHandle :
    DialectType<MemRefExt_Dialect,
                CPred<"::mlir::isa<::mlir::memref_ext::DmaHandleType>($_self)">,
                "DMA handle type",
                "::mlir::memref_ext::DmaHandleType">;

#endif // HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_TYPES_TD
