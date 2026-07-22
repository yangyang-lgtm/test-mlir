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

def MemRefExt_DoubleBufferDmaHandlesType
    : MemRefExt_Type<"DoubleBufferDmaHandles", "handles"> {
  let summary = "Double-buffered DMA handle pair";
  let description = [{
    A pair of DMA handles used by ping/pong double-buffered DMA schedules.
  }];

  let parameters = (ins "int":$numHandles);
  let assemblyFormat = "`<` $numHandles `>`";
}

def MemRefExt_DoubleBufferDmaHandles :
    DialectType<MemRefExt_Dialect,
                CPred<"::mlir::isa<::mlir::memref_ext::DoubleBufferDmaHandlesType>($_self)">,
                "double-buffered DMA handle pair type",
                "::mlir::memref_ext::DoubleBufferDmaHandlesType">;

def MemRefExt_DmaFetchDataType
    : MemRefExt_Type<"DmaFetchData", "dma_fetch_data"> {
  let summary = "DMA fetch data storage";
  let description = [{
    Per-transfer runtime fetch data used by asynchronous DMA operations.  The
    value must stay private to an in-flight transfer until that transfer is
    known to be synchronized.
  }];
}

def MemRefExt_DmaFetchData :
    DialectType<MemRefExt_Dialect,
                CPred<"::mlir::isa<::mlir::memref_ext::DmaFetchDataType>($_self)">,
                "DMA fetch data type",
                "::mlir::memref_ext::DmaFetchDataType">;

def MemRefExt_DoubleBufferDmaFetchDataType
    : MemRefExt_Type<"DoubleBufferDmaFetchData",
                     "fetch_datas"> {
  let summary = "Double-buffered DMA fetch data pair";
  let description = [{
    A pair of DMA fetch data values used by ping/pong double-buffered DMA
    schedules.
  }];

  let parameters = (ins "int":$numFetchData);
  let assemblyFormat = "`<` $numFetchData `>`";
}

def MemRefExt_DoubleBufferDmaFetchData :
    DialectType<MemRefExt_Dialect,
                CPred<"::mlir::isa<::mlir::memref_ext::DoubleBufferDmaFetchDataType>($_self)">,
                "double-buffered DMA fetch data pair type",
                "::mlir::memref_ext::DoubleBufferDmaFetchDataType">;

#endif // HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_TYPES_TD
