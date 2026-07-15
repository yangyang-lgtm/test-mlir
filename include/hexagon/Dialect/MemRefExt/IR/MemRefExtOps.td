//===- MemRefExtOps.td ----------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_OPS_TD
#define HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_OPS_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.td"
include "hexagon/Dialect/MemRefExt/IR/MemRefExtTypes.td"

class MemRefExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MemRefExt_Dialect, mnemonic, traits> {
}

def MemRefExt_DmaHandleOp : MemRefExt_Op<"dma_handle"> {
  let summary = "Create a DMA handle";

  let description = [{
    Creates a DMA handle SSA value. The handle is used by dma_start and
    dma_wait to group asynchronous DMA transfers that must be synchronized
    together. Independent handles keep ping and pong synchronization state
    independent.
  }];

  let results = (outs MemRefExt_DmaHandle:$handle);

  let assemblyFormat = "attr-dict `:` type($handle)";
}

def MemRefExt_DmaStartOp : MemRefExt_Op<"dma_start", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Start a DMA transfer using an explicit DMA handle";

  let description = [{
    Starts an asynchronous contiguous DMA transfer and updates the runtime DMA
    state associated with the given DmaHandle. Multiple starts may use the same
    handle and a later dma_wait on that handle synchronizes the grouped
    transfers. This is intentionally simpler than
    `memref.dma_start`: the transfer starts at zero indices of source and
    target, and the handle is passed directly so ping/pong phases can use
    independent DmaHandle values.

    `num_elements` is in elements.
  }];

  let arguments = (ins Arg<AnyMemRef, "source memref">:$source,
                       Arg<AnyMemRef, "target memref">:$target,
                       Index:$num_elements,
                       Arg<MemRefExt_DmaHandle, "DMA handle">:$handle);

  let assemblyFormat = [{
    $source `,` $target `,` $num_elements `,` $handle attr-dict
    `:` type($source) `to` type($target) `,` type($handle)
  }];
}

def MemRefExt_Dma2DStartOp : MemRefExt_Op<"dma_2d_start", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Start a 2D DMA transfer using an explicit DMA handle";

  let description = [{
    Starts an asynchronous 2D DMA transfer and updates the runtime DMA state
    associated with the given DmaHandle. Unlike `memref.dma_start`, this op
    explicitly carries both source and target row strides, so it can represent
    copies where either side is pitched and ping/pong phases can use independent
    DmaHandle values.

    `num_elements`, `src_stride`, `dst_stride`, and `width` are all in elements.
    The transfer starts at zero indices of source and target.
  }];

  let arguments = (ins Arg<AnyMemRef, "source memref">:$source,
                       Arg<AnyMemRef, "target memref">:$target,
                       Index:$num_elements,
                       Arg<MemRefExt_DmaHandle, "DMA handle">:$handle,
                       Index:$src_stride,
                       Index:$dst_stride,
                       Index:$width);

  let assemblyFormat = [{
    $source `,` $target `,` $num_elements `,` $handle `,`
    $src_stride `,` $dst_stride `,` $width attr-dict
    `:` type($source) `to` type($target) `,` type($handle)
  }];
}

def MemRefExt_DmaWaitOp : MemRefExt_Op<"dma_wait", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Wait for a DMA handle";

  let description = [{
    Waits for the DMA transfers grouped by the given DmaHandle.
  }];

  let arguments = (ins Arg<MemRefExt_DmaHandle, "DMA handle">:$handle);

  let assemblyFormat = "$handle attr-dict `:` type($handle)";
}

#endif // HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_OPS_TD
