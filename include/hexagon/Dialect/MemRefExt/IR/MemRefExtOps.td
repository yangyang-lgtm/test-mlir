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
include "triton/Dialect/Triton/IR/TritonTypes.td"

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

  let arguments = (ins Arg<AnyType, "source memory object">:$source,
                       Arg<AnyType, "target memory object">:$target,
                       Index:$num_elements,
                       Arg<MemRefExt_DmaHandle, "DMA handle">:$handle);
  let hasVerifier = 1;

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

def MemRefExt_DmaStartExOp : MemRefExt_Op<"dma_load_ex", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Start an asynchronous bounded external load";

  let description = [{
    Asynchronously loads up to `valid_elements` elements from an external
    pointer-like value into `target`, using the same bounded-load operands as
    `memref_ext.load_ex`. The explicit DMA handle groups the asynchronous
    transfer for a later `memref_ext.dma_wait`.
  }];

  let arguments = (ins Arg<AnyType, "external pointer">:$ptr,
                       I32Attr:$tensor_size,
                       TT_Int:$valid_size,
                       TT_Type:$other,
                       Arg<AnyMemRef, "target memref">:$target,
                       OptionalAttr<BoolAttr>:$is_other_valid,
                       Arg<MemRefExt_DmaHandle, "DMA handle">:$handle);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `,` $valid_size `,` $other `,` $target `,` $handle attr-dict
    `:` type($ptr) `,` type($valid_size) `,` type($other) `,`
    type($target) `,` type($handle)
  }];
}

def MemRefExt_DmaBlockDescriptorLoadOp
    : MemRefExt_Op<"dma_block_descriptor_load", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Start an asynchronous descriptor block load";

  let description = [{
    Asynchronously loads a block described by a Triton tensor descriptor into a
    local memref tile. The explicit DMA handle groups the transfer for a later
    `memref_ext.dma_wait`.
  }];

  let arguments = (ins Arg<TT_AnyTensorDescType, "tensor descriptor">:$desc,
                       DefaultValuedAttr<DenseI32ArrayAttr,
                           "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
                       OptionalAttr<AnyAttr>:$padding,
                       Arg<AnyMemRef, "target memref">:$target,
                       Arg<MemRefExt_DmaHandle, "DMA handle">:$handle);

  let assemblyFormat = [{
    $desc `,` $target `,` $handle attr-dict
    `:` qualified(type($desc)) `,` type($target) `,` type($handle)
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

def MemRefExt_SelectMemrefOp : MemRefExt_Op<"select_memref", [Pure]> {
  let summary = "Select between two memref values";

  let description = [{
    Selects `true_value` when `condition` is true and `false_value` otherwise.
    Both memref operands and the result must have the same type.
  }];

  let arguments = (ins I1:$condition,
                       Arg<AnyMemRef, "value selected when condition is true">:$true_value,
                       Arg<AnyMemRef, "value selected when condition is false">:$false_value);
  let results = (outs AnyMemRef:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value attr-dict
    `:` type($true_value) `,` type($false_value) `->` type($result)
  }];
}

def MemRefExt_LoadExOp : MemRefExt_Op<"load_ex", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a bounded external pointer range into a memref tile";

  let description = [{
    Loads up to `valid_elements` elements from an external pointer-like value
    into `target`. Elements outside the valid range are filled with `padding`.
    The pointer operand is intentionally typed as AnyType so the op can carry
    Triton pointers before they are lowered to memrefs.
  }];

  let arguments = (ins Arg<AnyType, "external pointer">:$ptr,
                       I32Attr:$tensor_size,
                       TT_Int:$valid_size,
                       TT_Type:$other,
                       Arg<AnyMemRef, "target memref">:$target,
                       OptionalAttr<BoolAttr>:$is_other_valid);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `,` $valid_size `,` $other `,` $target attr-dict
    `:` type($ptr) `,` type($valid_size) `,` type($other) `,` type($target)
  }];
}

def MemRefExt_StoreExOp : MemRefExt_Op<"store_ex", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a memref tile into a bounded external pointer range";

  let description = [{
    Stores up to `valid_elements` elements from `source` to an external
    pointer-like value. The pointer operand is intentionally typed as AnyType so
    the op can carry Triton pointers before they are lowered to memrefs.
  }];

  let arguments = (ins Arg<AnyType, "external pointer">:$ptr,
                       Arg<AnyMemRef, "source memref">:$value,
                       TT_Int:$valid_size);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `,` $value `,` $valid_size attr-dict
    `:` type($ptr) `,` type($value) `,` type($valid_size)
  }];
}

def MemRefExt_BlockDescriptorLoadOp : MemRefExt_Op<"block_descriptor_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a 2D descriptor block into a memref tile";

  let description = [{
    Loads a block described by a Triton block descriptor into a local memref
    tile. The target memref shape is expected to match the descriptor block.
  }];

  let arguments = (ins Arg<TT_AnyTensorDescType, "tensor descriptor">:$desc,
                       DefaultValuedAttr<DenseI32ArrayAttr,
                           "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
                       OptionalAttr<AnyAttr>:$padding,
                       Arg<AnyMemRef, "target memref">:$target);

  let assemblyFormat = [{
    $desc `,` $target attr-dict `:` qualified(type($desc)) `,` type($target)
  }];
}

def MemRefExt_BlockDescriptorStoreOp : MemRefExt_Op<"block_descriptor_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a memref tile through a 2D descriptor block";

  let description = [{
    Stores a local memref tile into the block described by a Triton block
    descriptor. The source memref shape is expected to match the descriptor
    block.
  }];

  let arguments = (ins Arg<TT_AnyTensorDescType, "tensor descriptor">:$desc,
                       Arg<AnyMemRef, "source memref">:$value,
                       DefaultValuedAttr<DenseI32ArrayAttr,
                           "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck);

  let assemblyFormat = [{
    $desc `,` $value attr-dict `:` qualified(type($desc)) `,` type($value)
  }];
}

def MemRefExt_BlockDescLoadOp : MemRefExt_Op<"block_desc_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a legacy Triton block descriptor into a memref tile";

  let arguments = (ins Arg<TT_BlockDescType, "legacy block descriptor">:$desc,
                       DefaultValuedAttr<DenseI32ArrayAttr,
                           "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
                       OptionalAttr<AnyAttr>:$padding,
                       Arg<AnyMemRef, "target memref">:$target);

  let assemblyFormat = [{
    $desc `,` $target attr-dict `:` qualified(type($desc)) `,` type($target)
  }];
}

def MemRefExt_BlockDescStoreOp : MemRefExt_Op<"block_desc_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a memref tile through a legacy Triton block descriptor";

  let arguments = (ins Arg<TT_BlockDescType, "legacy block descriptor">:$desc,
                       Arg<AnyMemRef, "source memref">:$value,
                       DefaultValuedAttr<DenseI32ArrayAttr,
                           "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck);

  let assemblyFormat = [{
    $desc `,` $value attr-dict `:` qualified(type($desc)) `,` type($value)
  }];
}

#endif // HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_OPS_TD
