//===- DMATransferUtil.cpp - some useful general DMA functions ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Note : There is presently some overlap between here and HexagonMemToDMA.cpp
//        that will be resolved in a future refactoring.
//        Also, Rank > 2 DMA-Transfers currently generate IR level loops and
//       these need to be resolved as well.

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace hexagon {

namespace {

/// Returns true if `type` is a ranked memref that this helper can lower.
bool isSafe(Type type) {
  auto memRefType = dyn_cast<MemRefType>(type);
  return memRefType && memRefType.getRank() <= 2;
}

static Value getI32Const(Location loc, IRRewriter &rewriter, int val) {
  return arith::ConstantIndexOp::create(rewriter, loc, val);
}

Value getInnermostDim(Location loc, IRRewriter &rewriter, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  int64_t dim = type.getRank() - 1;
  if (type.isDynamicDim(dim))
    return memref::DimOp::create(rewriter, loc, memref, dim);
  return getI32Const(loc, rewriter, type.getDimSize(dim));
}

bool getStaticOuterStride(MemRefType type, int64_t &stride) {
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(type.getStridesAndOffset(strides, offset)) || type.getRank() <= 1)
    return false;
  if (ShapedType::isDynamic(strides[type.getRank() - 2]) ||
      strides[type.getRank() - 1] != 1)
    return false;
  int64_t innerDim = type.getDimSize(type.getRank() - 1);
  if (!ShapedType::isDynamic(innerDim) && strides[type.getRank() - 2] <= innerDim)
    return false;
  stride = strides[type.getRank() - 2];
  return true;
}
} // unnamed namespace

Value createNumElements(Location loc, IRRewriter &rewriter, Value view) {
  auto viewType = view.getType();
  assert(isSafe(viewType) && "num elements cannot be created for unsafe type");
  auto memrefType = cast<MemRefType>(viewType);
  Value numElements = getI32Const(loc, rewriter, 1);
  for (auto dim : llvm::enumerate(memrefType.getShape())) {
    Value dimValue;
    if (ShapedType::isDynamic(dim.value()))
      dimValue = memref::DimOp::create(rewriter, loc, view, dim.index());
    else
      dimValue = getI32Const(loc, rewriter, dim.value());
    numElements = arith::MulIOp::create(rewriter, loc, numElements, dimValue);
  }
  return numElements;
}

bool createDMAStartOp(Location loc, IRRewriter &rewriter, Value source,
                      Value target, Value tagAlloc, Operation **createdOp) {
  auto sourceType = source.getType();
  auto targetType = target.getType();

  if (!isSafe(sourceType) || !isSafe(targetType))
    return false;

  MemRefType sourceMemRefType = cast<MemRefType>(sourceType);
  MemRefType targetMemRefType = cast<MemRefType>(targetType);
  if (sourceMemRefType.getRank() != targetMemRefType.getRank())
    return false;
  int64_t rank = sourceMemRefType.getRank();

  // Extract strides and offsets for source and target memrefs
  SmallVector<int64_t, 4> sourceStrides, targetStrides;
  int64_t srcOffset, targetOffset;
  if (failed(sourceMemRefType.getStridesAndOffset(sourceStrides, srcOffset)) ||
      failed(targetMemRefType.getStridesAndOffset(targetStrides, targetOffset)))
    return false;

  int sourceMemSpace, targetMemSpace;
  // Unknown memory space - don't lower
  if (!(isMemorySpaceIntTypeOrDefault(sourceMemRefType, sourceMemSpace)) ||
      !(isMemorySpaceIntTypeOrDefault(targetMemRefType, targetMemSpace)))
    return false;

  Value numElements = createNumElements(loc, rewriter, source);
  if (rank == 1 || (isContiguousMemrefType(sourceMemRefType) &&
                    isContiguousMemrefType(targetMemRefType))) {
    OperationState state(loc, "memref_ext.dma_start");
    state.addOperands({source, target, numElements, tagAlloc});
    Operation *dmaStart = rewriter.create(state);
    if (createdOp)
      *createdOp = dmaStart;
    return true;
  }

  Value width;
  int64_t sourceStride = 0;
  int64_t targetStride = 0;
  bool sourceStrided = getStaticOuterStride(sourceMemRefType, sourceStride);
  bool targetStrided = getStaticOuterStride(targetMemRefType, targetStride);

  if (sourceStrided) {
    width = getInnermostDim(loc, rewriter, source);
  } else if (targetStrided) {
    width = getInnermostDim(loc, rewriter, target);
  } else {
    return false;
  }

  OperationState state(loc, "memref_ext.dma_2d_start");
  Value srcStride = sourceStrided ? getI32Const(loc, rewriter, sourceStride)
                                  : width;
  Value dstStride = targetStrided ? getI32Const(loc, rewriter, targetStride)
                                  : width;
  state.addOperands(
      {source, target, numElements, tagAlloc, srcStride, dstStride, width});
  Operation *dmaStart = rewriter.create(state);
  if (createdOp)
    *createdOp = dmaStart;
  return true;
}

} // namespace hexagon
} // namespace mlir
