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

  // Create zero index for tag access
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value, 1> tagIndex = {zero};

  SmallVector<Value, 8> zeroIndices(rank, zero);
  SmallVector<Value, 8> tileIndices(rank, zero);
  SmallVector<Value, 2> zeroIndex(1, zero);

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

    auto dmaStart =
        memref::DmaStartOp::create(rewriter, loc, source, zeroIndices, target,
                                   zeroIndices, numElements, tagAlloc, zeroIndex);
    if (createdOp)
      *createdOp = dmaStart.getOperation();
    return true;
  }

  int64_t stride, width;
  assert(isStridedMultiDimMemrefType(sourceMemRefType, stride, width) ||
         isStridedMultiDimMemrefType(targetMemRefType, stride, width));
  auto dmaStart = memref::DmaStartOp::create(
      rewriter, loc, source, zeroIndices, target, zeroIndices, numElements,
      tagAlloc, zeroIndex, getI32Const(loc, rewriter, stride),
      getI32Const(loc, rewriter, width));
  if (createdOp)
    *createdOp = dmaStart.getOperation();
  return true;
}

} // namespace hexagon
} // namespace mlir
