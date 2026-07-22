//===- DMAToLLVMPass.cpp - Lowering DMA ops to LLVM -----------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert DMA dialect to LLVM dialect.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"
#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "dma-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DMATOLLVM
#include "hexagon/Conversion/DMAToLLVM/Passes.h.inc"
#undef GEN_PASS_DEF_DMATOLLVM

namespace {

// Function to get an LLVM pointer type
static LLVM::LLVMPointerType getPtrTy(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

// Function to get an LLVM void type
static LLVM::LLVMVoidType getVoidTy(MLIRContext *context) {
  return LLVM::LLVMVoidType::get(context);
}

// Function to create an LLVM constant of i32 type
static LLVM::ConstantOp getI32Constant(ConversionPatternRewriter &rewriter,
                                       Location loc, int64_t val) {
  // Get the context from the rewriter
  MLIRContext *context = rewriter.getContext();
  Type paramTy = rewriter.getI32Type();
  // Create and return an LLVM constant operation with the given value
  return LLVM::ConstantOp::create(rewriter, loc, paramTy, val);
}

// Function to truncate a value to i32 type
static Value truncToI32(ConversionPatternRewriter &rewriter, Location loc,
                        Value val) {
  // Ensure the value is of an integer type
  assert(llvm::isa<IntegerType>(val.getType()) && "Expected an integer type");
  MLIRContext *context = rewriter.getContext();
  // Create and return a truncation operation to i32 type
  return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), val);
}

//===----------------------------------------------------------------------===//
// Lower DMAStart
//===----------------------------------------------------------------------===//

// Structure to lower DMA start operation to LLVM pattern
struct LowerDMAStart : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerDMA2DStart : public ConvertToLLVMPattern {
  LowerDMA2DStart(MLIRContext *context,
                  const LLVMTypeConverter &typeConverter,
                  PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.dma_2d_start", context, typeConverter,
                             benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerDMAExtStart : public ConvertToLLVMPattern {
  LowerDMAExtStart(MLIRContext *context,
                   const LLVMTypeConverter &typeConverter,
                   PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.dma_start", context, typeConverter,
                             benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerDMAHandle : public ConvertToLLVMPattern {
  LowerDMAHandle(MLIRContext *context, const LLVMTypeConverter &typeConverter,
                 PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.dma_handle", context, typeConverter,
                             benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerCreateAndInitHandles : public ConvertToLLVMPattern {
  LowerCreateAndInitHandles(MLIRContext *context,
                            const LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.create_and_init_handles", context,
                             typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerDMAFetchData : public ConvertToLLVMPattern {
  LowerDMAFetchData(MLIRContext *context,
                    const LLVMTypeConverter &typeConverter,
                    PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.dma_fetch_data", context,
                             typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerCreateDoubleBufferFetchData : public ConvertToLLVMPattern {
  LowerCreateDoubleBufferFetchData(MLIRContext *context,
                                   const LLVMTypeConverter &typeConverter,
                                   PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.create_fetch_data", context,
                             typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerDMAExtWait : public ConvertToLLVMPattern {
  LowerDMAExtWait(MLIRContext *context, const LLVMTypeConverter &typeConverter,
                  PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.dma_wait", context, typeConverter,
                             benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerSelectDMAHandle : public ConvertToLLVMPattern {
  LowerSelectDMAHandle(MLIRContext *context,
                       const LLVMTypeConverter &typeConverter,
                       PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.select_dma_handle", context,
                             typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

struct LowerSelectDMAFetchData : public ConvertToLLVMPattern {
  LowerSelectDMAFetchData(MLIRContext *context,
                          const LLVMTypeConverter &typeConverter,
                          PatternBenefit benefit = 1)
      : ConvertToLLVMPattern("memref_ext.select_dma_fetch_data", context,
                             typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

// Function to get or create the DMA start function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMAStartFn(ModuleOp module, StringRef fnName,
              ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {getPtrTy(context), rewriter.getI32Type(),
                                 getPtrTy(context), rewriter.getI32Type(),
                                 rewriter.getI32Type(), rewriter.getI32Type(),
                                 rewriter.getI32Type(), getPtrTy(context)},
                                rewriter.getI32Type());
}

// Function to get or create the 2D DMA start function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMA2DStartFn(ModuleOp module, StringRef fnName,
                ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(
      rewriter, module, fnName,
      {getPtrTy(context), rewriter.getI32Type(), getPtrTy(context),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
      getPtrTy(context)},
      rewriter.getI32Type());
}

static FailureOr<LLVM::LLVMFuncOp>
getDMAWaitFn(ModuleOp module, StringRef fnName,
             ConversionPatternRewriter &rewriter);

// Function to convert a value to i32 type
Value convertToI32Type(mlir::Location loc, Value value,
                       ConversionPatternRewriter &rewriter,
                       mlir::MLIRContext *context) {
  // Get the type of the value
  mlir::Type type = value.getType();
  assert(mlir::LLVM::isCompatibleType(type));
  return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), value);
}

// Function to cast a pointer to a different address space
static Value AddrSpaceCast(mlir::Location loc, Value inputPtr,
                           ConversionPatternRewriter &rewriter,
                           mlir::MLIRContext *context) {
  auto i32Type = rewriter.getI32Type();
  auto asI32 = LLVM::PtrToIntOp::create(rewriter, loc, i32Type, inputPtr);
  return LLVM::IntToPtrOp::create(rewriter, loc, getPtrTy(context),
                                  asI32->getResult(0));
}

// Function to create DMAStart operation
LogicalResult
LowerDMAStart::matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  // Get the context and location of the operation
  auto context = op->getContext();
  auto loc = op->getLoc();

  // Get the parent module of the operation
  auto module = op->getParentOfType<ModuleOp>();

  // Get the name of the DMA start function
  auto dmaStartFnName = mlir::hexagon::getDMA2DStartFnName();
  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMA2DStartFn(module, dmaStartFnName, rewriter);
  if (failed(funcOp))
    return failure();

  auto isVTCM = [](Value memref) {
    assert(llvm::isa<MemRefType>(memref.getType()) && "Unexpected type passed");
    auto memrefType = cast<MemRefType>(memref.getType());
    return isInVTCMAddressSpace(memrefType);
  };

  // Get the LLVM pointer to the memory reference
  auto getMemRefPtr = [this, loc,
                       context](Value memref, int rank, signed offset,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) {
    auto memrefType = cast<MemRefType>(memref.getType());
    auto llvmMemref = *(adaptor.getOperands().begin() + offset);
    ValueRange indices = {
        std::next(adaptor.getOperands().begin(), offset + 1),
        std::next(adaptor.getOperands().begin(), offset + rank + 1)};
    auto llvmPtr =
        getStridedElementPtr(rewriter, loc, memrefType, llvmMemref, indices);
    return llvmPtr;
  };

  // Get the source memory reference and its rank
  auto srcMemref = op.getSrcMemRef();
  auto srcRank = op.getSrcMemRefRank();
  auto tagRank = op.getTagMemRefRank();
  // Get the LLVM pointer to the source memory reference
  auto llvmSrcPtr = getMemRefPtr(srcMemref, srcRank, 0, adaptor, rewriter);

  // Get the destination memory reference and its rank
  auto dstMemref = op.getDstMemRef();
  auto dstRank = op.getDstMemRefRank();
  // Get the LLVM pointer to the destination memory reference
  auto llvmDstPtr =
      getMemRefPtr(dstMemref, dstRank, 1 + srcRank, adaptor, rewriter);

  // Get the tag memory reference and its LLVM pointer
  // This is used to synchronize between dma_start and dma_wait operation
  auto tagMemref = op.getTagMemRef();
  auto llvmTagPtr = getMemRefPtr(tagMemref, op.getTagMemRefRank(),
                                 3 + srcRank + dstRank, adaptor, rewriter);

  // Convert the number of elements to an LLVM integer type
  auto numElements = convertToI32Type(
      loc, *(adaptor.getOperands().begin() + 2 + srcRank + dstRank), rewriter,
      context);
  // Get the size of the element type in bytes
  auto typeSizeInBytes = getI32Constant(rewriter, loc,
                                        cast<MemRefType>(srcMemref.getType())
                                                .getElementType()
                                                .getIntOrFloatBitWidth() /
                                            8);
  // Calculate the transfer size
  auto transferSize =
      LLVM::MulOp::create(rewriter, loc, numElements, typeSizeInBytes);

  // Get the source and destination memory spaces
  auto srcMemSpace = getI32Constant(rewriter, loc, isVTCM(srcMemref) ? 1 : 0);
  auto dstMemSpace = getI32Constant(rewriter, loc, isVTCM(dstMemref) ? 1 : 0);
  // Set the bypass cache flag to 0
  auto bypassCache = getI32Constant(rewriter, loc, 0);
  SmallVector<Value, 16> operands;

  if (op.isStrided()) {
    Value width = convertToI32Type(
        loc, *(adaptor.getOperands().begin() + srcRank + dstRank + tagRank + 5),
        rewriter, context);
    Value widthInBytes =
        LLVM::MulOp::create(rewriter, loc, width, typeSizeInBytes);
    Value height =
        (LLVM::SDivOp::create(rewriter, loc, transferSize, widthInBytes))
            ->getResult(0);
    // Stride arg is optional, and if present, it is only slow address space.
    // source stride iff stride is present, and dest is faster.
    // If no stride, use stride = width
    auto srcStride =
        (isVTCM(dstMemref) && (!(isVTCM(srcMemref))))
            ? convertToI32Type(loc,
                               (*(adaptor.getOperands().begin() + srcRank +
                                  dstRank + tagRank + 4)),
                               rewriter, context)
            : width;
    auto srcStrideInBytes =
        LLVM::MulOp::create(rewriter, loc, srcStride, typeSizeInBytes);
    // dest stride iff stride is present, and src is faster.
    auto dstStride =
        (isVTCM(srcMemref) && (!(isVTCM(dstMemref))))
            ? convertToI32Type(loc,
                               (*(adaptor.getOperands().begin() + srcRank +
                                  dstRank + tagRank + 4)),
                               rewriter, context)
            : width;
    auto dstStrideInBytes =
        LLVM::MulOp::create(rewriter, loc, dstStride, typeSizeInBytes);

    // Create a vector of operands for the DMA copy operation
    operands = {llvmSrcPtr,
                srcMemSpace,
                llvmDstPtr,
                dstMemSpace,
                widthInBytes,
                height,
                srcStrideInBytes,
                dstStrideInBytes,
                bypassCache,
                bypassCache,
                getI32Constant(rewriter, loc, 0), // isOrdered
                getI32Constant(rewriter, loc, 0), // Cache Allocation Policy
                llvmTagPtr};
  } else {
    dmaStartFnName = mlir::hexagon::getDMAStartFnName();
    funcOp = getDMAStartFn(module, dmaStartFnName, rewriter);
    operands = {llvmSrcPtr,   srcMemSpace, llvmDstPtr,  dstMemSpace,
                transferSize, bypassCache, bypassCache, llvmTagPtr};
  }
  // Erase the original operation
  rewriter.eraseOp(op);

  // Create the DMA copy operation
  auto dmaCopyOp =
      LLVM::CallOp::create(rewriter, loc, funcOp.value(), operands);

  // Get the token from the DMA copy operation
  auto token = *dmaCopyOp.getODSResults(0).begin();

  // Store the token in the tag memory reference
  LLVM::StoreOp::create(rewriter, loc, token, llvmTagPtr);

  return success();
}

LogicalResult LowerDMAExtStart::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *context = op->getContext();
  Location loc = op->getLoc();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (op->getNumOperands() != 5 || operands.size() != 5)
    return failure();

  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMAStartFn(module, mlir::hexagon::getDMAStartFnName(), rewriter);
  if (failed(funcOp))
    return failure();

  auto isVTCM = [](Value memref) {
    auto memrefType = cast<MemRefType>(memref.getType());
    return isInVTCMAddressSpace(memrefType);
  };

  auto getMemRefPtr = [this, loc](Value memref, Value llvmMemref,
                                  ConversionPatternRewriter &rewriter) {
    auto memrefType = cast<MemRefType>(memref.getType());
    SmallVector<Value> indices;
    for (unsigned i = 0; i < memrefType.getRank(); ++i)
      indices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, getIndexType(), 0));
    return getStridedElementPtr(rewriter, loc, memrefType, llvmMemref,
                                indices);
  };

  Value source = op->getOperand(0);
  Value target = op->getOperand(1);
  Value llvmSrcPtr = getMemRefPtr(source, operands[0], rewriter);
  Value llvmDstPtr = getMemRefPtr(target, operands[1], rewriter);
  Value llvmHandlePtr = operands[3];

  Value numElements = convertToI32Type(loc, operands[2], rewriter, context);
  Value typeSizeInBytes = getI32Constant(
      rewriter, loc,
      cast<MemRefType>(source.getType())
              .getElementType()
              .getIntOrFloatBitWidth() /
          8);
  Value transferSize =
      LLVM::MulOp::create(rewriter, loc, numElements, typeSizeInBytes);

  SmallVector<Value, 8> callOperands = {
      llvmSrcPtr,
      getI32Constant(rewriter, loc, isVTCM(source) ? 1 : 0),
      llvmDstPtr,
      getI32Constant(rewriter, loc, isVTCM(target) ? 1 : 0),
      transferSize,
      getI32Constant(rewriter, loc, 0),
      getI32Constant(rewriter, loc, 0),
      llvmHandlePtr};

  rewriter.eraseOp(op);
  auto dmaCopyOp =
      LLVM::CallOp::create(rewriter, loc, funcOp.value(), callOperands);
  Value token = *dmaCopyOp.getODSResults(0).begin();
  LLVM::StoreOp::create(rewriter, loc, token, llvmHandlePtr);
  return success();
}

LogicalResult LowerDMA2DStart::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *context = op->getContext();
  Location loc = op->getLoc();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (op->getNumOperands() != 8 || operands.size() != 8)
    return failure();

  FailureOr<LLVM::LLVMFuncOp> funcOp = getDMA2DStartFn(
      module, mlir::hexagon::getDMA2DStartFnName(), rewriter);
  if (failed(funcOp))
    return failure();

  auto isVTCM = [](Value memref) {
    auto memrefType = cast<MemRefType>(memref.getType());
    return isInVTCMAddressSpace(memrefType);
  };

  auto getMemRefPtr = [this, loc](Value memref, Value llvmMemref,
                                  ConversionPatternRewriter &rewriter) {
    auto memrefType = cast<MemRefType>(memref.getType());
    SmallVector<Value> indices;
    for (unsigned i = 0; i < memrefType.getRank(); ++i)
      indices.push_back(LLVM::ConstantOp::create(
          rewriter, loc, getIndexType(), 0));
    return getStridedElementPtr(rewriter, loc, memrefType, llvmMemref,
                                indices);
  };

  Value source = op->getOperand(0);
  Value target = op->getOperand(1);
  Value llvmSrcPtr = getMemRefPtr(source, operands[0], rewriter);
  Value llvmDstPtr = getMemRefPtr(target, operands[1], rewriter);
  Value llvmHandlePtr = operands[3];

  Value numElements = convertToI32Type(loc, operands[2], rewriter, context);
  Value typeSizeInBytes = getI32Constant(
      rewriter, loc,
      cast<MemRefType>(source.getType())
              .getElementType()
              .getIntOrFloatBitWidth() /
          8);
  Value width = convertToI32Type(loc, operands[7], rewriter, context);
  Value widthInBytes =
      LLVM::MulOp::create(rewriter, loc, width, typeSizeInBytes);
  Value height = LLVM::SDivOp::create(
      rewriter, loc,
      LLVM::MulOp::create(rewriter, loc, numElements, typeSizeInBytes),
      widthInBytes);
  Value srcStride = convertToI32Type(loc, operands[5], rewriter, context);
  Value dstStride = convertToI32Type(loc, operands[6], rewriter, context);
  Value srcStrideInBytes =
      LLVM::MulOp::create(rewriter, loc, srcStride, typeSizeInBytes);
  Value dstStrideInBytes =
      LLVM::MulOp::create(rewriter, loc, dstStride, typeSizeInBytes);

  SmallVector<Value, 16> callOperands = {
      llvmSrcPtr,
      getI32Constant(rewriter, loc, isVTCM(source) ? 1 : 0),
      llvmDstPtr,
      getI32Constant(rewriter, loc, isVTCM(target) ? 1 : 0),
      widthInBytes,
      height,
      srcStrideInBytes,
      dstStrideInBytes,
      getI32Constant(rewriter, loc, 0),
      getI32Constant(rewriter, loc, 0),
      getI32Constant(rewriter, loc, 0),
      getI32Constant(rewriter, loc, 0),
      llvmHandlePtr};

  rewriter.eraseOp(op);
  auto dmaCopyOp =
      LLVM::CallOp::create(rewriter, loc, funcOp.value(), callOperands);
  Value token = *dmaCopyOp.getODSResults(0).begin();
  LLVM::StoreOp::create(rewriter, loc, token, llvmHandlePtr);
  return success();
}

LogicalResult LowerDMAHandle::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                      rewriter.getIndexAttr(1));
  auto handle = LLVM::AllocaOp::create(rewriter, loc, getPtrTy(context),
                                       rewriter.getI32Type(), one);
  rewriter.replaceOp(op, handle.getResult());
  return success();
}

LogicalResult LowerCreateAndInitHandles::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto handlesType =
      cast<memref_ext::DoubleBufferDmaHandlesType>(op->getResult(0).getType());
  if (handlesType.getNumHandles() != 2)
    return failure();
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Type ptrType = getPtrTy(context);
  auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                      rewriter.getIndexAttr(1));
  Value ping = LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                      rewriter.getI32Type(), one);
  Value pong = LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                      rewriter.getI32Type(), one);
  auto llvmHandlesType =
      LLVM::LLVMStructType::getLiteral(context, {ptrType, ptrType});
  Value handles = LLVM::UndefOp::create(rewriter, loc, llvmHandlesType);
  handles = LLVM::InsertValueOp::create(rewriter, loc, handles, ping,
                                        ArrayRef<int64_t>{0});
  handles = LLVM::InsertValueOp::create(rewriter, loc, handles, pong,
                                        ArrayRef<int64_t>{1});
  rewriter.replaceOp(op, handles);
  return success();
}

LogicalResult LowerDMAFetchData::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                      rewriter.getIndexAttr(1));
  auto fetchData = LLVM::AllocaOp::create(rewriter, loc, getPtrTy(context),
                                          rewriter.getI32Type(), one);
  rewriter.replaceOp(op, fetchData.getResult());
  return success();
}

LogicalResult LowerCreateDoubleBufferFetchData::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto fetchDataType = cast<memref_ext::DoubleBufferDmaFetchDataType>(
      op->getResult(0).getType());
  if (fetchDataType.getNumFetchData() != 2)
    return failure();
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  Type ptrType = getPtrTy(context);
  auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                      rewriter.getIndexAttr(1));
  Value ping = LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                      rewriter.getI32Type(), one);
  Value pong = LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                      rewriter.getI32Type(), one);
  auto llvmFetchDataType =
      LLVM::LLVMStructType::getLiteral(context, {ptrType, ptrType});
  Value fetchData = LLVM::UndefOp::create(rewriter, loc, llvmFetchDataType);
  fetchData = LLVM::InsertValueOp::create(rewriter, loc, fetchData, ping,
                                          ArrayRef<int64_t>{0});
  fetchData = LLVM::InsertValueOp::create(rewriter, loc, fetchData, pong,
                                          ArrayRef<int64_t>{1});
  rewriter.replaceOp(op, fetchData);
  return success();
}

LogicalResult LowerDMAExtWait::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (op->getNumOperands() != 1 || operands.size() != 1)
    return failure();

  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMAWaitFn(module, mlir::hexagon::getDMAWaitFnName(), rewriter);
  if (failed(funcOp))
    return failure();

  Value token =
      LLVM::LoadOp::create(rewriter, loc, rewriter.getI32Type(), operands[0]);
  rewriter.eraseOp(op);
  LLVM::CallOp::create(rewriter, loc, funcOp.value(), ValueRange{token});
  return success();
}

LogicalResult LowerSelectDMAHandle::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (op->getNumOperands() != 2 || operands.size() != 2)
    return failure();
  Location loc = op->getLoc();
  Type ptrType = getPtrTy(op->getContext());
  Value ping = LLVM::ExtractValueOp::create(rewriter, loc, ptrType, operands[1],
                                            ArrayRef<int64_t>{0});
  Value pong = LLVM::ExtractValueOp::create(rewriter, loc, ptrType, operands[1],
                                            ArrayRef<int64_t>{1});
  auto selected = LLVM::SelectOp::create(rewriter, loc, ptrType, operands[0],
                                         ping, pong);
  rewriter.replaceOp(op, selected.getResult());
  return success();
}

LogicalResult LowerSelectDMAFetchData::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (op->getNumOperands() != 2 || operands.size() != 2)
    return failure();
  Location loc = op->getLoc();
  Type ptrType = getPtrTy(op->getContext());
  Value ping = LLVM::ExtractValueOp::create(rewriter, loc, ptrType, operands[1],
                                            ArrayRef<int64_t>{0});
  Value pong = LLVM::ExtractValueOp::create(rewriter, loc, ptrType, operands[1],
                                            ArrayRef<int64_t>{1});
  auto selected = LLVM::SelectOp::create(rewriter, loc, ptrType, operands[0],
                                         ping, pong);
  rewriter.replaceOp(op, selected.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// Lower DMAWait
//===----------------------------------------------------------------------===//

// Structure to lower DMA wait operation to LLVM pattern
struct LowerDMAWait : public ConvertOpToLLVMPattern<memref::DmaWaitOp> {
  // Inherit the constructor from the base class
  using ConvertOpToLLVMPattern<memref::DmaWaitOp>::ConvertOpToLLVMPattern;

  // Override the matchAndRewrite function to perform the conversion
  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Function to get or create the DMA wait function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMAWaitFn(ModuleOp module, StringRef fnName,
             ConversionPatternRewriter &rewriter) {
  // Get the context from the module
  MLIRContext *context = module->getContext();
  // Lookup or create the DMA wait function with the specified name and types
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {rewriter.getI32Type()}, getVoidTy(context));
}

// Override the matchAndRewrite function to perform the conversion
LogicalResult
LowerDMAWait::matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  // Get the context and location of the operation
  auto context = op->getContext();
  auto loc = op->getLoc();
  // Get the parent module of the operation
  auto module = op->getParentOfType<ModuleOp>();

  // Get the name of the DMA wait function
  auto dmaWaitFnName = mlir::hexagon::getDMAWaitFnName();
  // Get the LLVM function for DMA wait
  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMAWaitFn(module, dmaWaitFnName, rewriter);
  if (failed(funcOp))
    return failure();

  // Get the tag memory reference and its type
  auto tagMemref = op.getTagMemRef();
  auto tagMemrefType = cast<MemRefType>(tagMemref.getType());
  // Get the LLVM tag memory reference
  auto llvmTagMemref = adaptor.getTagMemRef();
  // Get the pointer to the tag memory reference
  auto llvmTagPtr = getStridedElementPtr(
      rewriter, loc, tagMemrefType, llvmTagMemref, adaptor.getTagIndices());
  // Erase the original operation
  rewriter.eraseOp(op);
  // Create a load operation to get the token
  auto tokenLoadOp =
      LLVM::LoadOp::create(rewriter, loc, rewriter.getI32Type(), llvmTagPtr);
  // Create a vector of operands for the DMA wait operation
  SmallVector<Value, 10> operands = {tokenLoadOp.getResult()};
  // Create the DMA wait operation
  auto DMAWaitOp =
      LLVM::CallOp::create(rewriter, loc, funcOp.value(), operands);
  // Return success
  return success();
}

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateDMAToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<LowerDMAStart, LowerDMAWait>(converter);
  patterns.add<LowerDMAHandle, LowerCreateAndInitHandles,
               LowerDMAFetchData, LowerCreateDoubleBufferFetchData,
               LowerDMAExtStart, LowerDMA2DStart, LowerDMAExtWait,
               LowerSelectDMAHandle, LowerSelectDMAFetchData>(
      patterns.getContext(), converter);
}

struct DMAToLLVMPass : public ::impl::DMAToLLVMBase<DMAToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, memref_ext::MemRefExtDialect,
                    LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context);
    LLVMTypeConverter typeConverter(context, options);

    hexagon::addTypeConversions(context, typeConverter);
    typeConverter.addConversion([context](memref_ext::DmaHandleType type) {
      return LLVM::LLVMPointerType::get(context);
    });
    typeConverter.addConversion(
        [context](memref_ext::DoubleBufferDmaHandlesType type) -> Type {
          if (type.getNumHandles() != 2)
            return Type();
          Type ptrType = LLVM::LLVMPointerType::get(context);
          return LLVM::LLVMStructType::getLiteral(context, {ptrType, ptrType});
        });
    typeConverter.addConversion([context](memref_ext::DmaFetchDataType type) {
      return LLVM::LLVMPointerType::get(context);
    });
    typeConverter.addConversion(
        [context](memref_ext::DoubleBufferDmaFetchDataType type) -> Type {
          if (type.getNumFetchData() != 2)
            return Type();
          Type ptrType = LLVM::LLVMPointerType::get(context);
          return LLVM::LLVMStructType::getLiteral(context, {ptrType, ptrType});
        });
    populateDMAToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createDMAToLLVMPass() {
  return std::make_unique<DMAToLLVMPass>();
}
