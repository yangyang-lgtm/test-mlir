//===- MemRefExtOps.cpp ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::memref_ext;

namespace {

template <typename DmaStartLikeOp>
void addDmaStartEffects(
    DmaStartLikeOp op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &op.getSourceMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &op.getTargetMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &op.getHandleMutable(),
                       DmaHandleResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &op.getHandleMutable(),
                       DmaHandleResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), DmaSyncResource::get());
}

} // namespace

void MemRefExtDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"

void DmaStartOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addDmaStartEffects(*this, effects);
}

LogicalResult DmaStartOp::verify() {
  bool sourceIsMemref = isa<BaseMemRefType>(getSource().getType());
  bool targetIsMemref = isa<BaseMemRefType>(getTarget().getType());
  bool sourceIsPtr = isa<triton::PointerType>(getSource().getType());
  bool targetIsPtr = isa<triton::PointerType>(getTarget().getType());

  if (!((sourceIsMemref || sourceIsPtr) && (targetIsMemref || targetIsPtr)))
    return emitOpError("expects source/target to be memref or !tt.ptr");
  if (sourceIsPtr && targetIsPtr)
    return emitOpError("does not support pointer-to-pointer DMA");
  return success();
}

void Dma2DStartOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addDmaStartEffects(*this, effects);
}

void DmaStartExOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getHandleMutable(),
                       DmaHandleResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getHandleMutable(),
                       DmaHandleResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), DmaSyncResource::get());
}

LogicalResult DmaStartExOp::verify() {
  if (!isa<triton::PointerType>(getPtr().getType()))
    return emitOpError("expects ptr to have !tt.ptr type");
  return success();
}

void DmaWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getHandleMutable(),
                       DmaHandleResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), DmaSyncResource::get());
}

LogicalResult SelectMemrefOp::verify() {
  if (getTrueValue().getType() != getFalseValue().getType())
    return emitOpError("expects true_value and false_value to have the same "
                       "memref type");
  if (getResult().getType() != getTrueValue().getType())
    return emitOpError("expects result type to match selected memref type");
  return success();
}

void LoadExOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getTargetMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult LoadExOp::verify() {
  if (!isa<triton::PointerType>(getPtr().getType()))
    return emitOpError("expects ptr to have !tt.ptr type");
  return success();
}

void StoreExOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult StoreExOp::verify() {
  if (!isa<triton::PointerType>(getPtr().getType()))
    return emitOpError("expects ptr to have !tt.ptr type");
  return success();
}
