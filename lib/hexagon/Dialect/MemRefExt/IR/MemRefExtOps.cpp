//===- MemRefExtOps.cpp ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h"
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

void Dma2DStartOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addDmaStartEffects(*this, effects);
}

void DmaWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getHandleMutable(),
                       DmaHandleResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), DmaSyncResource::get());
}
