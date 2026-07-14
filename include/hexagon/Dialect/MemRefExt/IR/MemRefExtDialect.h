//===- MemRefExtDialect.h -------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_DIALECT_H
#define HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hexagon/Dialect/MemRefExt/IR/MemRefExtDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtTypes.h.inc"

#define GET_OP_CLASSES
#include "hexagon/Dialect/MemRefExt/IR/MemRefExtOps.h.inc"

#endif // HEXAGON_DIALECT_MEMREFEXT_IR_MEMREFEXT_DIALECT_H
