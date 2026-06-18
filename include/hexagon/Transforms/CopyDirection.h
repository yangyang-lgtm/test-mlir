//===- CopyDirection.h - memref.copy direction constants --------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_COPYDIRECTION_H
#define HEXAGON_TRANSFORMS_COPYDIRECTION_H

#include "llvm/ADT/StringRef.h"

namespace mlir::hexagon {

inline constexpr llvm::StringLiteral kCopyDirectionAttrName = "copy_direction";
inline constexpr llvm::StringLiteral kGlobalToShared = "global_to_shared";
inline constexpr llvm::StringLiteral kSharedToGlobal = "shared_to_global";

} // namespace mlir::hexagon

#endif // HEXAGON_TRANSFORMS_COPYDIRECTION_H
