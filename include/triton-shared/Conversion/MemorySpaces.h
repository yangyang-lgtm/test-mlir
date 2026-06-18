//===- MemorySpaces.h - Triton lowering memory spaces ----------*- C++ -*-===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_CONVERSION_MEMORYSPACES_H
#define TRITON_SHARED_CONVERSION_MEMORYSPACES_H

#include <cstdint>

namespace mlir::triton {

inline constexpr int64_t kGlobalMemorySpace = 0;
inline constexpr int64_t kSharedMemorySpace = 128;

} // namespace mlir::triton

#endif // TRITON_SHARED_CONVERSION_MEMORYSPACES_H
