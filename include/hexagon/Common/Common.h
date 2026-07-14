//===- Common.h   - some useful common definitions types and functions ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_COMMON_COMMON_H
#define HEXAGON_COMMON_COMMON_H
#include <memory>
#include <optional>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir {
namespace hexagon {

constexpr StringLiteral kLoopKindAttr = "db_loop_kind";
constexpr StringLiteral kTiledGenericAttr = "tiled_generic";
constexpr StringLiteral kPointwise = "pointwise";
constexpr StringLiteral kReduce = "reduce";
constexpr StringLiteral kPrefetchRole = "prefetch";
constexpr StringLiteral kDB2GroupAttr = "db2.group";
constexpr StringLiteral kDB2KernelAttr = "db2.kernel";
constexpr StringLiteral kDB2PrologueAttr = "db2.prologue";
constexpr StringLiteral kDB2PrefetchAttr = "db2.prefetch";
constexpr StringLiteral kDB2RoleAttr = "db2.role";
constexpr StringLiteral kDB2StageAttr = "db2.stage";
constexpr StringLiteral kDB2TileAttr = "db2.tile";
constexpr StringLiteral kDB2ComputeAttr = "db2.compute";
constexpr StringLiteral kDB2PrefetchRole = kPrefetchRole;
constexpr StringLiteral kDB2StoreRole = "store";
constexpr StringLiteral kDB2PrologueStage = "prologue";
constexpr StringLiteral kDB2KernelStage = "kernel";

inline constexpr unsigned DEFAULT_DDR_ADDRESS_SPACE = 0;
inline constexpr unsigned VTCM_ADDRESS_SPACE = 1;

inline const SmallVector<unsigned> INT8_CROUTON_SHAPE = {8, 8, 32};
inline const SmallVector<unsigned> F16_CROUTON_SHAPE = {8, 2, 32, 2};

/// Return `true` if memref type either doesn't have any memspace attribute
/// (in which case it belongs to default DDR memspace 0) or has an integer
/// attribute. If so, it updates memSpaceID with the attribute's value.
bool isMemorySpaceIntTypeOrDefault(mlir::MemRefType type, int &memSpaceID);

/// Return `true` if memref address-space is VTCM.
bool isInVTCMAddressSpace(MemRefType type);

/// Return `true` if memref type has static shape and is contiguous.
bool isContiguousMemrefType(MemRefType type);

/// Return `true` if memref type is a strided multi-dim memref type. If so,
/// set the arguments' values to the memref type's stride and width.
bool isStridedMultiDimMemrefType(mlir::MemRefType memRefType, int64_t &stride,
                                 int64_t &width);

// LLVM ptr type conversion to support llvm hexagon backend.
void addTypeConversions(MLIRContext *context, LLVMTypeConverter &typeConverter);

template <typename T>
std::string toString(SmallVector<T> vec, std::string delim = ",") {
  std::stringstream msg;
  for (auto s : vec)
    msg << s << delim;
  return msg.str();
}

static bool isEnvTrue(const char *name) {
  if (const char *env = std::getenv(name)) {
    std::string envStr(env);
    std::transform(envStr.begin(), envStr.end(), envStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return envStr == "on" || envStr == "true" || envStr == "1";
  }
  return false;
}

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_COMMON_COMMON_H
