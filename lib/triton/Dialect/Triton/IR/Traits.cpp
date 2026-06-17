#include "triton/Dialect/Triton/IR/Traits.h"

#include <numeric>

#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
// #include "triton/Dialect/TritonGPU/IR/Types.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
// using namespace mlir::triton::gpu;

LogicalResult OpTrait::impl::verifyEquivalentType(Type typeA, Type typeB) {
  return success();
}

static LogicalResult verifySameEncoding(Type typeA, Type typeB,
                                        bool allowTensorPointerType) {
  return success();
}

LogicalResult
OpTrait::impl::verifySameOperandsEncoding(Operation *op,
                                          bool allowTensorPointerType) {
  return success();
}

LogicalResult OpTrait::impl::verifySameOperandsAndResultEncoding(
    Operation *op, bool allowTensorPointerType) {
  return success();
}

LogicalResult OpTrait::impl::verifyTensorSize(Operation *op) {
  return success();
}

// Check that the Triton layouts on op's operands and return types are valid.
// For example, we check that the number of warps per block in a Triton GPU
// blocked layout matches that of its module.
//
// It's a little weird to check these properties of a layout only when the
// layout is used in an op, since most of the properties don't actually depend
// on the op.  They do depend on the *module*, though, and a layout is attached
// to a module only by virtue of being used in one of the module's ops.
LogicalResult OpTrait::impl::verifyTensorLayouts(Operation *op) {
  return success();
}

static ArrayRef<int64_t> getTypeShape(Type type) {
  return ArrayRef<int64_t>();
}

LogicalResult OpTrait::impl::verifySameLoadStoreOperandsShape(Operation *op) {
  return success();
}

LogicalResult
OpTrait::impl::verifySameLoadStoreOperandsAndResultShape(Operation *op) {
  return success();
}
