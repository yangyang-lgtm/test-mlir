//===- SetTensorAllocSharedMemoryPass.cpp ---------------------------------===//
//
// Assign shared memory space to every tensor allocation before bufferization.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "triton-shared/Conversion/MemorySpaces.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::hexagon;

#define GEN_PASS_DEF_SETTENSORALLOCSHAREDMEMORY
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct SetTensorAllocSharedMemoryPass
    : public ::impl::SetTensorAllocSharedMemoryBase<
          SetTensorAllocSharedMemoryPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto sharedSpace =
        IntegerAttr::get(IntegerType::get(&getContext(), 64),
                         triton::kSharedMemorySpace);

    getOperation().walk([&](bufferization::AllocTensorOp alloc) {
      alloc.setMemorySpaceAttr(sharedSpace);
    });

    SmallVector<tensor::EmptyOp> emptyOps;
    getOperation().walk(
        [&](tensor::EmptyOp empty) { emptyOps.push_back(empty); });

    IRRewriter rewriter(&getContext());
    for (tensor::EmptyOp empty : emptyOps) {
      rewriter.setInsertionPoint(empty);
      auto alloc = bufferization::AllocTensorOp::create(
          rewriter, empty.getLoc(), empty.getType(), empty.getDynamicSizes(),
          Value(), Value(), sharedSpace);
      rewriter.replaceOp(empty, alloc.getResult());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createSetTensorAllocSharedMemoryPass() {
  return std::make_unique<SetTensorAllocSharedMemoryPass>();
}
