//
// Created by yaoshi16ultra on 2026/6/16.
//
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Transforms/Passes.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "hexagon/Transforms/Transforms.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "FileUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

template <typename... Ts> static void addDialects(mlir::MLIRContext &context) {
  (context.getOrLoadDialect<Ts>(), ...);
}

static void initContext(mlir::MLIRContext &context) {
  addDialects<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
              mlir::linalg::LinalgDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
              mlir::arith::ArithDialect, mlir::triton::TritonDialect,
              mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
              mlir::affine::AffineDialect, mlir::cf::ControlFlowDialect>(
      context);
}

static void initRegistry(mlir::DialectRegistry &registry) {
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModelsForPackUnPackOps(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::outs() << "run as : " << argv[0] << " xxx.mlir\n";
    return 0;
  }

  auto mlirPath = argv[1];

  mlir::DialectRegistry registry;
  initRegistry(registry);
  auto context = mlir::MLIRContext(registry);
  initContext(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath).failed()) {
    llvm::outs() << "parse ir string failed!\n";
    return 0;
  }

  mlir::PassManager manager(&context);
  manager.addPass(mlir::triton::createTritonToLinalgExperimentalPass());
  manager.addPass(mlir::createLinalgFoldUnitExtentDimsPass());

  {
    // bufferization
    manager.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
    manager.addNestedPass<mlir::func::FuncOp>(mlir::hexagon::createEraseVectorToTensorWritebackPass());

    mlir::bufferization::OneShotBufferizePassOptions passOpts;
    passOpts.bufferizeFunctionBoundaries = true;
    passOpts.allowReturnAllocsFromLoops = true;
    manager.addPass(mlir::bufferization::createOneShotBufferizePass(passOpts));
    manager.addPass(mlir::createCSEPass());
    manager.addPass(mlir::createCanonicalizerPass());

    manager.addNestedPass<mlir::func::FuncOp>(
        mlir::hexagon::createAnnotateMemrefCopyDirectionPass());
    manager.addNestedPass<mlir::func::FuncOp>(
        mlir::hexagon::createHoistDoubleBufferCopyInsPass());
    manager.addNestedPass<mlir::func::FuncOp>(mlir::hexagon::createHexagonDoubleBufferGenericS1Pass());
    manager.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());

    manager.addNestedPass<mlir::func::FuncOp>(mlir::hexagon::createCopyCanonicalizationPass());
    manager.addPass(mlir::createCanonicalizerPass());

    mlir::bufferization::buildBufferDeallocationPipeline(
        manager, mlir::bufferization::BufferDeallocationPipelineOptions{});
    manager.addPass(mlir::createCSEPass());

    manager.addNestedPass<mlir::func::FuncOp>(mlir::hexagon::createHexagonDoubleBufferGenericS2Pass());
    manager.addNestedPass<mlir::func::FuncOp>(mlir::hexagon::createConvertZeroSizeMemrefPass());
    manager.addPass(mlir::createConvertBufferizationToMemRefPass());
  }

  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createCSEPass());
  if (manager.run(*module).failed()) {
    llvm::outs() << "run pass failed\n";
    return -1;
  }

  module->print(llvm::outs());
  return 0;
}
