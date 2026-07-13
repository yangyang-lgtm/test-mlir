cmake_minimum_required(VERSION 3.20)

project(custom CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcoroutines>)

set(AIR_ENABLE_AIE ON)
set(LLVM_BUILD_PATH "/home/yaoshi16ultra/workspace/llvm-project/build")
set(LLVM_LIBRARY_DIR "${LLVM_BUILD_PATH}/lib")

function(add_triton_object name)
    cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
    add_library(${name} OBJECT)

    target_sources(${name}
            PRIVATE ${ARG_UNPARSED_ARGUMENTS}
            INTERFACE $<TARGET_OBJECTS:${name}>
    )

    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    if(ARG_LINK_LIBS)
        target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
    endif()
endfunction(add_triton_object)

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
    set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
    add_triton_object(${name} ${ARGN})
    target_compile_options(${name} PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
    set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
    add_triton_object(${name} ${ARGN})
endfunction()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
set(LLD_DIR ${LLVM_LIBRARY_DIR}/cmake/lld)

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

message(STATUS ${MLIR_CMAKE_DIR})
message(STATUS ${LLVM_CMAKE_DIR})

include(LLVMConfig)
include(MLIRConfig)
include(TableGen)
include(AddLLVM)
include(AddMLIR)

message(STATUS "LLVM_ENABLE_RTTI: ${LLVM_ENABLE_RTTI}")
if(LLVM_ENABLE_RTTI)
    string(APPEND CMAKE_CXX_FLAGS " -frtti")
else()
    string(APPEND CMAKE_CXX_FLAGS " -fno-rtti")
endif()

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(third_party/f2reduce)

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(TRITON_LIBRARIES
        ${triton_libs}
        ${triton_plugins}
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}

        # mlir
        MLIRAMDGPUDialect
        MLIRNVVMDialect
        MLIRNVVMToLLVMIRTranslation
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRGPUTransforms
        MLIRIR
        MLIRControlFlowToLLVM
        MLIRControlFlowTransforms
        MLIRArithTransforms
        MLIRAffineToStandard
        MLIRBytecodeWriter
        MLIRPass
        MLIRTransforms
        MLIRLLVMDialect
        MLIRFuncInlinerExtension
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRMathToLLVM
        MLIRROCDLToLLVMIRTranslation
        MLIRGPUDialect
        MLIRTensorTransforms
        MLIRVectorTransforms
        MLIRSCFTransforms
        MLIRMemRefTransforms
        MLIRLinalgTransforms
        MLIRSCFToControlFlow
        MLIRIndexToLLVM
        MLIRGPUToROCDLTransforms
        MLIRUBToLLVM
        MLIRPluginsLib

        # LLVM
        LLVMPasses
        LLVMNVPTXCodeGen
        LLVMAMDGPUCodeGen
        LLVMAMDGPUAsmParser

        # hexagon
        AffineToLLVM
        LinalgToLLVM
        DMAToLLVM
        HexKLToLLVM
        HexagonMemToLLVM
        Transforms

        MLIRRegisterAllDialects
)

#include_directories("/home/yaoshi16ultra/workspace/concurrencpp/include")
#link_directories("/home/yaoshi16ultra/workspace/concurrencpp/build/lib")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(
        ${PROJECT_NAME} PRIVATE
        ${TRITON_LIBRARIES}
#        concurrencpp
)
