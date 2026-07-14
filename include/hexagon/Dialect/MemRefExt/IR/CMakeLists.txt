set(LLVM_TARGET_DEFINITIONS MemRefExtDialect.td)
mlir_tablegen(MemRefExtDialect.h.inc -gen-dialect-decls -dialect=memref_ext)
mlir_tablegen(MemRefExtDialect.cpp.inc -gen-dialect-defs -dialect=memref_ext)

set(LLVM_TARGET_DEFINITIONS MemRefExtTypes.td)
mlir_tablegen(MemRefExtTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MemRefExtTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS MemRefExtOps.td)
mlir_tablegen(MemRefExtOps.h.inc -gen-op-decls)
mlir_tablegen(MemRefExtOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MemRefExtTableGen)
