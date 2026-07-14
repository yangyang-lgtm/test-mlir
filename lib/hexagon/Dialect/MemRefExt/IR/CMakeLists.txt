add_mlir_dialect_library(MemRefExt
  MemRefExtDialect.cpp
  MemRefExtOps.cpp
  MemRefExtTypes.cpp

  DEPENDS
  MemRefExtTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)
