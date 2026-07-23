module {
    func.func public @matmul_kernel(
        %arg0: !tt.ptr<bf16>,
        %arg1: !tt.ptr<bf16>,
        %arg2: !tt.ptr<bf16>,
        %arg3: i32,
        %arg4: i32,
        %arg5: i32
    ) attributes {noinline = false} {
        %c64_i32 = arith.constant 64 : i32
        %c0_i32 = arith.constant 0 : i32
        %c128_i32 = arith.constant 128 : i32
        %c1_i64 = arith.constant 1 : i64
        %c1_i32 = arith.constant 1 : i32
        %c63_i32 = arith.constant 63 : i32
        %c127_i32 = arith.constant 127 : i32
        %cst = arith.constant 0.000000e+00 : f32
        %alloc = memref.alloc() : memref<64x64xf32>
        %alloc_0 = memref.alloc() : memref<64x64xf32>
        linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<64x64xf32>)
        %0 = tt.get_program_id x : i32
        %1 = arith.addi %arg3, %c63_i32 : i32
        %2 = arith.divsi %1, %c64_i32 : i32
        %3 = arith.remsi %0, %2 : i32
        %4 = arith.divsi %0, %2 : i32
        %5 = arith.muli %3, %c64_i32 : i32
        %6 = arith.muli %4, %c64_i32 : i32
        %7 = arith.extsi %arg3 : i32 to i64
        %8 = arith.extsi %arg5 : i32 to i64
        %9 = "tt.make_block_desc"(%arg0, %7, %8, %8, %c1_i64, %5, %c0_i32) <{order = array<i32: 1, 0>, tensorShape = array<i64: 64, 128>}> : (!tt.ptr<bf16>, i64, i64, i64, i64, i32, i32) -> !tt.block_desc<bf16, 64, 128, 1, 0>
        %10 = arith.extsi %arg4 : i32 to i64
        %11 = "tt.make_block_desc"(%arg1, %8, %10, %10, %c1_i64, %c0_i32, %6) <{order = array<i32: 1, 0>, tensorShape = array<i64: 128, 64>}> : (!tt.ptr<bf16>, i64, i64, i64, i64, i32, i32) -> !tt.block_desc<bf16, 128, 64, 1, 0>
        %12 = arith.addi %arg5, %c127_i32 : i32
        %13 = arith.divsi %12, %c128_i32 : i32
        %14:2 = scf.for %arg6 = %c0_i32 to %13 step %c1_i32 iter_args(%arg7 = %9, %arg8 = %11) -> (!tt.block_desc<bf16, 64, 128, 1, 0>, !tt.block_desc<bf16, 128, 64, 1, 0>) : i32 {
            %alloc_2 = memref.alloc() : memref<64x128xbf16, strided<[128, 1]>>
            "memref_ext.block_desc_load"(%arg7, %alloc_2) <{boundaryCheck = array<i32 : 1>, padding = 1 : i32}> : (!tt.block_desc<bf16, 64, 128, 1, 0>, memref<64x128xbf16, strided<[128, 1]>>) -> ()
            %alloc_3 = memref.alloc() : memref<128x64xbf16, strided<[64, 1]>>
            "memref_ext.block_desc_load"(%arg8, %alloc_3) <{boundaryCheck = array<i32 : 1>, padding = 1 : i32}> : (!tt.block_desc<bf16, 128, 64, 1, 0>, memref<128x64xbf16, strided<[64, 1]>>) -> ()
            linalg.matmul ins(%alloc_2, %alloc_3 : memref<64x128xbf16, strided<[128, 1]>>, memref<128x64xbf16, strided<[64, 1]>>) outs(%alloc : memref<64x64xf32>)
            linalg.map { arith.addf } ins(%alloc_0, %alloc : memref<64x64xf32>, memref<64x64xf32>) outs(%alloc_0 : memref<64x64xf32>)
            %16 = "tt.block_desc_advance"(%arg7, %c0_i32, %c128_i32) : (!tt.block_desc<bf16, 64, 128, 1, 0>, i32, i32) -> !tt.block_desc<bf16, 64, 128, 1, 0>
            %17 = "tt.block_desc_advance"(%arg8, %c128_i32, %c0_i32) : (!tt.block_desc<bf16, 128, 64, 1, 0>, i32, i32) -> !tt.block_desc<bf16, 128, 64, 1, 0>
            scf.yield %16, %17 : !tt.block_desc<bf16, 64, 128, 1, 0>, !tt.block_desc<bf16, 128, 64, 1, 0>
        } {db_loop_kind = "reduce"}
        %alloc_1 = memref.alloc() : memref<64x64xbf16>
        linalg.map { arith.truncf } ins(%alloc_0 : memref<64x64xf32>) outs(%alloc_1 : memref<64x64xbf16>)
        %15 = "tt.make_block_desc"(%arg2, %7, %10, %10, %c1_i64, %5, %6) <{order = array<i32: 1, 0>, tensorShape = array<i64: 64, 64>}> : (!tt.ptr<bf16>, i64, i64, i64, i64, i32, i32) -> !tt.block_desc<bf16, 64, 64, 1, 0>
        "memref_ext.block_desc_store"(%15, %alloc_1) <{boundaryCheck = array<i32: 0, 1>}> : (!tt.block_desc<bf16, 64, 64, 1, 0>, memref<64x64xbf16>) -> ()
        return
    }
}
