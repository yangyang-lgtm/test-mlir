module {
    func.func public @max_first_dim(
        %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
    ) attributes {noinline = false} {
        %c3_i32 = arith.constant 3 : i32
        %c1_i32 = arith.constant 1 : i32
        %c1_i64 = arith.constant 1 : i64
        %c532_i64 = arith.constant 532 : i64
        %c73_i64 = arith.constant 73 : i64
        %c128_i32 = arith.constant 128 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %cst = arith.constant 0xFF800000 : f32
        %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32>
        linalg.fill ins(%cst : f32) outs(%alloc : memref<1x128xf32>)
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c128_i32 : i32
        %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
        %expand_shape = memref.expand_shape %alloc_0 [[0, 1]] output_shape [1, 128] : memref<128xf32> into memref<1x128xf32>
        scf.for %arg2 = %c0_i32 to %c3_i32 step %c1_i32 : i32 {
            %3 = arith.muli %arg2, %c32_i32 : i32
            %4 = "tt.make_block_desc"(%arg0, %c73_i64, %c532_i64, %c532_i64, %c1_i64, %3, %1) <{order = array<i32 : 1, 0>, tensorShape = array<i64 : 32, 128>}> : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> !tt.block_desc<f32, 32, 128, 1, 0>
            %alloc_1 = memref.alloc() : memref<32x128xf32, strided<[128, 1]>>
            "memref_ext.block_desc_load"(%4, %alloc_1) <{boundaryCheck = array<i32: 0, 1>, padding = 4 : i32}> : (!tt.block_desc<f32, 32, 128, 1, 0>, memref<32x128xf32, strided<[128, 1]>>) -> ()
            linalg.reduce ins(%alloc_1 : memref<32x128xf32, strided<[128, 1]>>) outs(%alloc_0 : memref<128xf32>) dimensions = [0]
            (%in : f32, %init : f32) {
                %5 = arith.maxnumf %in, %init : f32
                linalg.yield %5 : f32
            }
            linalg.map { arith.maxnumf } ins(%alloc, %expand_shape : memref<1x128xf32>, memref<1x128xf32>) outs(%alloc : memref<1x128xf32>)
        } { db_loop_kind = "reduce" }
        %2 = "tt.make_block_desc"(%arg1, %c1_i64, %c532_i64, %c532_i64, %c1_i64, %c0_i32, %1) <{order = array<i32 : 1, 0>, tensorShape = array<i64 : 1, 128>}> : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> !tt.block_desc<f32, 1, 128, 1, 0>
        "memref_ext.block_desc_store"(%2, %alloc) <{boundaryCheck = array<i32: 0, 1>}> : (!tt.block_desc<f32, 1, 128, 1, 0>, memref<1x128xf32>) -> ()
        return
    }
}