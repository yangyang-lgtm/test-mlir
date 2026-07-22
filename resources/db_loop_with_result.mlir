module {
    func.func public @softmax_kernel_inner(
        %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
        %arg2: i32,
        %arg3: i32
    ) attributes {noinline = false} {
        %cst = arith.constant 0xFF800000 : f32
        %c0_i32 = arith.constant 0 : i32
        %cst_0 = arith.constant 0x7F800000 : f32
        %c128_i32 = arith.constant 128 : i32
        %cst_1 = arith.constant 0.000000e+00 : f32
        %alloc = memref.alloc() : memref<128xf32>
        %alloc_2 = memref.alloc() : memref<128xf32>
        linalg.fill ins(%cst_1 : f32) outs(%alloc_2 : memref<128xf32>)
        %alloc_3 = memref.alloc() : memref<128xf32>
        linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<128xf32>)
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %arg3 : i32
        %alloc_4 = memref.alloc() : memref<128xi1>
        %alloc_5 = memref.alloc() : memref<128xf32>
        memref.copy %alloc_3, %alloc_5 : memref<128xf32> to memref<128xf32>
        %2:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c128_i32 iter_args(%arg5 = %alloc_2, %arg6 = %alloc_5) -> (memref<128xf32>, memref<128xf32>) : i32 {
            %5 = arith.addi %1, %arg4 : i32
            %6 = arith.subi %arg3, %arg4 : i32
            %7 = arith.minsi %6, %c128_i32 : i32
            %8 = tt.addptr %arg1, %5 : !tt.ptr<f32>, i32
            %alloc_9 = memref.alloc() : memref<128xf32, strided<[1]>>
            "memref_ext.load_ex"(%8, %7, %cst_0, %alloc_9) <{is_other_valid = true, tensor_size = 128 : i32}> : (!tt.ptr<f32>, i32, f32, memref<128xf32, strided<[1]>>) -> ()
            %alloc_10 = memref.alloc() : memref<128xf32>
            linalg.map { arith.maxnumf } ins(%arg6, %alloc_9 : memref<128xf32>, memref<128xf32, strided<[1]>>) outs(%alloc_10 : memref<128xf32>)
            linalg.map { arith.cmpf {predicate = 1 : i64} } ins(%alloc_10, %alloc_3 : memref<128xf32>, memref<128xf32>) outs(%alloc_4 : memref<128xi1>)
            linalg.map { arith.subf } ins(%arg6, %alloc_10 : memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>)
            linalg.map { arith.negf } ins(%alloc : memref<128xf32>) outs(%alloc : memref<128xf32>)
            %alloc_11 = memref.alloc() : memref<128xf32>
            linalg.map { arith.mulf } ins(%arg5, %alloc : memref<128xf32>, memref<128xf32>) outs(%alloc_11 : memref<128xf32>)
            linalg.map { arith.subf } ins(%alloc_9, %alloc_10 : memref<128xf32, strided<[1]>>, memref<128xf32>) outs(%alloc : memref<128xf32>)
            linalg.map { arith.negf } ins(%alloc : memref<128xf32>) outs(%alloc : memref<128xf32>)
            linalg.map { arith.addf } ins(%alloc_11, %alloc : memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>)
            linalg.map { arith.select } ins(%alloc_4, %arg5, %alloc : memref<128xi1>, memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>)
            %alloc_12 = memref.alloc() : memref<128xf32>
            memref.copy %alloc, %alloc_12 : memref<128xf32> to memref<128xf32>
            scf.yield %alloc_12, %alloc_10 : memref<128xf32>, memref<128xf32>
        } {db_loop_kind = "reduce"}
        %alloc_6 = memref.alloc() : memref<f32>
        linalg.reduce ins(%2#1 : memref<128xf32>) outs(%alloc_6 : memref<f32>) dimensions = [0]
        (%in : f32, %init : f32) {
            %5 = arith.maxnumf %in, %init : f32
            linalg.yield %5 : f32
        }
        %3 = memref.load %alloc_6[] : memref<f32>
        %alloc_7 = memref.alloc() : memref<128xf32>
        linalg.fill ins(%3 : f32) outs(%alloc_7 : memref<128xf32>)
        linalg.map { arith.subf } ins(%2#1, %alloc_7 : memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>)
        linalg.map { arith.negf } ins(%alloc : memref<128xf32>) outs(%alloc : memref<128xf32>)
        linalg.map { arith.mulf } ins(%2#0, %alloc : memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>)
        linalg.reduce ins(%alloc : memref<128xf32>) outs(%alloc_6 : memref<f32>) dimensions = [0]
        (%in : f32, %init : f32) {
            %5 = arith.addf %in, %init : f32
            linalg.yield %5 : f32
        }
        %4 = memref.load %alloc_6[] : memref<f32>
        %alloc_8 = memref.alloc() : memref<128xf32>
        linalg.fill ins(%4 : f32) outs(%alloc_8 : memref<128xf32>)
        scf.for %arg4 = %c0_i32 to %arg3 step %c128_i32 : i32 {
            %5 = arith.addi %1, %arg4 : i32
            %6 = arith.subi %arg3, %arg4 : i32
            %7 = arith.minsi %6, %c128_i32 : i32
            %8 = tt.addptr %arg1, %5 : !tt.ptr<f32>, i32
            %alloc_9 = memref.alloc() : memref<128xf32, strided<[1]>>
            "memref_ext.load_ex"(%8, %7, %cst_0, %alloc_9) <{is_other_valid = true, tensor_size = 128 : i32}> : (!tt.ptr<f32>, i32, f32, memref<128xf32, strided<[1]>>) -> ()
            linalg.map { arith.subf } ins(%alloc_9, %alloc_7 : memref<128xf32, strided<[1]>>, memref<128xf32>) outs(%alloc : memref<128xf32>)
            linalg.map { arith.negf } ins(%alloc : memref<128xf32>) outs(%alloc : memref<128xf32>)
            linalg.map { arith.divf } ins(%alloc, %alloc_8 : memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>)
            %9 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
            "memref_ext.store_ex"(%9, %alloc, %7) : (!tt.ptr<f32>, memref<128xf32>, i32) -> ()
        } {db_loop_kind = "pointwise"}
        return
    }
}
