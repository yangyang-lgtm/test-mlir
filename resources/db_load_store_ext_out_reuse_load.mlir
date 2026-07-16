module {
  func.func @load_store_ext_out_reuse_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                           %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>,
                                           %n: index) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %iv = %c0 to %n step %c128 {
      %end = arith.addi %iv, %c128 : index
      %limit = arith.minsi %end, %n : index
      %valid_idx = arith.subi %limit, %iv : index
      %iv_i32 = arith.index_cast %iv : index to i32
      %valid = arith.index_cast %valid_idx : index to i32

      %a_ptr = tt.addptr %arg0, %iv_i32 : !tt.ptr<f32>, i32
      %a = memref.alloc() : memref<128xf32>
      memref_ext.load_ex %a_ptr, %valid, %cst, %a
          {is_other_valid = false, tensor_size = 128 : i32}
          : !tt.ptr<f32>, i32, f32, memref<128xf32>

      %b_ptr = tt.addptr %arg1, %iv_i32 : !tt.ptr<f32>, i32
      %b = memref.alloc() : memref<128xf32>
      memref_ext.load_ex %b_ptr, %valid, %cst, %b
          {is_other_valid = false, tensor_size = 128 : i32}
          : !tt.ptr<f32>, i32, f32, memref<128xf32>

      linalg.map { arith.addf } ins(%a, %b : memref<128xf32>, memref<128xf32>)
          outs(%a : memref<128xf32>)

      %c_ptr = tt.addptr %arg2, %iv_i32 : !tt.ptr<f32>, i32
      %c = memref.alloc() : memref<128xf32>
      memref_ext.load_ex %c_ptr, %valid, %cst, %c
          {is_other_valid = false, tensor_size = 128 : i32}
          : !tt.ptr<f32>, i32, f32, memref<128xf32>

      linalg.map { arith.addf } ins(%a, %c : memref<128xf32>, memref<128xf32>)
          outs(%a : memref<128xf32>)

      %d_ptr = tt.addptr %arg3, %iv_i32 : !tt.ptr<f32>, i32
      memref_ext.store_ex %d_ptr, %a, %valid
          : !tt.ptr<f32>, memref<128xf32>, i32
    } {db_loop_kind = "pointwise"}
    return
  }
}
