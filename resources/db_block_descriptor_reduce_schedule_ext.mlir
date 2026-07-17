module {
  func.func @block_descriptor_reduce_schedule_ext(
      %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i64 = arith.constant 4 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32

    %acc = memref.alloc() {alignment = 64 : i64} : memref<4x128xf32>
    linalg.fill ins(%cst : f32) outs(%acc : memref<4x128xf32>)

    scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
      %a_desc = tt.make_tensor_descriptor %arg0, [%c4_i32, %c128_i32], [%c128_i64, %c1_i64]
          : !tt.ptr<f32>, !tt.tensordesc<tensor<4x128xf32>>
      %a = memref.alloc() : memref<4x128xf32>
      memref_ext.block_descriptor_load %a_desc, %a
          {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
          : !tt.tensordesc<tensor<4x128xf32>>, memref<4x128xf32>

      linalg.map { arith.addf }
          ins(%acc, %a : memref<4x128xf32>, memref<4x128xf32>)
          outs(%acc : memref<4x128xf32>)

      %b_desc = tt.make_tensor_descriptor %arg1, [%c4_i32, %c128_i32], [%c128_i64, %c1_i64]
          : !tt.ptr<f32>, !tt.tensordesc<tensor<4x128xf32>>
      %b = memref.alloc() : memref<4x128xf32>
      memref_ext.block_descriptor_load %b_desc, %b
          {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}
          : !tt.tensordesc<tensor<4x128xf32>>, memref<4x128xf32>

      linalg.map { arith.addf }
          ins(%acc, %b : memref<4x128xf32>, memref<4x128xf32>)
          outs(%acc : memref<4x128xf32>)
    } {db_loop_kind = "reduce"}

    %reduced = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    linalg.reduce ins(%acc : memref<4x128xf32>)
        outs(%reduced : memref<128xf32>)
        dimensions = [0]
        (%in: f32, %init: f32) {
      %sum = arith.addf %in, %init : f32
      linalg.yield %sum : f32
    }

    %expanded = memref.expand_shape %reduced [[0, 1]]
        output_shape [1, 128] : memref<128xf32> into memref<1x128xf32>
    %out_desc = tt.make_tensor_descriptor %arg2, [%c1_i32, %c128_i32], [%c128_i64, %c1_i64]
        : !tt.ptr<f32>, !tt.tensordesc<tensor<1x128xf32>>
    memref_ext.block_descriptor_store %out_desc, %expanded
        {boundaryCheck = array<i32: 0, 1>}
        : !tt.tensordesc<tensor<1x128xf32>>, memref<1x128xf32>
    return
  }
}
