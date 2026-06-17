module {
  func.func @kernel(%arg0: memref<1024x2048xf32>, %arg1: memref<1024x2048xf32>, %arg2: memref<1024x2048xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x2048xf32>
    scf.for %arg3 = %c0 to %c1024 step %c64 {
      %subview = memref.subview %arg0[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x2048xf32, 1>
      memref.copy %subview, %alloc_0 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      %subview_1 = memref.subview %arg1[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64x2048xf32, 1>
      memref.copy %subview_1, %alloc_2 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      %subview_3 = memref.subview %alloc[%arg3, 0] [64, 2048] [1, 1] : memref<1024x2048xf32> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<64x2048xf32, 1>
      memref.copy %subview_3, %alloc_4 : memref<64x2048xf32, strided<[2048, 1], offset: ?>> to memref<64x2048xf32, 1>
      scf.for %arg4 = %c0 to %c64 step %c1 {
        scf.for %arg5 = %c0 to %c2048 step %c32 {
          %subview_5 = memref.subview %alloc_0[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
          %subview_6 = memref.subview %alloc_2[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
          %subview_7 = memref.subview %alloc_4[%arg4, %arg5] [1, 32] [1, 1] : memref<64x2048xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
          %0 = vector.transfer_read %subview_5[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
          %1 = vector.transfer_read %subview_6[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
          %2 = arith.subf %0, %1 fastmath<fast> : vector<32xf32>
          %3 = math.exp %2 fastmath<fast> : vector<32xf32>
          vector.transfer_write %3, %subview_7[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
        }
      }
      memref.copy %alloc_4, %subview_3 : memref<64x2048xf32, 1> to memref<64x2048xf32, strided<[2048, 1], offset: ?>>
    } {all_parallel, tiled_generic}
    memref.copy %alloc, %arg2 : memref<1024x2048xf32> to memref<1024x2048xf32>
    return
  }
}
