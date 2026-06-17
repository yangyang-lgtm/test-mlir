#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c128_i32 = arith.constant 128 : i32
    %c128 = arith.constant 128 : index
    %0 = arith.muli %arg7, %arg3 : i32
    scf.for %arg10 = %0 to %arg3 step %c128_i32  : i32 {
      %1 = arith.index_cast %arg10 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
      %2 = arith.addi %1, %c128 : index
      %3 = arith.index_cast %arg3 : i32 to index
      %4 = arith.minsi %2, %3 : index
      %5 = arith.maxsi %4, %1 : index
      %6 = arith.subi %5, %1 : index
      %alloc = memref.alloc() : memref<128xf32>
      %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<128xf32>
      %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<128xf32> to memref<?xf32, strided<[1]>>
      memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_2 : memref<128xf32>, memref<128xf32>) outs(%alloc : memref<128xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %7 = arith.addf %in, %in_7 : f32
        linalg.yield %7 : f32
      }
      %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [128], strides: [1] : memref<*xf32> to memref<128xf32, strided<[1], offset: ?>>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_0, %subview_6 : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
    } {all_parallel, tiled_generic}
    return
  }
}
