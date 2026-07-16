
module{
tt.func public @kernel(*arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/driver/triton-mac/test.py":6:0), %arg1: !tt .ptr<f32> {tt.divisibility= 16 : i32} loc("/driver/triton-mac/test.py":6:0), *arg2: !tt.ptr<f32> {tt.divisibility= 16 : i 32} loc("/driver/triton-mac/test.py":6:0), %arg3: !tt.ptr<f32" (tt.divisibility= 16 : i32} loc("/driver/triton-mac/test.py ":6:0),%arg4: i32 {tt.divisibility=16 : i32} loc("/driver/triton-mac/test.py":6:0)) attributes (noinline=false} {
%c128 i32= arith.constant 128: i32 loc(#loc1)
%c0 i32= arith.constant 0: i32 loc(#loc2)
%alloc=memref.alloc() {alignment = 64 : i64} : memref-128xf32> loc(#loc3)
scf.for %arg5 = %c0_i32 to %arg4 step c128 i32
: i32 {
%0 = arith.addi %arg5, %c128 i32: i32 loc(#loc4)
%1= arith.minsi %0, %arg4 : i32 loc(#loc5)
%2= arith.subi %1, %arg5 : i32 loc(#loc6)
%3= tt.addptr %arg0,%arg5 : !tt.ptr<f32>, i32 loc(#loc7)
%alloc 0 = memref.alloc() : memref-128xf32, strided<[1]>> loc(#loc8)
%cst=arith.constant 0.000000e+00: f32 loc(#loc8)
"memref.load_ex"(%3,%2, %cst, %alloc 0) <{is other valid = false, tensor size = 128 : i32)> : (!tt.ptr<f32>, i32, f32, memref<128xf32, strided<[1]>>) -> () loc(#loc8)
%4= tt.addptr %arg1, %arg5 : !tt.ptr<f32>,i32 loc(#loc9)

alloc_1=memref.alloc() : memref<128xf32,strided<[1[p> loc(#loc10)
cst_2= arith.constant 0.000000e+00: f32 loc(#loc10)
"memref.load_ex"(%4,%2,%cst 2, alloc 1) <{is other valid = false, tensor_size = 128 : i32)> : (!tt.ptr<f32>,i32,f32,memref<128xf32, strided<[1]>>) -> () loc(#loc10)
linalg.map { arith.addf } ins(*alloc 0, %alloc_1 : memref<128xf32, strided<[1]>>, mimref<128xf32, strided<[1]>>) outs (salloc:memref<128xf32>)loc(#loc3)
%5= tt.addptr %arg2,%arg5 : !tt.ptr<f32", i32 loc(#loc11)
%alloc 3 = memref.alloc() : memref<128xf32, strided<[1]>>loc(#loc12)
%cst 4 = arith.constant 0.000000e+00 : f32 loc(#loc12)
"memref.load ex"(%5,%2, %cst 4, %alloc 3) <{is_other_valid = false, tensor_size=128: i32}>: (!tt.ptr<f32>,i32,memref<128xf32, strided<[1]>>) -> () loc(#loc12)
linalg.map {arith.addf } ins(%alloc, %alloc 3 : memref<128xf32>, memref<128xf32, strided<[1]>>) outs(*alloc : memref <128xf32>)loc(#loc13)
%6 = tt.addptr %arg3,%arg5: !tt.ptr<f32>, i32 loc(#loc14)
"memref.store ex"(%6, %alloc, %2) : (!tt.ptr<f32>, memref<128xf32>, i32) -> () loc(#loc15)
}{db loop_kind= "pointwise"} loc(#loc2)
tt.return loc(#loc16)
} loc(#loc)
}loc(#loc)
