// RUN: iree-opt --iree-gpu-test-target=sm_60 --pass-pipeline='builtin.module(func.func(iree-codegen-vector-reduction-to-gpu, canonicalize, cse))' %s | FileCheck %s
// RUN: iree-opt --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(func.func(iree-codegen-vector-reduction-to-gpu, canonicalize, cse))' %s | FileCheck %s

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline workgroup_size = [256, 1, 1] subgroup_size = 32>
#single_subgroup = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline workgroup_size = [32, 1, 1] subgroup_size = 32>
#partial_subgroup = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline workgroup_size = [16, 1, 1] subgroup_size = 32>
module {
  func.func @cross_subgroup_extract() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<256xf32>
    %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<256xf32>
    %input = vector.transfer_read %source[%c0], %zero {in_bounds = [true]} : memref<256xf32>, vector<256xf32>
    %a = vector.extract %input[31] : f32 from vector<256xf32>
    %b = vector.extract %input[32] : f32 from vector<256xf32>
    %c = vector.extract %input[255] : f32 from vector<256xf32>
    %ab = arith.addf %a, %b : f32
    %sum = arith.addf %ab, %c : f32
    %broadcast = vector.broadcast %sum : f32 to vector<256xf32>
    vector.transfer_write %broadcast, %destination[%c0] {in_bounds = [true]} : vector<256xf32>, memref<256xf32>
    return
  }

  func.func @single_subgroup_extract() attributes {translation_info = #single_subgroup} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32xi32>
    %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<32xi32>
    %input = vector.transfer_read %source[%c0], %zero {in_bounds = [true]} : memref<32xi32>, vector<32xi32>
    %scalar = vector.extract %input[31] : i32 from vector<32xi32>
    %broadcast = vector.broadcast %scalar : i32 to vector<32xi32>
    vector.transfer_write %broadcast, %destination[%c0] {in_bounds = [true]} : vector<32xi32>, memref<32xi32>
    return
  }

  func.func @partial_subgroup_extract() attributes {translation_info = #partial_subgroup} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<16xi32>
    %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<16xi32>
    %input = vector.transfer_read %source[%c0], %zero {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
    %scalar = vector.extract %input[15] : i32 from vector<16xi32>
    %broadcast = vector.broadcast %scalar : i32 to vector<16xi32>
    vector.transfer_write %broadcast, %destination[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32>
    return
  }
}

// CHECK-LABEL: func.func @cross_subgroup_extract
// CHECK-DAG: %[[FIRST:.*]] = arith.constant 31 : index
// CHECK-DAG: %[[SECOND:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[LAST:.*]] = arith.constant 255 : index
// CHECK-NOT: gpu.shuffle
// CHECK: memref.store {{.*}}, %[[SHARED0:.*]][%{{.*}}] : memref<256xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.barrier
// CHECK: memref.load %[[SHARED0]][%[[SECOND]]]
// CHECK: gpu.barrier
// CHECK: memref.store {{.*}}, %[[SHARED1:.*]][%{{.*}}] : memref<256xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.barrier
// CHECK: memref.load %[[SHARED1]][%[[FIRST]]]
// CHECK: gpu.barrier
// CHECK: memref.store {{.*}}, %[[SHARED2:.*]][%{{.*}}] : memref<256xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.barrier
// CHECK: memref.load %[[SHARED2]][%[[LAST]]]
// CHECK: gpu.barrier
// CHECK-NOT: gpu.shuffle
// CHECK: vector.transfer_write

// CHECK-LABEL: func.func @single_subgroup_extract
// CHECK-NOT: memref.alloc
// CHECK-NOT: gpu.barrier
// CHECK: %[[WIDTH:.*]] = arith.constant 32 : i32
// CHECK: gpu.shuffle idx {{.*}}, %[[WIDTH]] : i32
// CHECK-NOT: memref.alloc
// CHECK-NOT: gpu.barrier
// CHECK: return

// CHECK-LABEL: func.func @partial_subgroup_extract
// CHECK: memref.alloc() : memref<16xi32, #gpu.address_space<workgroup>>
// CHECK: memref.store
// CHECK: gpu.barrier
// CHECK: memref.load
// CHECK: gpu.barrier
// CHECK-NOT: gpu.shuffle
// CHECK: return
