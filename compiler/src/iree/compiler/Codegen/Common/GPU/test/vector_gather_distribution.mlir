// RUN: iree-opt --iree-gpu-test-target=sm_60 --pass-pipeline='builtin.module(func.func(iree-codegen-vector-reduction-to-gpu, canonicalize, cse))' %s | FileCheck %s

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline workgroup_size = [32, 1, 1] subgroup_size = 32>
module {
  func.func @masked_gather() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %limit = arith.constant dense<63> : vector<128xindex>
    %pass = arith.constant dense<-1.0> : vector<128xf32>
    %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024xf32>
    %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<128xf32>
    %zero = arith.constant 0.0 : f32
    %input = vector.transfer_read %source[%c0], %zero {in_bounds = [true]} : memref<1024xf32>, vector<128xf32>
    %integers = arith.fptosi %input : vector<128xf32> to vector<128xi32>
    %indices = arith.index_cast %integers : vector<128xi32> to vector<128xindex>
    %mask = arith.cmpi ult, %indices, %limit : vector<128xindex>
    %values = vector.gather %source[%c7][%indices], %mask, %pass : memref<1024xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32> into vector<128xf32>
    vector.transfer_write %values, %destination[%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
    return
  }

  func.func @reduction_with_independent_epilogue() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.0 : f32
    %init = arith.constant dense<0.0> : vector<128xf32>
    %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024xf32>
    %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<128xf32>
    %acc = scf.for %i = %c0 to %c1024 step %c128 iter_args(%a = %init) -> vector<128xf32> {
      %v = vector.transfer_read %source[%i], %zero {in_bounds = [true]} : memref<1024xf32>, vector<128xf32>
      %sum = arith.addf %a, %v : vector<128xf32>
      scf.yield %sum : vector<128xf32>
    }
    %sum = vector.reduction <add>, %acc : vector<128xf32> into f32
    %broadcast_load = vector.transfer_read %source[%c0], %zero {permutation_map = affine_map<(d0) -> (0)>, in_bounds = [true]} : memref<1024xf32>, vector<128xf32>
    %indices = arith.fptosi %broadcast_load : vector<128xf32> to vector<128xi32>
    %index_vector = arith.index_cast %indices : vector<128xi32> to vector<128xindex>
    %offset = vector.extract %index_vector[0] : index from vector<128xindex>
    %scalar = memref.load %source[%offset] : memref<1024xf32>
    %result = arith.addf %sum, %scalar : f32
    %v = vector.broadcast %result : f32 to vector<128xf32>
    vector.transfer_write %v, %destination[%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
    return
  }

  func.func @do_not_move_loop_across_write() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0.0 : f32
    %init = arith.constant dense<0.0> : vector<128xf32>
    %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024xf32>
    %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<128xf32>
    %acc = scf.for %i = %c0 to %c1024 step %c128 iter_args(%a = %init) -> vector<128xf32> {
      %v = vector.transfer_read %source[%i], %zero {in_bounds = [true]} : memref<1024xf32>, vector<128xf32>
      %sum = arith.addf %a, %v : vector<128xf32>
      scf.yield %sum : vector<128xf32>
    }
    memref.store %zero, %destination[%c0] : memref<128xf32>
    %sum = vector.reduction <add>, %acc : vector<128xf32> into f32
    %v = vector.broadcast %sum : f32 to vector<128xf32>
    vector.transfer_write %v, %destination[%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
    return
  }

  func.func @do_not_move_gather_across_write() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %mask = arith.constant dense<true> : vector<128xi1>
    %pass = arith.constant dense<0.0> : vector<128xf32>
    %indices_source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024xf32>
    %source = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : memref<128xf32>
    %input = vector.transfer_read %indices_source[%c0], %zero {in_bounds = [true]} : memref<1024xf32>, vector<128xf32>
    %integers = arith.fptosi %input : vector<128xf32> to vector<128xi32>
    %indices = arith.index_cast %integers : vector<128xi32> to vector<128xindex>
    %values = vector.gather %source[%c0][%indices], %mask, %pass : memref<128xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32> into vector<128xf32>
    memref.store %zero, %source[%c0] : memref<128xf32>
    vector.transfer_write %values, %source[%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
    return
  }
}

// CHECK-LABEL: func.func @masked_gather
// CHECK-NOT: gpu.warp_execute_on_lane_0
// CHECK-NOT: memref.alloc
// CHECK: vector.gather {{.*}} : memref<1024xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK: vector.transfer_write {{.*}} : vector<4xf32>, memref<128xf32>
// CHECK-NOT: vector<128x

// CHECK-LABEL: func.func @reduction_with_independent_epilogue
// CHECK-NOT: dense<0.000000e+00> : vector<128xf32>
// CHECK: scf.for
// CHECK-SAME: vector<4xf32>
// CHECK: vector.transfer_read {{.*}} vector<4xf32>
// CHECK: gpu.shuffle
// CHECK-NOT: vector<128x

// CHECK-LABEL: func.func @do_not_move_loop_across_write
// CHECK: scf.for
// CHECK-SAME: vector<128xf32>
// CHECK: memref.store

// CHECK-LABEL: func.func @do_not_move_gather_across_write
// CHECK: vector.gather
// CHECK-SAME: into vector<128xf32>
// CHECK: memref.store
