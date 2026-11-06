// RUN: iree-opt --iree-gpu-test-target=valhall1 --pass-pipeline='builtin.module(iree-codegen-spirv-configuration-pipeline,iree-codegen-linalg-to-spirv-pipeline)' %s | FileCheck %s

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @finite_predicate() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1 : i8
  %infinity = arith.constant 0x7F800000 : f32
  %source = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<82944xf16>>
  %destination = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i8>>
  %input = iree_tensor_ext.dispatch.tensor.load %source, offsets = [0], sizes = [82944], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<82944xf16>> -> tensor<82944xf16>
  %empty = tensor.empty() : tensor<i8>
  %init = linalg.fill ins(%one : i8) outs(%empty : tensor<i8>) -> tensor<i8>
  %extended_empty = tensor.empty() : tensor<82944xf32>
  %extended = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<82944xf16>) outs(%extended_empty : tensor<82944xf32>) {
  ^bb0(%in: f16, %out: f32):
    %value = arith.extf %in : f16 to f32
    linalg.yield %value : f32
  } -> tensor<82944xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
  } ins(%extended : tensor<82944xf32>) outs(%init : tensor<i8>) {
  ^bb0(%in: f32, %out: i8):
    %previous = arith.trunci %out : i8 to i1
    %absolute = math.absf %in : f32
    %finite = arith.cmpf one, %absolute, %infinity : f32
    %all = arith.andi %previous, %finite : i1
    %stored = arith.extui %all : i1 to i8
    linalg.yield %stored : i8
  } -> tensor<i8>
  iree_tensor_ext.dispatch.tensor.store %result, %destination, offsets = [], sizes = [], strides = [] : tensor<i8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i8>>
  return
}

// CHECK-NOT: Workgroup
// CHECK: spirv.func @finite_predicate
// CHECK: spirv.FConvert
// CHECK: spirv.LogicalAnd
// CHECK-NOT: Workgroup
