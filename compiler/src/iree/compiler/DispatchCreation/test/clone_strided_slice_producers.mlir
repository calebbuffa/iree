// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions{aggressive=true}))" %s | FileCheck %s

// A slice outside the dispatch will itself be cloned. Its bit-extend producer
// must not follow it: tile-and-fuse cannot cross the non-unit stride.
util.func public @external_strided_slice(%input: tensor<16xf16>, %bias: tensor<16xf16>) -> tensor<8xf32> {
  %wide_empty = tensor.empty() : tensor<16xf32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input, %bias : tensor<16xf16>, tensor<16xf16>) outs(%wide_empty : tensor<16xf32>) {
    ^bb0(%in: f16, %bias_in: f16, %out: f32):
      %sum = arith.addf %in, %bias_in : f16
      %ext = arith.extf %sum : f16 to f32
      linalg.yield %ext : f32
  } -> tensor<16xf32>
  %odd = tensor.extract_slice %wide[1] [8] [2] : tensor<16xf32> to tensor<8xf32>
  %result = flow.dispatch.region -> (tensor<8xf32>) {
    %empty = tensor.empty() : tensor<8xf32>
    %neg = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%odd : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %n = arith.negf %in : f32
        linalg.yield %n : f32
    } -> tensor<8xf32>
    flow.return %neg : tensor<8xf32>
  }
  util.return %result : tensor<8xf32>
}

// CHECK-LABEL: @external_strided_slice
// CHECK: %[[WIDE:.*]] = flow.dispatch.region
// CHECK: arith.addf
// CHECK: arith.extf
// CHECK: flow.return
// CHECK: %[[RESULT:.*]] = flow.dispatch.region
// CHECK-NOT: arith.extf
// CHECK: tensor.extract_slice %[[WIDE]][1] [8] [2]
// CHECK-NOT: arith.extf
// CHECK: arith.negf
// CHECK-NOT: arith.extf
// CHECK: flow.return
// CHECK: util.return %[[RESULT]]

// -----

// Also protect a producer with another use inside the dispatch. Walking only
// the slice's source during clone selection would depend on worklist order.
util.func public @internal_strided_slice_multiuse(%input: tensor<16xf16>) -> (tensor<8xf32>, tensor<16xf32>) {
  %wide_empty = tensor.empty() : tensor<16xf32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<16xf16>) outs(%wide_empty : tensor<16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %ext = arith.extf %in : f16 to f32
      linalg.yield %ext : f32
  } -> tensor<16xf32>
  %result:2 = flow.dispatch.region -> (tensor<8xf32>, tensor<16xf32>) {
    %odd = tensor.extract_slice %wide[1] [8] [2] : tensor<16xf32> to tensor<8xf32>
    %empty = tensor.empty() : tensor<8xf32>
    %neg = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%odd : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %n = arith.negf %in : f32
        linalg.yield %n : f32
    } -> tensor<8xf32>
    flow.return %neg, %wide : tensor<8xf32>, tensor<16xf32>
  }
  util.return %result#0, %result#1 : tensor<8xf32>, tensor<16xf32>
}

// CHECK-LABEL: @internal_strided_slice_multiuse
// CHECK: %[[WIDE:.*]] = flow.dispatch.region
// CHECK: arith.extf
// CHECK: flow.return
// CHECK: %[[RESULT:.*]]:2 = flow.dispatch.region
// CHECK-NOT: arith.extf
// CHECK: tensor.extract_slice %[[WIDE]][1] [8] [2]
// CHECK-NOT: arith.extf
// CHECK: arith.negf
// CHECK-NOT: arith.extf
// CHECK: flow.return {{.*}}, %[[WIDE]]
// CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1

// -----

// Unit-stride slices still allow the bit-extend producer to tile and fuse.
util.func public @unit_stride_still_clones(%input: tensor<16xf16>) -> tensor<8xf32> {
  %wide_empty = tensor.empty() : tensor<16xf32>
  %wide = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%input : tensor<16xf16>) outs(%wide_empty : tensor<16xf32>) {
    ^bb0(%in: f16, %out: f32):
      %ext = arith.extf %in : f16 to f32
      linalg.yield %ext : f32
  } -> tensor<16xf32>
  %slice = tensor.extract_slice %wide[1] [8] [1] : tensor<16xf32> to tensor<8xf32>
  %result = flow.dispatch.region -> (tensor<8xf32>) {
    %empty = tensor.empty() : tensor<8xf32>
    %neg = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%slice : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %n = arith.negf %in : f32
        linalg.yield %n : f32
    } -> tensor<8xf32>
    flow.return %neg : tensor<8xf32>
  }
  util.return %result : tensor<8xf32>
}

// CHECK-LABEL: @unit_stride_still_clones
// CHECK-NOT: arith.extf
// CHECK: %[[RESULT:.*]] = flow.dispatch.region
// CHECK: arith.extf
// CHECK: tensor.extract_slice {{.*}}[1] [8] [1]
// CHECK: arith.negf
// CHECK: flow.return
// CHECK-NOT: flow.dispatch.region
// CHECK: util.return %[[RESULT]]
