// RUN: iree-opt --split-input-file --verify-diagnostics %s
// RUN: iree-opt --split-input-file --canonicalize --verify-diagnostics %s

func.func @scatter_dynamic_updates_static_indices(
    %updates: tensor<?xi64>, %indices: tensor<6xi64>,
    %mask: tensor<6xi1>, %base: tensor<8xi64>) -> tensor<8xi64> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%updates, %indices, %mask : tensor<?xi64>, tensor<6xi64>, tensor<6xi1>)
      outs(%base : tensor<8xi64>) {
    ^bb0(%update: i64, %old: i64):
      %value = arith.maxsi %update, %old : i64
      iree_linalg_ext.yield %value : i64
  } -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----

func.func @scatter_static_updates_dynamic_indices(
    %updates: tensor<6x4xf32>, %indices: tensor<?x1xi32>,
    %mask: tensor<?xi1>, %base: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%updates, %indices, %mask : tensor<6x4xf32>, tensor<?x1xi32>, tensor<?xi1>)
      outs(%base : tensor<8x4xf32>) {
    ^bb0(%update: f32, %old: f32):
      %value = arith.addf %update, %old : f32
      iree_linalg_ext.yield %value : f32
  } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

func.func @gather_dynamic_output_static_indices(
    %base: tensor<8x4xf32>, %indices: tensor<6x1xi32>,
    %output: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = iree_linalg_ext.gather dimension_map = [0]
      ins(%base, %indices : tensor<8x4xf32>, tensor<6x1xi32>)
      outs(%output : tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

func.func @gather_static_output_dynamic_indices(
    %base: tensor<8xf32>, %indices: tensor<?xi32>,
    %output: tensor<0xf32>) -> tensor<0xf32> {
  %0 = iree_linalg_ext.gather dimension_map = [0]
      ins(%base, %indices : tensor<8xf32>, tensor<?xi32>)
      outs(%output : tensor<0xf32>) -> tensor<0xf32>
  return %0 : tensor<0xf32>
}

// -----

func.func @scatter_conflicting_static_batch(
    %updates: tensor<5xi64>, %indices: tensor<6xi64>,
    %base: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
      ins(%updates, %indices : tensor<5xi64>, tensor<6xi64>)
      outs(%base : tensor<8xi64>) {
    ^bb0(%update: i64, %old: i64):
      iree_linalg_ext.yield %update : i64
  } -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----

func.func @gather_conflicting_static_batch(
    %base: tensor<8xf32>, %indices: tensor<6xi32>,
    %output: tensor<0xf32>) -> tensor<0xf32> {
  // expected-error @+1 {{mismatch in shape of indices and output value at dim#0}}
  %0 = iree_linalg_ext.gather dimension_map = [0]
      ins(%base, %indices : tensor<8xf32>, tensor<6xi32>)
      outs(%output : tensor<0xf32>) -> tensor<0xf32>
  return %0 : tensor<0xf32>
}
