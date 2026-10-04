// matmul.mlir - a plain linalg.matmul, plus a transform-dialect sequence
// that tiles it. One file holds both the "payload" (the function you'd
// actually compile) and the "schedule" (how to transform it) - this is
// the current MLIR convention (named sequence + --transform-interpreter),
// replacing the older standalone `-linalg-tile=...` command-line flag.
//
// Run with run_mlir_tile.sh, or by hand:
//   mlir-opt matmul.mlir --transform-interpreter --cse --canonicalize
//
// If your mlir-opt doesn't recognize --transform-interpreter, you have an
// older LLVM build; check `mlir-opt --help | grep -i transform` for
// whatever your version calls it (older versions used
// `-test-transform-dialect-interpreter -test-transform-dialect-erase-schedule`
// instead) - MLIR's transform dialect API has moved more than once.

func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>,
                   %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // Find the linalg.matmul in the payload above.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // linalg.matmul has 3 iterator dims: M, N (parallel), K (reduction).
    // Tile sizes are given in that order; 0 means "leave this dim whole".
    // Tiling M and N by 32 turns the single 128x128x128 matmul into a
    // 4x4 grid of 32x32x128 matmuls, each wrapped in its own loop nest -
    // this is the exact same "block the problem so each chunk reuses
    // data" idea as phase 2's CUDA shared-memory tiling and phase 3's
    // NEON register blocking, just expressed at the IR level instead of
    // hand-written.
    %tiled, %loop_m, %loop_n = transform.structured.tile_using_for %matmul
      tile_sizes [32, 32, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
