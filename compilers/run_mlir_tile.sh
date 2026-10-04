#!/usr/bin/env bash
# run_mlir_tile.sh - apply the tiling transform in matmul.mlir and save
# both the untransformed and tiled IR so you can diff them.
#
# Usage: ./run_mlir_tile.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command -v mlir-opt >/dev/null || { echo "mlir-opt not found on PATH" >&2; exit 1; }

echo "== baseline (canonicalized, not tiled) ==" >&2
mlir-opt matmul.mlir --canonicalize -o baseline.mlir

echo "== applying transform-interpreter (tiling) ==" >&2
if ! mlir-opt matmul.mlir --transform-interpreter --cse --canonicalize -o tiled.mlir 2>err.log; then
  echo "--transform-interpreter failed - your mlir-opt is likely an older version." >&2
  echo "Check the flag name for your version:" >&2
  mlir-opt --help 2>/dev/null | grep -i transform || true
  echo "See err.log for the full error." >&2
  exit 1
fi

echo >&2
echo "wrote baseline.mlir and tiled.mlir. Diff them:" >&2
echo "  diff baseline.mlir tiled.mlir" >&2
echo >&2
echo "What to look for in tiled.mlir:" >&2
echo "  - two nested scf.for loops (the M and N tiling loops) that weren't" >&2
echo "    there before" >&2
echo "  - the linalg.matmul inside them now operating on tensor<32x32xf32>" >&2
echo "    slices (via tensor.extract_slice) instead of the full 128x128" >&2
echo "  - a tensor.insert_slice writing each tile's result back into the" >&2
echo "    full output - this + the extract_slice is literally the tiling:" >&2
echo "    same total work, reorganized into blocks" >&2
echo >&2
grep -c "scf.for" tiled.mlir | xargs -I{} echo "scf.for count in tiled.mlir: {} (expect 2)"
