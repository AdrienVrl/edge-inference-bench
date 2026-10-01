#!/usr/bin/env bash
# run_matmul.sh - run matmul_bench across versions/sizes and append each result
# to ../results.csv via bench.py add. Run `make` in this directory first.
#
# Usage:
#   ./run_matmul.sh "Quadro T1000 (WSL2)"
#   ./run_matmul.sh "Quadro T1000 (WSL2)" 512,1024,2048 20
set -euo pipefail
 
PLATFORM="${1:?usage: $0 <platform label> [sizes csv] [iters]}"
SIZES="${2:-1024}"
ITERS="${3:-20}"
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="$SCRIPT_DIR/../bench.py"
BIN="$SCRIPT_DIR/matmul_bench"
 
[ -x "$BIN" ] || { echo "matmul_bench not found - run 'make' in $SCRIPT_DIR first" >&2; exit 1; }
 
IFS=',' read -ra SIZE_LIST <<< "$SIZES"
 
for n in "${SIZE_LIST[@]}"; do
  for version in naive tiled cublas; do
    echo "== $version N=$n ==" >&2
    line="$("$BIN" "$version" "$n" "$ITERS")"
    IFS=',' read -r v size median_ms gflops <<< "$line"
 
    label="naive"
    [ "$version" = "tiled" ] && label="tiled (shared-mem)"
    [ "$version" = "cublas" ] && label="cuBLAS (reference)"
 
    python3 "$BENCH_PY" add \
      phase=2 \
      platform="$PLATFORM" \
      config="CUDA matmul $label N=$size" \
      backend=cuda \
      latency_median_ms="$median_ms" \
      notes="GFLOPS=$gflops; square NxN fp32 sgemm; CUDA event timing"
  done
done
 
echo "done. see ../results.csv / ../README.md" >&2

