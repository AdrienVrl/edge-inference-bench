#!/usr/bin/env bash
# run_neon.sh - runs ON THE ARM VM. Loops neon_bench (and neon_bench_dotprod,
# if built and supported) across sizes, and writes results_neon.csv in this
# directory. No Python/bench.py needed here - that stays on your laptop.
#
# Build first:  make            (baseline)
#               make dotprod    (optional, needs Armv8.2+ dotprod support)
#
# Usage:
#   ./run_neon.sh
#   ./run_neon.sh 4096,65536,1048576 200
set -euo pipefail

SIZES="${1:-4096,65536,1048576}"
ITERS="${2:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/results_neon.csv"

[ -x "$SCRIPT_DIR/neon_bench" ] || { echo "neon_bench not found - run 'make' first" >&2; exit 1; }

HAVE_DOTPROD=0
if [ -x "$SCRIPT_DIR/neon_bench_dotprod" ]; then
  if grep -q asimddp /proc/cpuinfo 2>/dev/null; then
    HAVE_DOTPROD=1
  else
    echo "neon_bench_dotprod built, but this CPU has no asimddp - skipping dotprod runs" >&2
  fi
fi

echo "version,N,median_ms,gops" > "$OUT"

IFS=',' read -ra SIZE_LIST <<< "$SIZES"
for n in "${SIZE_LIST[@]}"; do
  for version in naive autovec neon; do
    echo "== $version N=$n ==" >&2
    "$SCRIPT_DIR/neon_bench" "$version" "$n" "$ITERS" | tee -a "$OUT"
  done
  if [ "$HAVE_DOTPROD" = "1" ]; then
    echo "== dotprod N=$n ==" >&2
    "$SCRIPT_DIR/neon_bench_dotprod" dotprod "$n" "$ITERS" | tee -a "$OUT"
  fi
done

echo "wrote $OUT - scp this back to your laptop and run import_neon_results.sh" >&2
