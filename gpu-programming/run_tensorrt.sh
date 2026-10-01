#!/usr/bin/env bash
# run_tensorrt.sh - build FP32/FP16/INT8 TensorRT engines with trtexec and
# append each result to ../results.csv via bench.py add.
#
# INT8 reuses ../models/mobilenet_v2.int8_trt.onnx from phase 1: that graph
# already has QuantizeLinear/DequantizeLinear nodes with calibrated scales
# baked in (from quantize_static.py), so trtexec's --int8 flag picks up
# those explicit scales directly - no separate TensorRT calibration pass
# needed. This is the "QDQ format is what TensorRT wants" point from phase 1,
# now paying off.
#
# Requires: TensorRT installed with trtexec on PATH (check: trtexec --help).
# Usage:
#   ./run_tensorrt.sh "Quadro T1000 (WSL2)"
set -euo pipefail

PLATFORM="${1:?usage: $0 <platform label>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="$SCRIPT_DIR/../utils/bench.py"
MODELS_DIR="$SCRIPT_DIR/../models"
ENGINES_DIR="$SCRIPT_DIR/engines"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$ENGINES_DIR" "$LOG_DIR"

command -v trtexec >/dev/null || { echo "trtexec not found on PATH" >&2; exit 1; }

# name, onnx path, extra trtexec flags, notes
CONFIGS=(
  "fp32|$MODELS_DIR/mobilenet_v2.onnx||no extra precision flags (trtexec defaults some layers to fp16/tf32 automatically - see log for the actual layer precisions)"
  "fp16|$MODELS_DIR/mobilenet_v2.onnx|--fp16|explicit --fp16"
  "int8|$MODELS_DIR/mobilenet_v2.int8_trt.onnx|--int8|reuses phase-1 QDQ scales (Conv-only quantization), no separate TRT calibration"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name onnx_path flags notes <<< "$cfg"

  if [ ! -f "$onnx_path" ]; then
    echo "skip $name: $onnx_path not found" >&2
    continue
  fi

  engine="$ENGINES_DIR/mobilenet_v2_${name}.engine"
  log="$LOG_DIR/${name}.log"

  echo "== building/running $name engine ==" >&2
  # shellcheck disable=SC2086
  trtexec --onnx="$onnx_path" --saveEngine="$engine" $flags \
    --avgRuns=100 --iterations=200 --warmUp=200 2>&1 | tee "$log"

  # trtexec prints a line like:
  #   [I] GPU Compute Time: min = 1.10 ms, max = 1.55 ms, mean = 1.22 ms, median = 1.19 ms, percentile(99%) = 1.44 ms
  #   [I] Throughput: 812.4 qps
  # Parsing is version-dependent - if these greps come back empty, open the
  # log and adjust the sed pattern to match your trtexec's actual wording.
  median_ms="$(grep -oP 'GPU Compute Time:.*median = \K[0-9.]+' "$log" | tail -1)"
  throughput="$(grep -oP 'Throughput:\s*\K[0-9.]+' "$log" | tail -1)"
  engine_size_mb="$(python3 -c "import pathlib; print(round(pathlib.Path('$engine').stat().st_size/1e6, 2))" 2>/dev/null || echo "")"

  if [ -z "$median_ms" ]; then
    echo "  warning: could not parse median latency from $log - inspect it and add the row manually" >&2
    continue
  fi

  python3 "$BENCH_PY" add \
    phase=2 \
    platform="$PLATFORM" \
    config="TensorRT $name" \
    backend=trtexec \
    latency_median_ms="$median_ms" \
    throughput_ips="${throughput:-}" \
    model_size_mb="${engine_size_mb:-}" \
    notes="GPU compute time only (excludes H2D/D2H copy); $notes"
done

echo "done. see ../results.csv / ../README.md, full logs in $LOG_DIR" >&2
