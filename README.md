# edge-inference-bench
Personal project to learn various edge AI techniques and compare them

One model, many ways to run it. Each phase adds rows to the table below:
quantization and graph optimization, GPU (CUDA/TensorRT), ARM SIMD (NEON),
ML compilers (TVM/MLIR), embedded Linux (Yocto), and a Cortex-M bonus.

The table is generated from `results.csv` by `bench.py`. Don't edit it by hand.

## Results

<!-- BENCH-TABLE:START -->
| Phase | Platform | Config | Median (ms) | p95 (ms) | Throughput (img/s) | Peak mem (MB) | Size (MB) | Accuracy | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Laptop CPU (WSL2) | PyTorch FP32 (v2) | 23.83 | 28.80 | 42.0 | 725.9 | 14.0 |  |  |
| 1 | Laptop CPU (WSL2) | ORT FP32 (all opts) | 4.26 | 6.53 | 234.7 | 108.6 | 14.0 |  | providers=CPU |
| 1 | Laptop CPU (WSL2) | ORT INT8 static (QDQ, per-channel) | 6.92 | 8.15 | 144.6 | 81.8 | 3.9 | 63.1 (top1@1000) | providers=CPU; agree_with_ref=73.5% |
| 1 | Laptop CPU (WSL2) | Unstructured L1 prune 30% | 4.16 | 6.55 | 240.2 | 98.9 | 14.0 | 23.5 (top1@1000) | providers=CPU; agree_with_ref=26.0% |
| 1 | Laptop CPU (WSL2) | Structured channel prune 30% | 5.84 | 7.19 | 171.2 | 93.4 | 8.0 | 0.0 (top1@1000) | providers=CPU; agree_with_ref=0.1% |
| 1 | Laptop CPU (WSL2) | Unstructured L1 prune 5% | 4.12 | 6.53 | 242.4 | 98.8 | 14.0 | 72.8 (top1@1000) | providers=CPU; agree_with_ref=95.3% |
| 1 | Laptop CPU (WSL2) | Structured channel prune 5% | 7.03 | 11.04 | 142.3 | 96.3 | 12.9 | 14.3 (top1@1000) | providers=CPU; agree_with_ref=14.9% |
| 1 | Laptop CPU (WSL2) | Structured channel prune 5% + BN restat | 10.67 | 12.79 | 93.7 | 96.4 | 12.9 | 53.7 (top1@1000) | providers=CPU; agree_with_ref=59.5% |
| 2 | Quadro T1000 (WSL2) | TensorRT fp32 | 1.22 |  | 602.7 |  | 14.4 |  | GPU compute time only (excludes H2D/D2H copy); no extra precision flags (trtexec defaults some layers to fp16/tf32 automatically - see log for the actual layer precisions) |
| 2 | Quadro T1000 (WSL2) | TensorRT fp16 | 1.17 |  | 561.2 |  | 10.0 |  | GPU compute time only (excludes H2D/D2H copy); explicit --fp16 |
| 2 | Quadro T1000 (WSL2) | TensorRT int8 | 0.79 |  | 818.4 |  | 8.1 |  | GPU compute time only (excludes H2D/D2H copy); reuses phase-1 QDQ scales (Conv-only quantization), no separate TRT calibration |
| 3 | ARM VM (Ampere Altra) | ORT FP32 | 52.08 | 68.65 | 19.2 | 93.2 | 14.0 | 73.4 (top1@1000) | providers=CPU; agree_with_ref=100.0% |
| 3 | ARM VM (Ampere Altra) | ORT INT8 static (Conv-only) | 77.60 | 99.56 | 12.9 | 90.2 | 7.5 | 68.2 (top1@1000) | providers=CPU; agree_with_ref=82.3% |
| 4 | Laptop CPU (WSL2) | TVM (MetaSchedule tuned) | 324.06 | 329.30 | 3.1 |  | 14.8 |  | target=llvm -mcpu=native; IR=relax; tuning requested but fell back to untuned (see stderr); accuracy not separately evaluated here - same graph/weights as the matching ORT FP32 row, see that row for accuracy |
| 4 | Laptop CPU (WSL2) | TVM (untuned | tvm | 306.64 | 316.8 | 3.3 |  | 14.8 |  |
| 5 | ARM VM (QEMU) |  ARM64 ORT | 5000.46 |  |  |  |  |  | QEMU TCG, not real time |
<!-- BENCH-TABLE:END -->

### Findings
- **INT8 static PTQ gave no CPU speedup** over FP32 (6.92ms vs. 4.26ms) — root cause: this CPU lacks AVX2/AVX512-VNNI, and QDQ format relies on ORT's CPU EP fusing Q/DQ into true INT8 ops, which it does inconsistently for MobileNetV2's depthwise convs.
- **Unstructured pruning behaved exactly as theory predicts**: shapes unchanged → no latency/size change, accuracy held (72.8% at 5%).
- **Structured pruning (torch-pruning) caused prediction collapse at both 30% and 5%** (up to 81% of predictions in one class), traced to stale BatchNorm statistics. A BN-restat pass (reset + forward passes on real images) recovered accuracy substantially but not fully (14.3%→53.7% at 5%, vs. 72.8% for unstructured) — residual gap likely from MobileNetV2's depthwise/grouped convs, a known sharp edge for channel-pruning tools.
- **TensorRT INT8 (0.79ms) beat FP16 (1.17ms) and FP32 (1.22ms)**: the T1000 has no tensor cores, so these gains come from reduced memory traffic/simpler math, not tensor-core throughput.
- **On the ARM VM, ORT INT8 was slower than FP32**: (77.6ms vs. 52.1ms), need further investigation
- **TVM (0.27.0.post1, Relax): ~70× slower than ORT, ~13× slower than PyTorch**: this build's meta_schedule (autotuner) is entirely absent, and without it Relax's default lowering appears to produce schedule-free TIR with no parallel/vectorize annotations.
- **ort-infer ran successfully inside the Yocto image (~5s/inference)**: confirms the full cross-compile→package→boot→link pipeline works, but the number is a QEMU TCG software-emulation artifact.

## Methodology

Fixed for every row, so numbers are comparable:

- **Model:** MobileNetV2 (torchvision), 224x224 input, batch 1 unless stated.
- **Timing:** 20 warm-up calls, then 200 timed calls; median and p95 reported.
  Random fixed-seed input, so the timing measures inference only (no image decoding).
- **Threads:** 4 by default (`--threads`), recorded in `results.csv`.
- **Memory:** peak process RSS for CPU runs; `torch.cuda.max_memory_allocated`
  for PyTorch CUDA runs. It includes the runtime and the model, not just the weights.
- **Accuracy:** top-1 on 1000 evenly spaced images from the ImageNet validation set
  (`--val-dir`, `--max-images`). If ImageNet isn't available, use top-1 agreement with the
  FP32 predictions (`--ref-preds preds/<fp32 config>.npy`) and say so in the notes.
- **GPU rows:** ORT CUDA/TensorRT timing includes host<->device copies unless stated;
  `trtexec` reports GPU compute time only. Both are labeled in the Notes column.
- **Versions:** Python and runtime versions are stored per row in `results.csv`.
- **Platforms:** always give the label a hardware description, e.g.
  `Laptop CPU (WSL2)`, `Quadro T1000 (WSL2)`, `ARM VM (Graviton2)`.

## Usage

```bash
pip install -r requirements.txt

# Phase 0: PyTorch FP32 baseline on CPU
python bench.py run --phase 0 --backend torch --model mobilenet_v2 --device cpu \
    --platform "Laptop CPU (WSL2)" --config "PyTorch FP32" --pretrained \
    --val-dir /path/to/imagenet/val

# Phase 1: export, then run with ONNX Runtime and dump the optimized graph
python bench.py export --model mobilenet_v2 --out models/mobilenet_v2.onnx --pretrained
python bench.py run --phase 1 --backend ort --model models/mobilenet_v2.onnx \
    --platform "Laptop CPU (WSL2)" --config "ORT FP32 (all opts)" \
    --dump-optimized models/mobilenet_v2.opt.onnx \
    --val-dir /path/to/imagenet/val --ref-preds preds/pytorch_fp32.npy

# Rows measured with other tools (trtexec, perf, DWT cycle counter, ...)
python bench.py add phase=2 platform="Quadro T1000 (WSL2)" config="TensorRT FP16" \
    backend=trtexec latency_median_ms=3.1 throughput_ips=322 \
    notes="trtexec GPU compute time"

python bench.py table   # regenerate the table from results.csv
```
