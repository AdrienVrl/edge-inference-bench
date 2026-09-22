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
<!-- BENCH-TABLE:END -->

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
  `Laptop CPU (WSL2)`, `Quadro T1000 (WSL2)`, `ARM VM (Graviton2)`, `Nucleo F446RE @180 MHz`.

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
