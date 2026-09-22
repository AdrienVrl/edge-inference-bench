#!/usr/bin/env python3
"""bench.py - tiny benchmark harness for edge-inference-bench.
 
Every phase of the project adds rows to results.csv; the table in README.md is
regenerated from that file after each row.
 
Commands
--------
  export   Export a torchvision model to ONNX.
  run      Time a model (PyTorch or ONNX Runtime), optionally check accuracy,
           and append a row.
  add      Append a row measured elsewhere (trtexec, perf, DWT cycles, ...).
  table    Regenerate the README table from results.csv.
 
Examples
--------
  python bench.py export --model mobilenet_v2 --out models/mobilenet_v2.onnx --pretrained
  python bench.py run --phase 0 --backend torch --model mobilenet_v2 --device cpu \
      --platform "Laptop CPU (WSL2)" --config "PyTorch FP32" --pretrained
  python bench.py run --phase 1 --backend ort --model models/mobilenet_v2.onnx \
      --platform "Laptop CPU (WSL2)" --config "ORT FP32 (all opts)" \
      --dump-optimized models/mobilenet_v2.opt.onnx
  python bench.py add phase=2 platform="T1000 (WSL2)" config="TensorRT FP16" \
      backend=trtexec latency_median_ms=3.1 throughput_ips=322 notes="trtexec, GPU compute time"
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

try:  # Unix only (WSL2 / Linux / ARM VM all have it)
    import resource
except ImportError:  # pragma: no cover
    resource = None

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results.csv"
README_PATH = ROOT / "README.md"
PREDS_DIR = ROOT / "preds"
TABLE_START = "<!-- BENCH-TABLE:START -->"
TABLE_END = "<!-- BENCH-TABLE:END -->"

FIELDS = [
    "phase",
    "platform",
    "config",
    "backend",
    "latency_median_ms",
    "latency_p95_ms",
    "throughput_ips",
    "peak_mem_mb",
    "model_size_mb",
    "accuracy",
    "acc_metric",
    "batch",
    "threads",
    "versions",
    "notes",
]

# (csv field, table header)
TABLE_COLS = [
    ("phase", "Phase"),
    ("platform", "Platform"),
    ("config", "Config"),
    ("latency_median_ms", "Median (ms)"),
    ("latency_p95_ms", "p95 (ms)"),
    ("throughput_ips", "Throughput (img/s)"),
    ("peak_mem_mb", "Peak mem (MB)"),
    ("model_size_mb", "Size (MB)"),
    ("accuracy", "Accuracy"),
    ("notes", "Notes"),
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def percentile(values, p):
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def time_fn(fn, warmup, runs, sync=None):
    """Return a list of per-call latencies in ms."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def peak_rss_mb():
    if resource is None:
        return None
    # ru_maxrss is in KB on Linux (bytes on macOS)
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0 if sys.platform != "darwin" else kb / (1024.0 * 1024.0)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Backend:
    versions: dict = {}
    size_mb: float = 0.0
    notes: str = ""

    def prepare(self, x: np.ndarray):
        raise NotImplementedError

    def run(self, inp):
        raise NotImplementedError

    def to_numpy(self, out) -> np.ndarray:
        raise NotImplementedError

    def sync(self):
        return None

    def peak_mem_mb(self):
        return peak_rss_mb()

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.to_numpy(self.run(self.prepare(x)))


class TorchBackend(Backend):
    def __init__(self, name, device, threads, pretrained):
        import torch
        import torchvision

        self.torch = torch
        self.device = device
        torch.set_num_threads(threads)
        weights = "DEFAULT" if pretrained else None
        self.model = (
            getattr(torchvision.models, name)(weights=weights).eval().to(device)
        )
        self.size_mb = (
            sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
        )
        self.versions = {"torch": torch.__version__}
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def prepare(self, x):
        return self.torch.from_numpy(x).to(self.device)

    def run(self, inp):
        with self.torch.inference_mode():
            return self.model(inp)

    def to_numpy(self, out):
        return out.cpu().numpy()

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def peak_mem_mb(self):
        if self.device == "cuda":
            return self.torch.cuda.max_memory_allocated() / 1e6
        return peak_rss_mb()


class OrtBackend(Backend):
    OPT_LEVELS = {
        "disable": "ORT_DISABLE_ALL",
        "basic": "ORT_ENABLE_BASIC",
        "extended": "ORT_ENABLE_EXTENDED",
        "all": "ORT_ENABLE_ALL",
    }
    PROVIDERS = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    }

    def __init__(self, path, provider, threads, opt_level, dump_optimized):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, self.OPT_LEVELS[opt_level]
        )
        if dump_optimized:
            Path(dump_optimized).parent.mkdir(parents=True, exist_ok=True)
            so.optimized_model_filepath = str(dump_optimized)
        self.sess = ort.InferenceSession(
            str(path), so, providers=self.PROVIDERS[provider]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.size_mb = Path(path).stat().st_size / 1e6
        self.versions = {"onnxruntime": ort.__version__}
        self.notes = "providers=" + "+".join(
            p.replace("ExecutionProvider", "") for p in self.sess.get_providers()
        )
        if provider != "cpu":
            self.notes += "; timing includes host<->device copies"

    def prepare(self, x):
        return {self.input_name: x}

    def run(self, inp):
        return self.sess.run(None, inp)

    def to_numpy(self, out):
        return out[0]


def make_backend(a):
    if a.backend == "torch":
        return TorchBackend(a.model, a.device, a.threads, a.pretrained)
    if a.backend == "ort":
        return OrtBackend(a.model, a.provider, a.threads, a.ort_opt, a.dump_optimized)
    raise SystemExit(f"unknown backend {a.backend}")


# --------------------------------------------------------------------------- #
# Accuracy (optional)
# --------------------------------------------------------------------------- #
def eval_predictions(backend, val_dir, max_images, input_size):
    """Top-1 predictions on a deterministic, evenly spaced subset of an ImageFolder.

    For real top-1 the folder must be the standard ImageNet val layout
    (1000 wnid sub-folders), so that sorted folder order == torchvision class index.
    """
    from torchvision import datasets, transforms

    resize = int(round(input_size * 256 / 224))
    tf = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    ds = datasets.ImageFolder(str(val_dir), tf)
    n = min(max_images, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    preds, labels = [], []
    for i in idx:
        img, y = ds[int(i)]
        out = backend.infer(img.unsqueeze(0).numpy().astype(np.float32))
        preds.append(int(out.argmax()))
        labels.append(y)
    return np.array(preds), np.array(labels)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


# --------------------------------------------------------------------------- #
# CSV + README table
# --------------------------------------------------------------------------- #
def append_row(row):
    new_file = (not CSV_PATH.exists()) or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    render_table()


def _fmt(v, nd=2):
    if v in ("", None):
        return ""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _phase_key(p):
    try:
        return (0, float(p))
    except (TypeError, ValueError):
        return (1, 0.0)


def build_table():
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: _phase_key(r.get("phase")))  # stable: keeps insertion order

    header = "| " + " | ".join(h for _, h in TABLE_COLS) + " |"
    sep = "|" + "|".join("---" for _ in TABLE_COLS) + "|"
    lines = [header, sep]
    for r in rows:
        cells = []
        for key, _ in TABLE_COLS:
            if key == "accuracy":
                acc = _fmt(r.get("accuracy"), 1)
                metric = r.get("acc_metric", "")
                cell = f"{acc} ({metric})" if acc and metric else acc
            elif key in ("latency_median_ms", "latency_p95_ms"):
                cell = _fmt(r.get(key), 2)
            elif key in ("throughput_ips", "peak_mem_mb", "model_size_mb"):
                cell = _fmt(r.get(key), 1)
            else:
                cell = str(r.get(key, ""))
            cells.append(cell.replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_table():
    table = build_table()
    if not README_PATH.exists():
        print(table)
        return
    text = README_PATH.read_text()
    if TABLE_START not in text or TABLE_END not in text:
        print(
            f"[warn] table markers not found in {README_PATH.name}; printing table instead\n"
        )
        print(table)
        return
    pattern = re.compile(
        re.escape(TABLE_START) + r".*?" + re.escape(TABLE_END), re.DOTALL
    )
    new = f"{TABLE_START}\n{table}\n{TABLE_END}"
    README_PATH.write_text(pattern.sub(lambda _m: new, text))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_export(a):
    import torch
    import torchvision

    model = getattr(torchvision.models, a.model)(
        weights="DEFAULT" if a.pretrained else None
    ).eval()
    x = torch.randn(1, 3, a.input_size, a.input_size)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=a.opset)
    try:
        torch.onnx.export(model, x, str(out), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, x, str(out), **kwargs)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_run(a):
    backend = make_backend(a)
    x = (
        np.random.default_rng(0)
        .standard_normal((a.batch, 3, a.input_size, a.input_size))
        .astype(np.float32)
    )
    inp = backend.prepare(x)

    times = time_fn(lambda: backend.run(inp), a.warmup, a.runs, backend.sync)
    med = statistics.median(times)
    p95 = percentile(times, 95)
    mem = backend.peak_mem_mb()  # measured before accuracy eval loads a dataset

    notes = [n for n in (backend.notes, a.notes) if n]
    row = {
        "phase": a.phase,
        "platform": a.platform,
        "config": a.config,
        "backend": a.backend,
        "latency_median_ms": f"{med:.3f}",
        "latency_p95_ms": f"{p95:.3f}",
        "throughput_ips": f"{a.batch / (med / 1000.0):.2f}",
        "peak_mem_mb": f"{mem:.1f}" if mem is not None else "",
        "model_size_mb": f"{backend.size_mb:.2f}",
        "batch": a.batch,
        "threads": a.threads,
        "versions": " ".join(
            [f"py{platform.python_version()}"]
            + [f"{k}={v}" for k, v in backend.versions.items()]
        ),
    }

    if a.val_dir:
        preds, labels = eval_predictions(backend, a.val_dir, a.max_images, a.input_size)
        row["accuracy"] = f"{(preds == labels).mean() * 100:.2f}"
        row["acc_metric"] = f"top1@{len(preds)}"
        PREDS_DIR.mkdir(exist_ok=True)
        np.save(PREDS_DIR / f"{slug(a.config)}.npy", preds)
        if a.ref_preds:
            ref = np.load(a.ref_preds)
            if len(ref) == len(preds):
                notes.append(f"agree_with_ref={(preds == ref).mean() * 100:.1f}%")
            else:
                notes.append("ref_preds length mismatch (different --max-images?)")
    row["notes"] = "; ".join(notes)

    print(
        f"[{a.config}] median {med:.3f} ms | p95 {p95:.3f} ms | {row['throughput_ips']} img/s "
        f"| mem {row['peak_mem_mb']} MB | size {row['model_size_mb']} MB "
        f"| acc {row.get('accuracy', '-')}"
    )
    if a.no_record:
        print("(--no-record: row not saved)")
    else:
        append_row(row)


def cmd_add(a):
    row = {}
    for kv in a.pairs:
        if "=" not in kv:
            raise SystemExit(f"expected key=value, got: {kv}")
        k, v = kv.split("=", 1)
        if k not in FIELDS:
            raise SystemExit(f"unknown field '{k}'. Valid fields: {', '.join(FIELDS)}")
        row[k] = v
    for required in ("phase", "platform", "config"):
        if required not in row:
            raise SystemExit(f"missing required field: {required}")
    append_row(row)
    print("row added")


def cmd_table(_a):
    render_table()
    print("table regenerated")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="export a torchvision model to ONNX")
    e.add_argument("--model", default="mobilenet_v2")
    e.add_argument("--out", required=True)
    e.add_argument("--input-size", type=int, default=224)
    e.add_argument("--opset", type=int, default=17)
    e.add_argument("--pretrained", action="store_true")
    e.set_defaults(fn=cmd_export)

    r = sub.add_parser("run", help="benchmark a model and append a row")
    r.add_argument("--backend", choices=["torch", "ort"], required=True)
    r.add_argument(
        "--model", required=True, help="torchvision name (torch) or .onnx path (ort)"
    )
    r.add_argument("--config", required=True, help='row label, e.g. "ORT INT8 static"')
    r.add_argument(
        "--platform", default=platform.node(), help='e.g. "Laptop CPU (WSL2)"'
    )
    r.add_argument("--phase", default="0")
    r.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="torch backend only"
    )
    r.add_argument(
        "--provider",
        choices=["cpu", "cuda", "tensorrt"],
        default="cpu",
        help="ort backend only",
    )
    r.add_argument(
        "--ort-opt", choices=["disable", "basic", "extended", "all"], default="all"
    )
    r.add_argument(
        "--dump-optimized",
        help="ort: save the optimized graph here (open it in Netron)",
    )
    r.add_argument(
        "--pretrained", action="store_true", help="torch: load ImageNet weights"
    )
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--input-size", type=int, default=224)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--runs", type=int, default=200)
    r.add_argument(
        "--val-dir", help="ImageNet-style val folder (1000 class sub-folders) for top-1"
    )
    r.add_argument("--max-images", type=int, default=1000)
    r.add_argument(
        "--ref-preds", help="preds/*.npy from the FP32 run, to report top-1 agreement"
    )
    r.add_argument("--notes", default="")
    r.add_argument(
        "--no-record", action="store_true", help="print only, do not append a row"
    )
    r.set_defaults(fn=cmd_run)

    ad = sub.add_parser("add", help="append a row measured elsewhere (key=value ...)")
    ad.add_argument("pairs", nargs="+", metavar="key=value")
    ad.set_defaults(fn=cmd_add)

    t = sub.add_parser("table", help="regenerate the README table from results.csv")
    t.set_defaults(fn=cmd_table)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
