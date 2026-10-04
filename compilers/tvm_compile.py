#!/usr/bin/env python3
"""tvm_compile.py - compile an ONNX model with Apache TVM, benchmark it, and
append a row to the main results table via bench.py add.

Auto-detects which IR your TVM build has: Relay (the long-standing, stable
IR) or Relax (its replacement in newer/nightly builds - Relay has been
dropped entirely from some recent wheels). Whichever is present, this
script uses it; it prints which one it picked.

Default (fast, always works): untuned build, just to get a baseline TVM
row next to your ORT rows.

  python tvm_compile.py --model ../models/mobilenet_v2.onnx \
      --platform "Laptop CPU (WSL2)" --config "TVM (untuned)"

--tune attempts MetaSchedule autotuning on top (slower, best-effort - the
tuning entry points have different names under Relay vs. Relax and have
changed across TVM versions; this is wrapped in a try/except that falls
back to the untuned build with a warning if it doesn't match your
installed version):

  python tvm_compile.py --model ../models/mobilenet_v2.onnx \
      --platform "Laptop CPU (WSL2)" --config "TVM (MetaSchedule tuned)" \
      --tune --trials 200

`--target "llvm -mcpu=native"` (the default) autodetects whichever CPU
it's run on, so this script also works unmodified on the ARM VM - copy
it and bench.py over, pip install apache-tvm there, run the same
commands, then scp results.csv back and merge the new rows.

If TVM's API has moved again since this was written and neither path
works, check `python3 -c "import tvm; print(tvm.__version__)"` against
the current docs at https://tvm.apache.org/docs/ rather than trusting
this script's exact call shape - this is explicitly one of the more
volatile corners of the TVM stack right now.
"""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx


def get_input_name_and_shape(onnx_model, input_size):
    inp = onnx_model.graph.input[0]
    return inp.name, (1, 3, input_size, input_size)


def to_device_array(x, dev):
    """Construct a TVM array from a numpy array, trying several API
    locations in order. TVM 0.27's Relax/FFI rewrite has moved this across
    at least a couple of names; tvm.nd doesn't exist at all in some builds
    (not just unloaded). If none of these work, it prints what IS
    available on your install so we can target the right one instead of
    guessing again."""
    import tvm

    try:
        return tvm.nd.array(x, dev)
    except AttributeError:
        pass
    try:
        from tvm.runtime import ndarray as tvm_ndarray

        return tvm_ndarray.array(x, dev)
    except (ImportError, AttributeError):
        pass
    try:
        return tvm.runtime.tensor(
            x, dev
        )  # some 0.2x builds expose a lowercase factory fn
    except AttributeError:
        pass
    try:
        return tvm.runtime.Tensor.from_numpy(x).copyto(dev)  # newer tvm_ffi Tensor type
    except AttributeError:
        pass

    candidates = [
        n
        for n in dir(tvm.runtime)
        if any(k in n.lower() for k in ("array", "tensor", "nd"))
    ]
    raise RuntimeError(
        "Could not find a working TVM array constructor for this version "
        f"(tvm.__version__={tvm.__version__}). Candidates found in tvm.runtime: "
        f"{candidates!r}. Try one of these directly, e.g. "
        "`tvm.runtime.<name>(x, dev)` or `tvm.runtime.<name>.from_numpy(x)`, "
        "and tell me which one exists so I can fix this properly."
    )


def parse_target(target_str):
    """Accept either the old CLI-style target string ("llvm -mcpu=native")
    or a JSON dict string ('{"kind": "llvm", "mcpu": "native"}'), and return
    something tvm.target.Target() accepts. Recent TVM builds dropped
    support for the CLI-style string entirely (another Relax-era API
    change, unrelated to the relay/relax one) - this keeps the familiar
    flag syntax working regardless of which parser your TVM has."""
    s = target_str.strip()
    if s.startswith("{"):
        import json

        return json.loads(s)
    parts = s.split()
    d = {"kind": parts[0]}
    for tok in parts[1:]:
        tok = tok.lstrip("-")
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
        else:
            d[tok] = True
    return d


def has_relay():
    try:
        from tvm import relay  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Relay path (older, stable TVM builds)
# --------------------------------------------------------------------------- #
def build_relay(onnx_model, shape_dict, target, tune, trials):
    import tvm
    from tvm import relay, transform
    from tvm.contrib import graph_executor

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    tuned = False
    lib = None
    if tune:
        try:
            from tvm import meta_schedule as ms

            with tempfile.TemporaryDirectory() as work_dir:
                database = ms.relay_integration.tune_relay(
                    mod=mod,
                    params=params,
                    target=target,
                    work_dir=work_dir,
                    max_trials_global=trials,
                )
                lib = ms.relay_integration.compile_relay(
                    database=database, mod=mod, target=target, params=params
                )
            tuned = True
        except Exception as e:  # noqa: BLE001 - see module docstring
            print(
                f"[warn] MetaSchedule (relay) tuning failed ({type(e).__name__}: {e}); "
                f"falling back to untuned build.",
                file=sys.stderr,
            )

    if lib is None:
        with transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    dev = tvm.cpu(0)  # this script is CPU-only; tvm.device(str(target.kind), 0)
    # is not reliable across TVM versions ("llvm" isn't
    # always accepted as a device-type alias for "cpu")
    module = graph_executor.GraphModule(lib["default"](dev))

    def run(x):
        module.set_input(shape_dict_input_name, x)
        module.run()

    return lib, run, tuned


# --------------------------------------------------------------------------- #
# Relax path (newer/nightly TVM builds, Relay removed)
# --------------------------------------------------------------------------- #
def build_relax(onnx_model, shape_dict, target, tune, trials):
    import tvm
    from tvm.relax.frontend.onnx import from_onnx as relax_from_onnx

    mod = relax_from_onnx(onnx_model, shape_dict=shape_dict)

    # Without fusion, every op (conv, bias-add, batchnorm sub-ops, the
    # ReLU6/Clip, ...) lowers to its own separate TIR kernel, each one
    # materializing its full output to memory before the next op reads it
    # back in - same idea as phase 1's Conv+Clip fusion finding in ORT,
    # compounding across many more unfused ops here. AnnotateTIROpPattern
    # must run before FuseOps (it tags ops with the pattern FuseOps reads);
    # FuseTIR then actually merges the fused Relax ops into single TIR
    # PrimFuncs. Applied defensively in case a name is missing on some
    # build - confirmed present on 0.27.0.post1.
    applied = []
    for pass_name in ("LegalizeOps", "AnnotateTIROpPattern", "FuseOps", "FuseTIR"):
        pass_fn = getattr(tvm.relax.transform, pass_name, None)
        if pass_fn is None:
            print(
                f"[warn] relax.transform.{pass_name} not found on this TVM build - skipping",
                file=sys.stderr,
            )
            continue
        mod = pass_fn()(mod)
        applied.append(pass_name)
    print(f"relax passes applied: {', '.join(applied)}")

    tuned = False
    ex = None
    if tune:
        try:
            from tvm import meta_schedule as ms

            with tempfile.TemporaryDirectory() as work_dir:
                database = ms.relax_integration.tune_relax(
                    mod=mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_global=trials,
                )
                ex = ms.relax_integration.compile_relax(
                    database=database, mod=mod, target=target
                )
            tuned = True
        except Exception as e:  # noqa: BLE001 - see module docstring
            print(
                f"[warn] MetaSchedule (relax) tuning failed ({type(e).__name__}: {e}); "
                f"falling back to untuned build. Relax's MetaSchedule entry points are "
                f"newer and less stable than Relay's - this is the most likely spot to "
                f"need adjusting for your exact TVM version.",
                file=sys.stderr,
            )

    if ex is None:
        ex = tvm.relax.build(mod, target=target)

    dev = tvm.cpu(0)  # see note in build_relay
    vm = tvm.relax.VirtualMachine(ex, dev)

    def run(x):
        vm["main"](to_device_array(x, dev))

    return ex, run, tuned


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--platform", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--phase", default="4")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--target", default="llvm -mcpu=native")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument(
        "--ir",
        choices=["auto", "relay", "relax"],
        default="auto",
        help="force a specific IR instead of auto-detecting",
    )
    p.add_argument(
        "--bench-py",
        default=str(Path(__file__).resolve().parent.parent / "utils/bench.py"),
    )
    p.add_argument("--no-record", action="store_true")
    args = p.parse_args()

    import tvm

    onnx_model = onnx.load(args.model)
    global shape_dict_input_name  # used by build_relay's closure
    input_name, shape = get_input_name_and_shape(onnx_model, args.input_size)
    shape_dict_input_name = input_name
    shape_dict = {input_name: shape}

    target = tvm.target.Target(parse_target(args.target))

    use_relay = (args.ir == "relay") or (args.ir == "auto" and has_relay())
    ir_name = "relay" if use_relay else "relax"
    print(f"using TVM IR: {ir_name} (tvm.__version__ = {tvm.__version__})")

    builder = build_relay if use_relay else build_relax
    artifact, run_once, tuned = builder(
        onnx_model, shape_dict, target, args.tune, args.trials
    )

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    for _ in range(args.warmup):
        run_once(x)

    times_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        run_once(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    med_ms = float(statistics.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    throughput = 1000.0 / med_ms

    with tempfile.TemporaryDirectory() as tmp:
        lib_path = Path(tmp) / "model.so"
        artifact.export_library(str(lib_path))
        size_mb = lib_path.stat().st_size / 1e6

    tuning_note = (
        f"MetaSchedule tuned, {args.trials} trials"
        if tuned
        else (
            "tuning requested but fell back to untuned (see stderr)"
            if args.tune
            else "untuned"
        )
    )
    print(
        f"[{args.config}] median {med_ms:.3f} ms | p95 {p95_ms:.3f} ms | "
        f"{throughput:.2f} img/s | size {size_mb:.2f} MB | IR={ir_name} | {tuning_note}"
    )

    if args.no_record:
        print("(--no-record: row not saved)")
        return

    cmd = [
        sys.executable,
        args.bench_py,
        "add",
        f"phase={args.phase}",
        f"platform={args.platform}",
        f"config={args.config}",
        "backend=tvm",
        f"latency_median_ms={med_ms:.4f}",
        f"latency_p95_ms={p95_ms:.4f}",
        f"throughput_ips={throughput:.2f}",
        f"model_size_mb={size_mb:.2f}",
        f"notes=target={args.target}; IR={ir_name}; {tuning_note}; "
        f"accuracy not separately evaluated here - same graph/weights as the "
        f"matching ORT FP32 row, see that row for accuracy",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
