#!/usr/bin/env python3
"""prune.py - prune a torchvision model and export it to ONNX, in the same
shape bench.py's `export` command produces, so the output drops straight
into `bench.py run --backend ort ...`.
 
Two modes:
 
  unstructured   Magnitude-based weight pruning (torch.nn.utils.prune).
                  Zeroes individual weights but does NOT reshape any layer,
                  so on ordinary dense kernels this should NOT speed anything
                  up or shrink the ONNX file much - that's expected, and is
                  the point of including this row in the benchmark table.
 
  structured     Channel pruning via the `torch-pruning` library (traces the
                  dependency graph, including skip connections, and
                  physically removes channels). This DOES shrink the model
                  and should show a real latency/size drop. Requires
                  `pip install torch-pruning`.
 
Usage
-----
  python prune.py --model mobilenet_v2 --pretrained --mode unstructured \
      --amount 0.3 --out models/mobilenet_v2_pruned_unstructured.onnx
 
  python prune.py --model mobilenet_v2 --pretrained --mode structured \
      --amount 0.3 --out models/mobilenet_v2_pruned_structured.onnx
 
Then benchmark exactly like any other ONNX model:
  python bench.py run --phase 1 --backend ort --model models/mobilenet_v2_pruned_unstructured.onnx \
      --platform "Laptop CPU (WSL2)" --config "Unstructured L1 prune 30%" \
      --val-dir /path/to/imagenet/val --ref-preds preds/pytorch_fp32.npy
 
BatchNorm restat (structured pruning only)
-------------------------------------------
Structured pruning changes the distribution of surviving channels' activations,
but BatchNorm's running_mean/running_var were estimated for the ORIGINAL
(unpruned) activations. Left as-is, this alone can tank accuracy or cause
prediction collapse even at small pruning ratios. --restat-batches re-estimates
BN running stats by running forward passes in train() mode (BN's stat-tracking
mode - no labels, no gradients, no optimizer step; this is NOT fine-tuning):
 
  python prune.py --model mobilenet_v2 --pretrained --mode structured \
      --amount 0.3 --out models/mobilenet_v2_pruned_structured.onnx \
      --restat-batches 50 --calib-dir /path/to/some/images
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.prune as prune
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(name, pretrained, weights_path):
    model = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.eval()


def nonzero_fraction(model):
    total = zero = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            total += m.weight.numel()
            zero += (m.weight == 0).sum().item()
    return 1 - zero / total if total else float("nan")


# --------------------------------------------------------------------------- #
# Unstructured pruning
# --------------------------------------------------------------------------- #
def prune_unstructured(model, amount):
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    for m in convs:
        prune.l1_unstructured(m, name="weight", amount=amount)
        prune.remove(m, "weight")  # bake the mask into m.weight permanently
    print(f"unstructured L1 pruning: amount={amount} on {len(convs)} Conv2d layers")
    return model


# --------------------------------------------------------------------------- #
# Structured pruning (channel pruning, via torch-pruning)
# --------------------------------------------------------------------------- #
def prune_structured(model, amount, example_inputs):
    try:
        import torch_pruning as tp
    except ImportError:
        raise SystemExit(
            "structured mode needs the torch-pruning package: pip install torch-pruning"
        )

    # Keep the final classifier layer intact - pruning its output channels
    # would change the number of classes, not just the model's width.
    ignored = []
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        for m in classifier.modules():
            if isinstance(m, torch.nn.Linear):
                ignored.append(m)
    fc = getattr(model, "fc", None)  # ResNet-style head
    if isinstance(fc, torch.nn.Linear):
        ignored.append(fc)

    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=amount,
        ignored_layers=ignored,
    )
    before = sum(p.numel() for p in model.parameters())
    pruner.step()
    after = sum(p.numel() for p in model.parameters())
    print(
        f"structured channel pruning: ratio={amount}, params {before:,} -> {after:,} "
        f"({100 * (1 - after / before):.1f}% removed), ignored {len(ignored)} layer(s)"
    )
    return model


# --------------------------------------------------------------------------- #
# BatchNorm restat (re-estimate running_mean/running_var post-pruning)
# --------------------------------------------------------------------------- #
def _image_batches(calib_dir, input_size, num_batches, batch_size):
    """Yield (batch_size, 3, H, W) float tensors, preprocessed like bench.py's
    accuracy path. Falls back to fixed-seed random tensors if calib_dir is None
    -- fine for exercising BN's running code path, but real images give a much
    more meaningful restat since they match the true activation distribution."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    if calib_dir:
        from PIL import Image

        paths = sorted(
            p
            for p in Path(calib_dir).rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if not paths:
            raise SystemExit(f"no images found under {calib_dir}")
        resize = int(round(input_size * 256 / 224))
        needed = num_batches * batch_size
        idx = np.linspace(0, len(paths) - 1, min(needed, len(paths))).astype(int)
        imgs = []
        for i in idx:
            img = Image.open(paths[int(i)]).convert("RGB")
            w, h = img.size
            scale = resize / min(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
            w, h = img.size
            left, top = (w - input_size) // 2, (h - input_size) // 2
            img = img.crop((left, top, left + input_size, top + input_size))
            x = torch.from_numpy(
                np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )
            imgs.append((x.unsqueeze(0) - mean) / std)
        if len(imgs) < needed:  # not enough images: cycle to fill out the batches
            imgs = (imgs * (needed // len(imgs) + 1))[:needed]
        for b in range(num_batches):
            yield torch.cat(imgs[b * batch_size : (b + 1) * batch_size], dim=0)
    else:
        rng = np.random.default_rng(0)
        for _ in range(num_batches):
            x = rng.standard_normal((batch_size, 3, input_size, input_size)).astype(
                np.float32
            )
            yield torch.from_numpy(x)


def restat_batchnorm(model, num_batches, batch_size, input_size, calib_dir):
    """Re-estimate BN running_mean/running_var for the pruned model's actual
    activations. Resets running stats first so old (pre-pruning, wrong-shape-
    adjacent) values can't linger or bias the new estimate, then averages
    over num_batches forward passes in train() mode. No labels, no gradients,
    no optimizer step - this only touches BN buffers, not any weight."""
    if not calib_dir:
        print(
            "[warn] --restat-batches given without --calib-dir: restating BN on random "
            "noise. This exercises the code path but won't reflect real activation "
            "statistics - use --calib-dir for a result you can trust."
        )

    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()  # discard stats computed for the unpruned model

    model.train()
    with torch.no_grad():
        for i, batch in enumerate(
            _image_batches(calib_dir, input_size, num_batches, batch_size)
        ):
            model(batch)
    model.eval()
    print(
        f"BN restat: {num_batches} batches x {batch_size} images through {len(bns)} "
        f"BatchNorm2d layers ({'real images' if calib_dir else 'random noise'})"
    )
    return model


# --------------------------------------------------------------------------- #
# Export (same recipe as bench.py's cmd_export)
# --------------------------------------------------------------------------- #
def export_onnx(model, out_path, input_size, opset):
    x = torch.randn(1, 3, input_size, input_size)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset)
    try:
        torch.onnx.export(model, x, str(out), dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, x, str(out), **kwargs)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="mobilenet_v2", help="torchvision model name")
    p.add_argument(
        "--pretrained", action="store_true", help="load ImageNet weights before pruning"
    )
    p.add_argument(
        "--weights",
        help="optional local state_dict .pt to load instead of --pretrained",
    )
    p.add_argument("--mode", choices=["unstructured", "structured"], required=True)
    p.add_argument(
        "--amount",
        type=float,
        required=True,
        help="fraction to prune, e.g. 0.3 for 30%%",
    )
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument(
        "--save-state-dict",
        help="optionally also save the pruned model's state_dict here",
    )
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--restat-batches",
        type=int,
        default=0,
        help="re-estimate BatchNorm running stats after pruning over this many "
        "batches (0 = skip). Recommended for --mode structured; almost "
        "always needed to avoid an accuracy hit from stale BN stats.",
    )
    p.add_argument("--restat-batch-size", type=int, default=16)
    p.add_argument(
        "--calib-dir",
        help="folder of representative images for --restat-batches "
        "(same idea as quantize_static.py's --calib-dir)",
    )
    args = p.parse_args()

    model = load_model(args.model, args.pretrained, args.weights)

    if args.mode == "unstructured":
        model = prune_unstructured(model, args.amount)
        frac = nonzero_fraction(model)
        print(
            f"nonzero weight fraction after pruning: {frac:.3f} "
            f"(shapes unchanged - expect ~unchanged latency/size vs. the unpruned model)"
        )
    else:
        example_inputs = torch.randn(1, 3, args.input_size, args.input_size)
        model = prune_structured(model, args.amount, example_inputs)

    if args.restat_batches > 0:
        model = restat_batchnorm(
            model,
            args.restat_batches,
            args.restat_batch_size,
            args.input_size,
            args.calib_dir,
        )

    if args.save_state_dict:
        Path(args.save_state_dict).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_state_dict)
        print(f"wrote {args.save_state_dict}")

    export_onnx(model, args.out, args.input_size, args.opset)


if __name__ == "__main__":
    main()
