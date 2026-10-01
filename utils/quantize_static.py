#!/usr/bin/env python3
"""quantize_static.py - Static INT8 PTQ with ONNX Runtime.

Wraps onnxruntime.quantization.quantize_static with a CalibrationDataReader
that either pulls real images from an ImageFolder-style val dir, or (if
--val-dir is omitted) uses fixed-seed random tensors, matching bench.py's
--val-dir behaviour so both scripts stay comparable.

Usage
-----
  # with a calibration folder (a subset of ImageNet val, or any representative
  # images; doesn't need labels/class folders for calibration itself)
  python quantize_static.py --model models/mobilenet_v2.onnx \
      --out models/mobilenet_v2.int8.onnx --calib-dir /path/to/calib_images \
      --num-images 200

  # without real images (correctness/speed smoke test only - do not report
  # accuracy numbers from this, only latency/size)
  python quantize_static.py --model models/mobilenet_v2.onnx \
      --out models/mobilenet_v2.int8.onnx --num-images 200
"""

import argparse
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageCalibrationReader:
    """Feeds calibration inputs to quantize_static, one dict per get_next()."""

    def __init__(self, input_name, input_size, calib_dir=None, num_images=200, seed=0):
        self.input_name = input_name
        if calib_dir:
            self._batches = iter(self._from_images(calib_dir, input_size, num_images))
        else:
            self._batches = iter(self._from_random(input_size, num_images, seed))

    @staticmethod
    def _from_images(calib_dir, input_size, num_images):
        from PIL import Image

        paths = sorted(
            p
            for p in Path(calib_dir).rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if not paths:
            raise SystemExit(f"no images found under {calib_dir}")
        idx = np.linspace(0, len(paths) - 1, min(num_images, len(paths))).astype(int)
        resize = int(round(input_size * 256 / 224))
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        for i in idx:
            img = Image.open(paths[int(i)]).convert("RGB")
            # resize shorter side, then center-crop - same recipe as bench.py's eval_predictions
            w, h = img.size
            scale = resize / min(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
            w, h = img.size
            left, top = (w - input_size) // 2, (h - input_size) // 2
            img = img.crop((left, top, left + input_size, top + input_size))
            x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            x = (x - mean) / std
            yield x[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _from_random(input_size, num_images, seed):
        # No calibration folder given: same fixed-seed distribution as bench.py's
        # timing-only inputs. Fine to sanity-check that quantization runs and to
        # measure latency/size, but activation ranges won't reflect real images,
        # so don't trust --val-dir accuracy numbers from a model calibrated this way.
        rng = np.random.default_rng(seed)
        for _ in range(num_images):
            yield rng.standard_normal((1, 3, input_size, input_size)).astype(np.float32)

    def get_next(self):
        x = next(self._batches, None)
        return None if x is None else {self.input_name: x}

    def rewind(self):
        pass  # single pass is enough for calibration; not needed here


def get_input_name(model_path):
    import onnxruntime as ort

    return (
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        .get_inputs()[0]
        .name
    )


def find_classifier_nodes(model_path):
    """Return the name(s) of the last Gemm/MatMul node in the graph (in the
    node list's topological order) - almost always the final classifier
    layer in a torchvision model. Used to exclude just that node from
    quantization, which is the standard TensorRT+ORT QDQ workflow, rather
    than excluding a whole op type (which creates dangling DQ nodes feeding
    untouched ops like Clip - see the --op-types help text)."""
    import onnx

    model = onnx.load(str(model_path))
    matches = [n.name for n in model.graph.node if n.op_type in ("Gemm", "MatMul")]
    return matches[
        -1:
    ]  # last one only; extend the slice if a model has a multi-node head


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="input FP32 .onnx path")
    p.add_argument("--out", required=True, help="output quantized .onnx path")
    p.add_argument("--calib-dir", help="folder of representative images (recommended)")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument(
        "--format",
        choices=["qdq", "qoperator"],
        default="qdq",
        help="QDQ works better with TensorRT/newer ORT EPs; QOperator is the classic CPU path",
    )
    p.add_argument("--per-channel", action="store_true", default=True)
    p.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    p.add_argument(
        "--calibration-method",
        choices=["minmax", "entropy", "percentile"],
        default="minmax",
    )
    p.add_argument(
        "--op-types",
        default="all",
        help="comma-separated ONNX op types to quantize, or 'all' (default, ORT's "
        "own default: every supported op, including Clip/Add - this is what "
        "TensorRT's parser expects). Restricting this to e.g. 'Conv' looks like "
        "a fix for a quantized-Gemm/bias parser error, but it actually creates "
        "a WORSE problem: interior DequantizeLinear nodes end up feeding "
        "untouched ops like Clip (MobileNetV2's ReLU6), which TensorRT's "
        "explicit-quantization importer also rejects. Use --exclude-nodes "
        "instead to skip just the classifier layer.",
    )
    p.add_argument(
        "--exclude-nodes",
        help="comma-separated exact ONNX node names to exclude from quantization "
        "(passed as nodes_to_exclude). Takes priority over --auto-exclude-classifier.",
    )
    p.add_argument(
        "--auto-exclude-classifier",
        action="store_true",
        default=True,
        help="auto-detect and exclude the final Gemm/MatMul node (default on). "
        "This is the standard TensorRT+ORT workflow: quantize the whole graph, "
        "leave only the classification head in FP32 - avoids the INT32-bias "
        "DequantizeLinear that TensorRT's parser can't import for a quantized "
        "Gemm. Ignored if --exclude-nodes is given.",
    )
    p.add_argument(
        "--no-auto-exclude-classifier",
        dest="auto_exclude_classifier",
        action="store_false",
    )
    p.add_argument(
        "--symmetric-activations",
        action="store_true",
        default=True,
        help="force zero_point=0 for activation quantization (default on). "
        "TensorRT's explicit-quantization import only accepts a zero "
        "zero-point on GPU (non-zero is DLA-only); ORT's own default is "
        "asymmetric, which trtexec rejects with "
        "'Non-zero zero point is not supported'. Turn off with "
        "--no-symmetric-activations only for CPU-only models, where "
        "asymmetric can give slightly better accuracy.",
    )
    p.add_argument(
        "--no-symmetric-activations", dest="symmetric_activations", action="store_false"
    )
    p.add_argument(
        "--force-quantize-no-input-check",
        action="store_true",
        default=True,
        help="force Q/DQ insertion on every activation ORT would otherwise skip, "
        "including the graph's global input (default on). Without this, ORT "
        "can leave the network's raw input unquantized while still quantizing "
        "the first Conv's weight - a mixed quantized/float state TensorRT's "
        "explicit-quantization importer rejects (surfaces as a confusing "
        "'DequantizeLayer ... only activation datatypes allowed' error on "
        "that first Conv's weight-DequantizeLinear node).",
    )
    p.add_argument(
        "--no-force-quantize-no-input-check",
        dest="force_quantize_no_input_check",
        action="store_false",
    )
    p.add_argument(
        "--quantize-bias",
        action="store_true",
        default=False,
        help="quantize Conv/Gemm bias to INT32 (ORT's own default is True). Default "
        "here is False: bias stays FP32. Evidence from this project's own "
        "debugging - the node TensorRT's parser rejected had an int32, "
        "zero-point-0 DequantizeLinear feeding a Conv, which is ORT's standard "
        "bias-quantization signature, not a weight - so leaving bias unquantized "
        "sidesteps the whole pattern. TensorRT natively supports INT8 weight + "
        "INT8 activation + FLOAT bias on a Conv.",
    )
    p.add_argument("--no-quantize-bias", dest="quantize_bias", action="store_false")
    args = p.parse_args()
    op_types = None if args.op_types.lower() == "all" else args.op_types.split(",")

    if args.exclude_nodes:
        exclude_nodes = args.exclude_nodes.split(",")
        print(f"excluding node(s) from quantization (explicit): {exclude_nodes}")
    elif args.auto_exclude_classifier:
        exclude_nodes = find_classifier_nodes(args.model)
        if exclude_nodes:
            print(
                f"excluding node(s) from quantization (auto-detected classifier): {exclude_nodes}"
            )
        else:
            print(
                "[warn] --auto-exclude-classifier found no Gemm/MatMul node - "
                "nothing excluded; check the model if you expected a classifier head"
            )
    else:
        exclude_nodes = []

    if not args.calib_dir:
        print(
            "[warn] no --calib-dir given: calibrating on random tensors. "
            "Latency/size numbers are still valid, but do NOT report accuracy "
            "for a model quantized this way - use a real --calib-dir for that."
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    input_name = get_input_name(args.model)
    reader = ImageCalibrationReader(
        input_name, args.input_size, args.calib_dir, args.num_images
    )

    quantize_static(
        model_input=args.model,
        model_output=args.out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        op_types_to_quantize=op_types,
        nodes_to_exclude=exclude_nodes or None,
        extra_options={
            "ActivationSymmetric": args.symmetric_activations,
            "WeightSymmetric": True,  # ORT's own default for QInt8 weights; set explicitly for clarity
            "ForceQuantizeNoInputCheck": args.force_quantize_no_input_check,
            "QuantizeBias": args.quantize_bias,
        },
        calibrate_method={
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }[args.calibration_method],
    )

    in_size = Path(args.model).stat().st_size / 1e6
    out_size = Path(args.out).stat().st_size / 1e6
    print(
        f"wrote {args.out}  ({in_size:.1f} MB -> {out_size:.1f} MB, "
        f"{args.format}, per_channel={args.per_channel}, {args.calibration_method}, "
        f"op_types={op_types or 'all (ORT default)'}, "
        f"symmetric_activations={args.symmetric_activations}, "
        f"force_quantize_no_input_check={args.force_quantize_no_input_check}, "
        f"quantize_bias={args.quantize_bias}, "
        f"excluded_nodes={exclude_nodes or 'none'})"
    )


if __name__ == "__main__":
    main()
