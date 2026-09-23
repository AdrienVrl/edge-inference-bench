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
    args = p.parse_args()

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
        f"{args.format}, per_channel={args.per_channel}, {args.calibration_method})"
    )


if __name__ == "__main__":
    main()
