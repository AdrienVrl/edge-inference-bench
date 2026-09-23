#!/usr/bin/env python3
"""reorganize_imagenet_val.py - turn the flat ILSVRC2012 val set into an
ImageFolder-compatible layout (one subfolder per class, named by WNID).
 
Needs the official devkit files (from ILSVRC2012_devkit_t12.tar.gz):
  - meta.mat                              maps ILSVRC2012_ID -> WNID
  - ILSVRC2012_validation_ground_truth.txt  one ILSVRC2012_ID per image,
                                             in filename-sorted order
 
By default creates symlinks (fast, no extra disk space); pass --copy to
duplicate files instead (e.g. if symlinks don't survive your setup).
 
Usage
-----
  python reorganize_imagenet_val.py \
      --images-dir /path/to/flat_val_images \
      --meta meta.mat \
      --ground-truth ILSVRC2012_validation_ground_truth.txt \
      --out /path/to/imagenet_val_sorted
"""

import argparse
import shutil
from pathlib import Path


def load_id_to_wnid(meta_path):
    from scipy.io import loadmat  # pip install scipy

    meta = loadmat(meta_path, squeeze_me=True)["synsets"]
    id_to_wnid = {}
    for entry in meta:
        # entry fields: ILSVRC2012_ID, WNID, words, ... ; only the first 1000
        # entries (the leaf/classification synsets) have ILSVRC2012_ID 1..1000
        ilsvrc_id = int(entry["ILSVRC2012_ID"])
        if 1 <= ilsvrc_id <= 1000:
            id_to_wnid[ilsvrc_id] = str(entry["WNID"])
    if len(id_to_wnid) != 1000:
        raise SystemExit(f"expected 1000 classes in meta.mat, found {len(id_to_wnid)}")
    return id_to_wnid


#!/usr/bin/env python3
"""reorganize_imagenet_val.py - turn the flat ILSVRC2012 val set into an
ImageFolder-compatible layout (one subfolder per class).

Handles three common ground-truth formats (auto-detected, or force with --format):

  loc-csv         Kaggle's LOC_val_solution.csv: "ImageId,PredictionString"
                  header, PredictionString starts with the WNID directly
                  (e.g. "n01751748 34 23 456 500 ..."). No --meta needed.
                  This is the ground truth for Kaggle's "ImageNet Object
                  Localization Challenge" dataset, which reuses the same
                  images/filenames as the official image-net.org ILSVRC2012
                  validation set.

  plain           one ILSVRC2012_ID (1-1000) per line, in filename-sorted image
                  order. This is the raw devkit's
                  ILSVRC2012_validation_ground_truth.txt. Needs --meta (meta.mat,
                  from ILSVRC2012_devkit_t12.tar.gz) to map ID -> WNID.

  filename-label  "<filename> <label>" per line (order doesn't matter, filename
                  is matched directly). Two sub-cases, also auto-detected:
                    - label in 0-999: already the standard sorted-synset index
                      used by pretrained torchvision/timm models. No --meta
                      needed; images are sorted into zero-padded "000".."999"
                      folders, which ImageFolder will assign 0-999 in that same
                      order.
                    - label in 1-1000: treated as ILSVRC2012_ID same as the
                      plain format; needs --meta.

By default creates symlinks (fast, no extra disk space); pass --copy to
duplicate files instead (e.g. if symlinks don't survive your setup).

Usage
-----
  # Kaggle LOC_val_solution.csv
  python reorganize_imagenet_val.py --images-dir /path/to/flat_val_images \
      --ground-truth LOC_val_solution.csv --out /path/to/imagenet_val_sorted

  # devkit's plain ground-truth file
  python reorganize_imagenet_val.py --images-dir /path/to/flat_val_images \
      --ground-truth ILSVRC2012_validation_ground_truth.txt --meta meta.mat \
      --out /path/to/imagenet_val_sorted

  # "filename label" val.txt style file
  python reorganize_imagenet_val.py --images-dir /path/to/flat_val_images \
      --ground-truth val.txt --out /path/to/imagenet_val_sorted
"""
import argparse
import csv
import shutil
from pathlib import Path


def load_id_to_wnid(meta_path):
    from scipy.io import loadmat  # pip install scipy

    meta = loadmat(meta_path, squeeze_me=True)["synsets"]
    id_to_wnid = {}
    for entry in meta:
        # entry fields: ILSVRC2012_ID, WNID, words, ... ; only the first 1000
        # entries (the leaf/classification synsets) have ILSVRC2012_ID 1..1000
        ilsvrc_id = int(entry["ILSVRC2012_ID"])
        if 1 <= ilsvrc_id <= 1000:
            id_to_wnid[ilsvrc_id] = str(entry["WNID"])
    if len(id_to_wnid) != 1000:
        raise SystemExit(f"expected 1000 classes in meta.mat, found {len(id_to_wnid)}")
    return id_to_wnid


def parse_loc_csv(csv_path):
    """Kaggle LOC_val_solution.csv -> dict[filename -> WNID]."""
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "ImageId" not in reader.fieldnames:
            raise SystemExit(
                f"{csv_path} doesn't look like LOC_val_solution.csv "
                f"(expected an 'ImageId' column, got {reader.fieldnames})"
            )
        for row in reader:
            wnid = row["PredictionString"].split()[0]  # first token of each box group
            mapping[row["ImageId"]] = (
                wnid  # ImageId has no extension, e.g. ILSVRC2012_val_00000001
            )
    return mapping


def parse_ground_truth(gt_path, fmt):
    """Returns (kind, data):
    kind='plain'          data = list[int]                       (ILSVRC2012_ID per line, image order)
    kind='wnid_by_stem'   data = dict[str filename-stem -> WNID]  (Kaggle CSV; no extension in key)
    kind='filename_index' data = dict[str filename -> int label]  (label meaning depends on max value)
    """
    if fmt == "auto" and Path(gt_path).suffix.lower() == ".csv":
        fmt = "loc-csv"
    if fmt == "loc-csv":
        return "wnid_by_stem", parse_loc_csv(gt_path)

    lines = [ln.strip() for ln in Path(gt_path).read_text().splitlines() if ln.strip()]
    first_tokens = lines[0].split()

    detected = "plain" if len(first_tokens) == 1 else "filename-label"
    fmt = detected if fmt == "auto" else fmt

    if fmt == "plain":
        return "plain", [int(ln) for ln in lines]

    mapping = {}
    for ln in lines:
        parts = ln.split()
        fname, label = parts[0], parts[-1]  # tolerate extra columns
        mapping[Path(fname).name] = int(label)
    return "filename_index", mapping


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--images-dir", required=True, help="folder of flat ILSVRC2012_val_*.JPEG files"
    )
    p.add_argument(
        "--meta", help="path to meta.mat (needed unless labels are already 0-999)"
    )
    p.add_argument(
        "--ground-truth",
        required=True,
        help="devkit ground-truth file, or a 'filename label' val.txt",
    )
    p.add_argument(
        "--format",
        choices=["auto", "loc-csv", "plain", "filename-label"],
        default="auto",
    )
    p.add_argument(
        "--out", required=True, help="output folder (created), ImageFolder-ready"
    )
    p.add_argument(
        "--copy", action="store_true", help="copy files instead of symlinking"
    )
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt_arg = args.format  # parse_ground_truth handles "auto" internally now
    kind, data = parse_ground_truth(args.ground_truth, fmt_arg)

    images = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.upper() in (".JPEG", ".JPG")
        and p.name.startswith("ILSVRC2012_val_")
    )
    if not images:
        raise SystemExit(
            f"no ILSVRC2012_val_*.JPEG files found under {args.images_dir}"
        )

    if kind == "wnid_by_stem":
        missing = [img.name for img in images if img.stem not in data]
        if missing:
            raise SystemExit(
                f"{len(missing)} images have no entry in {args.ground_truth} "
                f"(e.g. {missing[0]}). Check the CSV matches this image set."
            )
        pairs = [(img, data[img.stem]) for img in images]
        print(
            f"detected: Kaggle LOC-style CSV -> WNID read directly, folders = WNIDs, no --meta needed"
        )
    elif kind == "plain":
        if len(images) != len(data):
            raise SystemExit(
                f"images ({len(images)}) and ground-truth lines ({len(data)}) don't match. "
                "Check --images-dir points at the 50,000 flat val JPEGs."
            )
        if not args.meta:
            raise SystemExit(
                "--meta meta.mat is required for the 'plain' (ID-per-line) format"
            )
        id_to_wnid = load_id_to_wnid(args.meta)
        pairs = [(img, id_to_wnid[ilsvrc_id]) for img, ilsvrc_id in zip(images, data)]
        print(
            f"detected: plain ground-truth (ILSVRC2012_ID per line) -> using meta.mat, folders = WNIDs"
        )
    else:
        missing = [img.name for img in images if img.name not in data]
        if missing:
            raise SystemExit(
                f"{len(missing)} images have no entry in {args.ground_truth} "
                f"(e.g. {missing[0]}). Check the file matches this image set."
            )
        max_label = max(data.values())
        min_label = min(data.values())
        if max_label <= 999:
            # already 0-999, standard sorted-synset index used by pretrained
            # torchvision/timm models -> use zero-padded numeric folders directly,
            # ImageFolder will sort "000".."999" into the same 0-999 order
            pairs = [(img, f"{data[img.name]:03d}") for img in images]
            print(
                f"detected: filename-label ground-truth, labels {min_label}-{max_label} "
                f"-> already 0-999 sorted-class index, no --meta needed, folders = zero-padded index"
            )
        else:
            if not args.meta:
                raise SystemExit(
                    f"labels go up to {max_label} (>999), so they look like 1-indexed "
                    "ILSVRC2012_IDs, not ready-to-use class indices. --meta meta.mat is "
                    "required to map these to WNIDs."
                )
            id_to_wnid = load_id_to_wnid(args.meta)
            pairs = [(img, id_to_wnid[data[img.name]]) for img in images]
            print(
                f"detected: filename-label ground-truth, labels {min_label}-{max_label} "
                f"-> treated as ILSVRC2012_ID, using meta.mat, folders = WNIDs"
            )

    made_dirs = set()
    for img, class_name in pairs:
        class_dir = out_dir / class_name
        if class_name not in made_dirs:
            class_dir.mkdir(exist_ok=True)
            made_dirs.add(class_name)
        dest = class_dir / img.name
        if dest.exists():
            continue
        if args.copy:
            shutil.copy2(img, dest)
        else:
            dest.symlink_to(img.resolve())

    print(
        f"sorted {len(pairs)} images into {len(made_dirs)} class folders under {out_dir}"
    )
    print(
        "sanity check: torchvision.datasets.ImageFolder(out_dir).classes[:3] should be either "
        "sorted WNIDs (e.g. ['n01440764', 'n01443537', 'n01484850']) or ['000', '001', '002']"
    )


if __name__ == "__main__":
    main()
