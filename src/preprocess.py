"""
preprocess.py
=============
Prepares the raw dataset for training:
  1. Reads images from dataset/dataset/<class_name>/*
  2. Converts all images to RGB JPEG (fixes PNG-with-alpha, WEBP, CMYK, etc.
     issues that break TensorFlow's decoder or the TFLite Micro pipeline)
  3. Performs a STRATIFIED 70/15/15 train/val/test split (same class ratio
     in every split, no image appears in more than one split -> no leakage)
  4. Writes the result to dataset_split/{train,val,test}/<class_name>/*.jpg

Run:
    python src/preprocess.py
"""
import os
import random
import shutil
from PIL import Image

RAW_ROOT = "dataset/dataset"
OUT_ROOT = "dataset_split"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

random.seed(SEED)


def load_class_files(raw_root):
    classes = sorted(
        d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))
    )
    class_files = {}
    for cls in classes:
        cls_dir = os.path.join(raw_root, cls)
        files = [
            f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))
        ]
        class_files[cls] = files
    return classes, class_files


def stratified_split(files, ratios, seed=SEED):
    """Shuffle a list of filenames and split it according to ratios dict."""
    files = files[:]
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    # whatever remains goes to test, guarantees all files are used exactly once
    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def safe_convert_to_jpeg(src_path, dst_path):
    """Open any supported image (jpg/png/webp/etc), convert to RGB, save as jpeg."""
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img.save(dst_path, format="JPEG", quality=95)


def main():
    classes, class_files = load_class_files(RAW_ROOT)
    print(f"Found {len(classes)} classes: {classes}")

    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    summary = {split: {} for split in SPLITS}

    for cls in classes:
        files = class_files[cls]
        split_files = stratified_split(files, SPLITS)

        for split_name, split_list in split_files.items():
            out_dir = os.path.join(OUT_ROOT, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            count = 0
            for fname in split_list:
                src = os.path.join(RAW_ROOT, cls, fname)
                base, _ = os.path.splitext(fname)
                dst = os.path.join(out_dir, base + ".jpg")
                try:
                    safe_convert_to_jpeg(src, dst)
                    count += 1
                except Exception as e:
                    print(f"  [SKIP] {src}: {e}")
            summary[split_name][cls] = count

    print("\n=== SPLIT SUMMARY ===")
    header = f"{'class':<20}" + "".join(f"{s:>10}" for s in SPLITS)
    print(header)
    totals = {s: 0 for s in SPLITS}
    for cls in classes:
        row = f"{cls:<20}"
        for s in SPLITS:
            c = summary[s][cls]
            row += f"{c:>10}"
            totals[s] += c
        print(row)
    total_row = f"{'TOTAL':<20}" + "".join(f"{totals[s]:>10}" for s in SPLITS)
    print(total_row)
    print(f"\nGrand total images: {sum(totals.values())}")
    print(f"Output written to: {OUT_ROOT}/")


if __name__ == "__main__":
    main()
