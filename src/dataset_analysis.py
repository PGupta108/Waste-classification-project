import os
from PIL import Image
import imagehash
from collections import defaultdict

ROOT = "dataset/dataset"

classes = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
print("Classes found:", classes)

report = {}
all_hashes = defaultdict(list)  # hash -> list of (class, filename)
corrupted = []
dims = defaultdict(list)
formats = defaultdict(int)
sizes_bytes = defaultdict(list)

for cls in classes:
    cls_dir = os.path.join(ROOT, cls)
    files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    n_valid = 0
    for f in files:
        path = os.path.join(cls_dir, f)
        try:
            with Image.open(path) as img:
                img.verify()
            # reopen since verify() closes usability for further ops
            with Image.open(path) as img:
                w, h = img.size
                fmt = img.format
                dims[cls].append((w, h))
                formats[fmt] += 1
                sizes_bytes[cls].append(os.path.getsize(path))
                # perceptual hash for near-dup detection
                try:
                    h_ = imagehash.average_hash(img.convert("RGB"))
                    all_hashes[str(h_)].append((cls, f))
                except Exception:
                    pass
            n_valid += 1
        except Exception as e:
            corrupted.append((cls, f, str(e)))
    report[cls] = {"total_files": len(files), "valid_images": n_valid}

print("\n=== PER-CLASS FILE COUNTS ===")
for cls in classes:
    print(f"{cls}: total={report[cls]['total_files']}  valid={report[cls]['valid_images']}")

print("\n=== IMAGE FORMATS (overall) ===")
for fmt, cnt in formats.items():
    print(f"{fmt}: {cnt}")

print("\n=== DIMENSION SUMMARY PER CLASS (min/max/most common) ===")
for cls in classes:
    d = dims[cls]
    if not d:
        continue
    ws = [x[0] for x in d]
    hs = [x[1] for x in d]
    from collections import Counter
    common = Counter(d).most_common(3)
    print(f"{cls}: n={len(d)} width[min={min(ws)},max={max(ws)}] height[min={min(hs)},max={max(hs)}] most_common_dims={common}")

print("\n=== FILE SIZE (KB) PER CLASS ===")
for cls in classes:
    s = sizes_bytes[cls]
    if not s:
        continue
    print(f"{cls}: min={min(s)/1024:.1f}KB max={max(s)/1024:.1f}KB avg={sum(s)/len(s)/1024:.1f}KB")

print("\n=== CORRUPTED / INVALID FILES ===")
if corrupted:
    for cls, f, err in corrupted:
        print(f"{cls}/{f}: {err}")
else:
    print("None found.")

print("\n=== POTENTIAL DUPLICATES (same perceptual hash) ===")
dup_count = 0
for h_, items in all_hashes.items():
    if len(items) > 1:
        dup_count += 1
        print(f"hash={h_}: {items}")
print(f"Total duplicate groups: {dup_count}")

print("\n=== CLASS BALANCE ===")
counts = {cls: report[cls]['valid_images'] for cls in classes}
total = sum(counts.values())
for cls, c in counts.items():
    print(f"{cls}: {c} images ({c/total*100:.1f}%)")
mx, mn = max(counts.values()), min(counts.values())
print(f"Imbalance ratio (max/min): {mx/mn:.2f}")
