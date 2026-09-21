# Smart Waste Classification System

An edge-deployable image classifier for an IoT waste-sorting camera, built
around your uploaded dataset and targeting the **Arduino UNO Q (2GB)**.

## 1. Project Objective

Classify an image captured by a camera into a waste category (Metal /
Organic / Plastic, based on your actual dataset), then hand the result to a
decision layer that drives a physical sorting mechanism (servo/motor),
without coupling the ML model to that hardware.

## 2. Dataset

Your uploaded dataset (`dataset.zip`, ~104 MB) contains:

| Class | Images |
|---|---:|
| Metal waste | 96 |
| Organic waste | 93 |
| Plastic waste | 87 |
| **Total** | **276** |

Well balanced (imbalance ratio 1.10 — no class weighting needed), formats
JPEG/PNG/WEBP, resolutions ranging from 300×319 to 4000×6000, no corrupted
files. A handful of likely burst-shot near-duplicates were flagged but kept
per your instruction.

**Important caveat:** 276 images is a small dataset for deep learning.
Transfer learning (a frozen ImageNet-pretrained backbone) is what makes
this workable at all — training a CNN from scratch on this much data would
badly overfit. Expect solid but not bulletproof real-world accuracy, and
strongly consider collecting more images (especially straight from your
actual IoT camera — see "Domain Shift" below) before deploying for good.

Re-run the analysis or the split at any time:
```bash
python src/dataset_analysis.py   # prints the full report used above
python src/preprocess.py         # regenerates dataset_split/{train,val,test}
```

## 3. Installation

```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow will download ImageNet-pretrained MobileNetV2 weights on first
run of `train.py` — this requires normal internet access to
`storage.googleapis.com`. (If you see a 403/blocked download, you're
likely behind a restrictive proxy/firewall; the script will still run,
falling back to random-initialized weights, but accuracy will be much
worse — pretrained weights are what make this work on so little data.)

## 4. Training

```bash
python src/train.py --img-size 160 --alpha 0.35
```

Architecture: **MobileNetV2** (width multiplier `alpha=0.35`), chosen over
EfficientNet-Lite0 because it has far fewer parameters (better fit for a
276-image dataset, lower overfitting risk) and more mature TFLite Micro
support, at the cost of a slightly lower theoretical accuracy ceiling.

Training is two-phase:
1. Base frozen, train the classification head (fast, protects the
   pretrained features from a tiny dataset).
2. Unfreeze the top ~30 layers, fine-tune at a low learning rate.

Try `--img-size 128` or `--img-size 224` and compare `models/evaluation_metrics.json`
across runs to pick the best size/accuracy trade-off for your case — 160 is
a reasonable default starting point.

Outputs: `models/waste_classifier.keras`, `models/class_names.json`,
`models/train_config.json`.

## 5. Evaluation

```bash
python src/evaluate.py --model models/waste_classifier.keras --img-size 160
```

Reports test accuracy/precision/recall/F1, a full classification report,
confusion matrix (printed + saved as `models/confusion_matrix.png`),
per-class accuracy, and the biggest confusions with likely causes.

## 6. Model Conversion & Quantization

```bash
python src/quantize.py --model models/waste_classifier.keras --img-size 160
```

Produces:

| Model | What it is |
|---|---|
| `models/waste_classifier.keras` | Original Keras/float32 model |
| `models/waste_classifier_float32.tflite` | TFLite, no quantization |
| `models/waste_classifier_int8.tflite` | **Full-integer INT8** TFLite — both input and output tensors are int8, using a representative-dataset calibration drawn from the *train* split (never test, to avoid leakage) |

The script prints a size/accuracy comparison table and, since INT8 is what
you deploy, evaluates it on the same held-out test set as the Keras model
so you can see exactly how much (if any) accuracy quantization cost you.
In our smoke test the INT8 conversion was **~7.6x smaller** than the Keras
model with no additional accuracy loss beyond quantization noise — re-run
this after real training with pretrained weights to get your real numbers.

It also exports the INT8 model as a C byte array
(`deployment/stm32_sketch/model_data.cc/.h`) as an **optional fallback**
for a future bare-metal-microcontroller port — see below for why the UNO Q
doesn't need this for its primary deployment path.

## 7. Hardware: Arduino UNO Q (2GB)

The UNO Q is not a classic memory-constrained microcontroller — it's a
**dual-brain board**: a Qualcomm QRB2210 quad-core CPU running full Debian
Linux (2GB LPDDR4 RAM, 16GB eMMC), paired with an STM32U585 microcontroller
for real-time I/O, connected via Arduino App Lab's RPC bridge.

| Requirement | UNO Q 2GB | Notes |
|---|---|---|
| RAM | 2 GB | Runs the full TF Lite Python runtime comfortably |
| Storage | 16 GB eMMC | Room for OS, Python env, dataset, model |
| Camera | via USB-C or high-speed connector | Standard UVC webcam or Arduino-compatible camera module |
| Model runtime | Standard **TensorFlow Lite** (not TFLite Micro) | Because it has a full Linux OS |
| Real-time actuator control | STM32U585 side | Talks to the Linux side over the RPC bridge |

Because of the 2GB RAM and Linux OS, **you don't need TensorFlow Lite
Micro's bare-metal, no-OS runtime here** — that runtime exists specifically
for microcontrollers with kilobytes of RAM (a plain Arduino Uno/Nano, for
instance, genuinely cannot run this: no camera interface and far too
little RAM for even the INT8 model plus a camera frame buffer). The UNO Q
sidesteps that problem entirely.

**Architecture:**
```
Camera (USB-C/UVC)
   │
   ▼
UNO Q — Linux side (Qualcomm QRB2210, 2GB RAM)
  deployment/uno_q_linux/infer.py
   │  resize → normalize/quantize → TFLite inference → confidence gate
   ▼
deployment/uno_q_linux/decision_layer.py
  (label + confidence) → bin/servo mapping
   │  RPC call: set_servo_angle(angle)
   ▼
UNO Q — STM32U585 side (real-time)
  deployment/stm32_sketch/waste_sorter.ino
   │
   ▼
Servo → physical sorting mechanism
```

## 8. Running on the Board

```bash
# On the UNO Q's Linux side, inside Arduino App Lab's Python environment:
cd deployment/uno_q_linux
pip install tflite-runtime opencv-python-headless   # lighter than full tensorflow
python3 infer.py --model ../../models/waste_classifier_int8.tflite \
                  --confidence-threshold 0.70
```

Flash `deployment/stm32_sketch/waste_sorter.ino` to the STM32 side via the
Arduino IDE / App Lab as you would any Arduino sketch. **Note:** the exact
RPC binding syntax (`RPC.bind(...)`, `Bridge.expose(...)`, etc.) depends on
which Arduino App Lab / RPC library version ships with your board — the
sketch and `infer.py`'s `get_rpc_client()` both have clearly marked lines
to adapt once you check the App Lab examples on your specific unit. Until
you've wired that up, both sides run in a safe "dry run" mode that prints
what they would do.

## 9. Confidence Thresholding

Objects outside your training classes (or a bare conveyor belt, a hand,
etc.) will still produce *some* prediction — softmax always sums to 1.
`predict.py` / `infer.py` gate on a confidence threshold (default 0.70):
below it, the system reports `"Unknown / Low Confidence"` instead of
guessing. Tune `--confidence-threshold` based on your evaluation results —
lower it if too many real items get rejected, raise it if you're getting
confident-but-wrong misclassifications.

## 10. Domain Shift: Dataset Photos vs. Real Camera Feed

Your dataset images are almost certainly cleaner/better-lit/more
centered than what a cheap IoT camera on a conveyor will capture. This
mismatch (**domain shift**) is one of the most common reasons a model that
tests well on held-out dataset images performs worse in the field. Practical
mitigations, roughly in order of effort/impact:

1. **Capture ~20-30 sample images per class straight from your actual
   camera** (same lighting, angle, distance you'll use in production) and
   add them into `dataset/dataset/<class>/`, then re-run
   `preprocess.py` + `train.py`. This single step usually helps the most.
2. Fine-tune on the combined dataset rather than training from scratch again.
3. Keep the augmentation pipeline in `train.py` (brightness/contrast/zoom/
   rotation/translation) — it's specifically tuned to simulate camera
   variability without distorting the object's identifying features.
4. Control what you can physically: consistent lighting and a plain
   background behind the conveyor reduce domain shift more cheaply than
   any amount of extra ML work.

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `403 Forbidden` downloading MobileNetV2 weights | Restricted network/proxy blocking `storage.googleapis.com`. Try from an unrestricted connection; the script falls back to random-init weights otherwise, but accuracy will be poor. |
| Val/test accuracy stuck near random (~33% for 3 classes) | Check pretrained weights actually loaded (see console: "Loaded ImageNet-pretrained MobileNetV2 weights."); if not, see above. |
| INT8 model accuracy far below float32 | Increase representative-dataset sample count in `quantize.py` (`n_samples`), or check the representative data isn't accidentally empty. |
| Webcam won't open (`predict.py --webcam`) | Try a different `--camera-index`, and confirm no other app is holding the camera. |
| RPC calls to the STM32 side don't do anything | You haven't adapted the marked "ADAPT THIS" lines in `infer.py`/`waste_sorter.ino` to your specific Arduino App Lab version yet — check the app-lab examples that shipped with your board. |
| Predictions all say "Unknown / Low Confidence" | Confidence threshold may be too high for an under-trained model — this is expected on the smoke-test model in this repo (trained without internet access to fetch pretrained weights); re-train with real pretrained weights and this should resolve. |

## Project Structure

```
waste_classification/
├── dataset/dataset/<class>/...        # your raw uploaded images
├── dataset_split/{train,val,test}/<class>/...  # generated by preprocess.py
├── models/
│   ├── waste_classifier.keras
│   ├── waste_classifier_float32.tflite
│   ├── waste_classifier_int8.tflite
│   ├── class_names.json
│   ├── confusion_matrix.png
│   └── evaluation_metrics.json
├── src/
│   ├── dataset_analysis.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── quantize.py
│   └── predict.py
├── deployment/
│   ├── uno_q_linux/
│   │   ├── infer.py            # main real-time loop for the UNO Q Linux side
│   │   └── decision_layer.py   # ML <-> actuator interface
│   └── stm32_sketch/
│       ├── waste_sorter.ino    # servo control, real-time side
│       ├── model_data.cc/.h    # optional fallback for bare-metal TFLite Micro
│       └── README_fallback.md
├── requirements.txt
└── README.md
```
