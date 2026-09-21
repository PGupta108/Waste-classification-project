"""
infer.py
========
Runs on the Arduino UNO Q's Linux (Qualcomm QRB2210 / Debian) side.

Camera -> resize/normalize -> INT8 TFLite inference -> confidence gate
-> decision layer -> RPC call to the STM32 side to move the sorting servo.

This is intentionally close to src/predict.py (same WasteClassifier class)
so behavior in testing matches behavior in deployment. The only new piece
here is wiring it to the actuator via the decision layer, and grabbing
frames from the UNO Q's camera connector instead of a laptop webcam.

Setup on the board (once you have it):
    # Arduino App Lab already ships Python + a Debian environment.
    pip install tensorflow-lite-runtime opencv-python-headless

Run:
    python3 infer.py --model models/waste_classifier_int8.tflite \
                      --confidence-threshold 0.70

Press Ctrl+C to stop.
"""
import argparse
import sys
import time

sys.path.insert(0, "../../src")  # reuse WasteClassifier from src/predict.py
from predict import WasteClassifier  # noqa: E402

from decision_layer import DecisionLayer, ClassificationResult

try:
    import cv2
except ImportError:
    cv2 = None


def get_rpc_client():
    """
    Returns an object with a `.call(method, **kwargs)` method that talks to
    the STM32 co-processor over Arduino App Lab's RPC bridge.

    Arduino App Lab exposes this via its Python "Bridge" library -- the
    exact import path depends on the App Lab version installed on your
    board. Two commonly used names are `arduino.app_bridge` and
    `app_lab.bridge`. Check `pip show` / App Lab's own examples once you
    have the board, and update the import below accordingly. Until then,
    this returns None, so decision_layer.py runs in dry-run / print mode.
    """
    try:
        from arduino.app_bridge import Bridge  # type: ignore
        return Bridge()
    except ImportError:
        print("NOTE: Arduino App Lab RPC bridge not found -- running in "
              "dry-run mode (prints actions instead of moving hardware).")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../../models/waste_classifier_int8.tflite")
    parser.add_argument("--class-names", default="../../models/class_names.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError("opencv-python(-headless) is required on the UNO Q side.")

    classifier = WasteClassifier(
        args.model, args.class_names, args.confidence_threshold
    )
    decision_layer = DecisionLayer(rpc_client=get_rpc_client())

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print("Starting real-time waste classification. Ctrl+C to stop.")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("WARNING: failed to grab frame, retrying...")
                time.sleep(0.5)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            label, confidence, latency_ms = classifier.predict(frame_rgb)

            print(f"Waste detected: {label.upper()}")
            print(f"Confidence: {confidence*100:.1f}%  (inference: {latency_ms:.1f} ms)")

            decision_layer.handle(ClassificationResult(label, confidence))

            time.sleep(0.2)  # ~5 classifications/sec; tune to your conveyor speed
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
