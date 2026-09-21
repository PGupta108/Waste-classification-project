"""
predict.py
==========
Real-time / single-image inference using the quantized INT8 TFLite model.

Pipeline: camera frame -> resize -> normalize/quantize -> TFLite Micro-style
inference -> confidence check -> class label or "Unknown / Low Confidence".

This is the SAME logic that later runs on the UNO Q's Linux side
(see deployment/uno_q_linux/infer.py), just runnable here for quick testing
on your laptop with a webcam or a folder of test images.

Usage:
    # Single image
    python src/predict.py --model models/waste_classifier_int8.tflite --image path/to/img.jpg

    # Webcam (press 'q' to quit)
    python src/predict.py --model models/waste_classifier_int8.tflite --webcam
"""
import argparse
import json
import time

import numpy as np
import tensorflow as tf

try:
    import cv2
except ImportError:
    cv2 = None


class WasteClassifier:
    """Thin wrapper around a TFLite interpreter with a confidence gate.

    Deliberately has NO knowledge of servos/motors/actuators -- see
    section 13 of the spec ("clean interface"). It only produces
    (label, confidence) or ("Unknown / Low Confidence", confidence).
    The decision/actuator layer lives separately in
    deployment/uno_q_linux/decision_layer.py so you can swap the hardware
    mechanism later without touching this file.
    """

    def __init__(self, model_path, class_names_path="models/class_names.json",
                 confidence_threshold=0.70):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.img_size = self.input_details["shape"][1]  # assumes square input
        self.is_int8 = self.input_details["dtype"] == np.int8
        self.confidence_threshold = confidence_threshold

        with open(class_names_path) as f:
            self.class_names = json.load(f)

    def preprocess(self, frame_rgb):
        img = tf.image.resize(frame_rgb, (self.img_size, self.img_size)).numpy()
        img = np.expand_dims(img, axis=0)

        if self.is_int8:
            scale, zero_point = self.input_details["quantization"]
            img = img / scale + zero_point
            img = np.clip(np.round(img), -128, 127).astype(np.int8)
        else:
            img = img.astype(np.float32)
        return img

    def predict(self, frame_rgb):
        x = self.preprocess(frame_rgb)
        t0 = time.time()
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        latency_ms = (time.time() - t0) * 1000

        out = self.interpreter.get_tensor(self.output_details["index"])[0]

        if self.is_int8:
            scale, zero_point = self.output_details["quantization"]
            out = (out.astype(np.float32) - zero_point) * scale

        class_idx = int(np.argmax(out))
        confidence = float(out[class_idx])
        label = self.class_names[class_idx]

    # Always return the underlying prediction, but flag if it's below threshold
        is_confident = confidence >= self.confidence_threshold
        return label, confidence, latency_ms, is_confident

def run_on_image(classifier, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3).numpy()
    label, conf, latency, is_confident = classifier.predict(img)
    status = "CONFIDENT" if is_confident else "LOW CONFIDENCE"
    print(f"Predicted class: {label.upper()}")
    print(f"Confidence: {conf*100:.1f}% ({status})")
    print(f"Inference time: {latency:.1f} ms")


def run_on_webcam(classifier):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for --webcam mode. "
                            "Install with: pip install opencv-python")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0)")

    print("Press 'q' to quit.")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        label, conf, latency, is_confident = classifier.predict(frame_rgb)

        text = f"{label} ({conf*100:.1f}%) [{latency:.0f}ms]"
        cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Waste Classifier", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/waste_classifier_int8.tflite")
    parser.add_argument("--class-names", default="models/class_names.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--image", help="Path to a single image to classify")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam inference")
    args = parser.parse_args()

    classifier = WasteClassifier(
        args.model, args.class_names, args.confidence_threshold
    )

    if args.webcam:
        run_on_webcam(classifier)
    elif args.image:
        run_on_image(classifier, args.image)
    else:
        print("Specify either --image <path> or --webcam")


if __name__ == "__main__":
    main()
