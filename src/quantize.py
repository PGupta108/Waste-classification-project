"""
quantize.py
===========
Converts the trained Keras model to:
  1. Float32 TensorFlow Lite  (models/waste_classifier_float32.tflite)
  2. INT8 quantized TensorFlow Lite, full-integer I/O
     (models/waste_classifier_int8.tflite)

Uses post-training full-integer quantization with a representative dataset
drawn from the TRAIN split (never test, to avoid leaking test data into the
quantization calibration).

Also exports the INT8 model as a C byte array (model_data.cc / .h) for
environments where you want to embed it directly in firmware (e.g. as a
fallback if you ever move off the UNO Q's Linux side to a bare
microcontroller running TensorFlow Lite Micro).

Run:
    python src/quantize.py --model models/waste_classifier.keras --img-size 160
"""
import argparse
import os

import numpy as np
import tensorflow as tf

DATA_DIR = "dataset_split"


def representative_dataset_gen(img_size, n_samples=100):
    train_dir = os.path.join(DATA_DIR, "train")
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=1,
        label_mode=None,
        shuffle=True,
        seed=123,
    )
    count = 0
    for batch in ds:
        img = tf.cast(batch, tf.float32)
        yield [img]
        count += 1
        if count >= n_samples:
            break


def convert_float32(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return len(tflite_model)


def convert_int8(model, img_size, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(img_size)
    # Full integer quantization: both input and output tensors become
    # int8, which is required for TensorFlow Lite Micro / most MCU runtimes.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return len(tflite_model)


def write_c_array(tflite_path, out_dir, var_name="waste_classifier_model"):
    with open(tflite_path, "rb") as f:
        data = f.read()

    os.makedirs(out_dir, exist_ok=True)
    cc_path = os.path.join(out_dir, "model_data.cc")
    h_path = os.path.join(out_dir, "model_data.h")

    with open(cc_path, "w") as f:
        f.write('#include "model_data.h"\n\n')
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = data[i : i + 12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"  {line},\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

    with open(h_path, "w") as f:
        guard = "MODEL_DATA_H_"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"extern const unsigned char {var_name}[];\n")
        f.write(f"extern const unsigned int {var_name}_len;\n\n")
        f.write(f"#endif  // {guard}\n")

    return cc_path, h_path, len(data)


def evaluate_tflite_accuracy(tflite_path, img_size, is_int8):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=(img_size, img_size),
        batch_size=1,
        label_mode="categorical",
        shuffle=False,
    )

    correct, total = 0, 0
    if is_int8:
        in_scale, in_zero = input_details["quantization"]

    for batch_x, batch_y in test_ds:
        x = batch_x.numpy()
        if is_int8:
            x = x / in_scale + in_zero
            x = np.clip(np.round(x), -128, 127).astype(np.int8)
        else:
            x = x.astype(np.float32)

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])[0]
        pred = np.argmax(out)
        true = np.argmax(batch_y.numpy()[0])
        correct += int(pred == true)
        total += 1

    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/waste_classifier.keras")
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    keras_size = os.path.getsize(args.model)

    f32_path = "models/waste_classifier_float32.tflite"
    int8_path = "models/waste_classifier_int8.tflite"

    print("Converting to Float32 TFLite...")
    f32_size = convert_float32(model, f32_path)

    print("Converting to INT8 TFLite (full integer, with representative dataset)...")
    int8_size = convert_int8(model, args.img_size, int8_path)

    print("\nEvaluating accuracy of each format on the TEST set...")
    f32_acc = evaluate_tflite_accuracy(f32_path, args.img_size, is_int8=False)
    int8_acc = evaluate_tflite_accuracy(int8_path, args.img_size, is_int8=True)

    print("\n=== COMPARISON TABLE ===")
    print(f"{'Model':<20}{'Size (KB)':>12}{'Test Acc':>12}")
    print(f"{'Keras/Float32':<20}{keras_size/1024:>12.1f}{'(see evaluate.py)':>18}")
    print(f"{'Float32 TFLite':<20}{f32_size/1024:>12.1f}{f32_acc:>12.4f}")
    print(f"{'INT8 TFLite':<20}{int8_size/1024:>12.1f}{int8_acc:>12.4f}")

    print(f"\nSize reduction (Keras -> INT8): {keras_size/int8_size:.1f}x smaller")

    print("\nGenerating C byte array for microcontroller/TFLite Micro deployment...")
    cc_path, h_path, n_bytes = write_c_array(int8_path, "deployment/stm32_sketch")
    print(f"Wrote {cc_path} and {h_path} ({n_bytes} bytes, "
          f"~{n_bytes/1024:.1f} KB flash needed to store the model)")


if __name__ == "__main__":
    main()
