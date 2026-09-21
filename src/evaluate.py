"""
evaluate.py
===========
Runs the trained Keras model against the held-out TEST set and reports:
  - overall accuracy, precision, recall, F1
  - full classification report
  - confusion matrix (printed + saved as PNG)
  - per-class accuracy
  - which classes get confused with which, and a plain-English guess why
  - model size / parameter count

Run:
    python src/evaluate.py --model models/waste_classifier.keras --img-size 160
"""
import argparse
import json
import os

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "dataset_split"


def load_test_set(img_size, batch_size=16):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = ds.class_names
    return ds, class_names


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/waste_classifier.keras")
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = tf.keras.models.load_model(args.model)

    test_ds, class_names = load_test_set(args.img_size)

    y_true, y_pred, y_prob = [], [], []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(tf.argmax(batch_y, axis=1).numpy().tolist())
        y_pred.extend(tf.argmax(preds, axis=1).numpy().tolist())
        y_prob.extend(tf.reduce_max(preds, axis=1).numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n=== OVERALL TEST METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("=== CONFUSION MATRIX ===")
    print("Rows = true class, Columns = predicted class")
    print(f"{'':<15}" + "".join(f"{c[:10]:>12}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<15}" + "".join(f"{v:>12}" for v in row))

    os.makedirs("models", exist_ok=True)
    plot_confusion_matrix(cm, class_names, "models/confusion_matrix.png")

    # Per-class accuracy
    print("\n=== PER-CLASS ACCURACY ===")
    per_class_acc = {}
    for i, cname in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i].sum()
        pc_acc = correct / total if total > 0 else 0.0
        per_class_acc[cname] = pc_acc
        print(f"{cname}: {pc_acc:.4f} ({correct}/{total})")

    # Misclassification analysis: find the biggest off-diagonal confusions
    print("\n=== TOP CONFUSIONS (true -> predicted) ===")
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    if confusions:
        for count, true_c, pred_c in confusions[:10]:
            print(f"  {true_c} -> {pred_c}: {count} images")
        print(
            "\nLikely causes: visually overlapping textures/colors between "
            "these classes (e.g. crumpled metallic wrappers vs. plastic film), "
            "similar backgrounds in the source photos, or too few training "
            "examples covering the object's full range of appearances "
            "(lighting, angle, packaging state)."
        )
    else:
        print("  No confusions -- every test image was classified correctly.")

    # Model size / params
    n_params = model.count_params()
    keras_size_mb = os.path.getsize(args.model) / (1024 * 1024)
    print("\n=== MODEL SIZE ===")
    print(f"Parameters: {n_params:,}")
    print(f"Keras (.keras) file size: {keras_size_mb:.2f} MB")

    # Save metrics to json for later reference (e.g. in README table)
    metrics = {
        "test_accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "per_class_accuracy": per_class_acc,
        "num_params": n_params,
        "keras_size_mb": keras_size_mb,
    }
    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics to models/evaluation_metrics.json")


if __name__ == "__main__":
    main()
