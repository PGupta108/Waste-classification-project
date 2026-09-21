"""
train.py
========
Transfer-learning training pipeline for the waste classifier.

Architecture: MobileNetV2 (alpha=0.35), pretrained on ImageNet
  -> frozen base
  -> GlobalAveragePooling2D
  -> Dropout
  -> Dense(num_classes, softmax)

Two-phase training:
  Phase 1: base frozen, train only the head        (fast, prevents destroying
                                                      pretrained features on a
                                                      tiny dataset)
  Phase 2: unfreeze the last N layers of the base, fine-tune at low LR

Run:
    python src/train.py --img-size 160 --alpha 0.35
"""
import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

DATA_DIR = "dataset_split"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def build_datasets(img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names


def make_augmentation():
    # Augmentations chosen to mimic real camera conditions (lighting,
    # slight movement, orientation, distance) WITHOUT destroying the
    # object's key visual features (no channel shuffling, no extreme
    # color inversion, no aggressive cutout).
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),        # ~30 degrees max, mild
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(img_size, num_classes, alpha):
    # Try to load ImageNet-pretrained weights (downloaded from
    # storage.googleapis.com on first run). This requires normal internet
    # access. If that host is unreachable (e.g. a locked-down CI/sandbox
    # network), we fall back to random initialization so the pipeline still
    # runs end-to-end -- but note that transfer learning is what makes this
    # work well on a ~276-image dataset, so pretrained weights are strongly
    # recommended for real training.
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            alpha=alpha,
            include_top=False,
            weights="imagenet",
        )
        print("Loaded ImageNet-pretrained MobileNetV2 weights.")
    except Exception as e:
        print(f"WARNING: could not download ImageNet weights ({e}).")
        print("Falling back to random initialization -- accuracy will be "
              "much lower. Re-run with internet access for real training.")
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            alpha=alpha,
            include_top=False,
            weights=None,
        )
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = make_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-head", type=int, default=20)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--finetune-layers", type=int, default=30,
                         help="Number of final base-model layers to unfreeze")
    args = parser.parse_args()

    print(f"Config: img_size={args.img_size} alpha={args.alpha} batch={args.batch_size}")

    train_ds, val_ds, test_ds, class_names = build_datasets(args.img_size, args.batch_size)
    num_classes = len(class_names)
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    model, base = build_model(args.img_size, num_classes, args.alpha)

    ckpt_path = os.path.join(MODEL_DIR, f"waste_classifier_{args.img_size}.keras")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    # ---------- Phase 1: train head only ----------
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== PHASE 1: training classification head (base frozen) ===")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, callbacks=cbs)

    # ---------- Phase 2: fine-tune top of base ----------
    base.trainable = True
    for layer in base.layers[: -args.finetune_layers]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== PHASE 2: fine-tuning top layers of MobileNetV2 ===")
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, callbacks=cbs)

    # ---------- Save final model + metadata ----------
    final_path = os.path.join(MODEL_DIR, "waste_classifier.keras")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    with open(os.path.join(MODEL_DIR, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # quick sanity check on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nQuick test accuracy right after training: {test_acc:.4f}")


if __name__ == "__main__":
    main()
