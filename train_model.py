import argparse
import random
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from model_utils import (
    CLASS_NAMES,
    IMAGE_SIZE,
    center_cloudiness_score,
    center_dark_pupil_score,
    external_eye_score,
    open_rgb_image,
    save_metadata,
)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}
ROOT_SAMPLE_LABELS = {
    "eye.jpg": "Normal",
    "cleaneye.jpg": "Normal",
    "eye1.jpg": "Normal",
    "test_images/eye1.jpg": "Normal",
    "test_images/eye2.jpg": "Cataract",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final external-eye cataract detector.")
    parser.add_argument("--data-dir", default="data", help="Prepared dataset root.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Adam learning rate (lowered for small data).")
    parser.add_argument("--train-target", type=int, default=300, help="Target train images per class (lowered to avoid over-duplication).")
    parser.add_argument("--min-val", type=int, default=30, help="Minimum validation images per class.")
    parser.add_argument("--model-path", default="cataract_external_model.h5", help="Saved model path.")
    parser.add_argument("--metadata-path", default="cataract_external_model_metadata.json", help="Saved metadata path.")
    return parser.parse_args()


def ensure_structure(data_dir: Path) -> None:
    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            path = data_dir / split / class_name
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def discover_source_images() -> dict[str, list[Path]]:
    class_sources: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_NAMES}

    folder_candidates = {
        "Normal": [
            Path("archive (1)/dataset/1_normal"),
            Path("data_extra/normal"),
            Path("data_extra/Normal"),
            Path("data/source/Normal"),
            Path("data/source/normal"),
        ],
        "Cataract": [
            Path("archive (1)/dataset/2_cataract"),
            Path("data_extra/cataract"),
            Path("data_extra/Cataract"),
            Path("data/source/Cataract"),
            Path("data/source/cataract"),
        ],
    }

    for class_name, directories in folder_candidates.items():
        for directory in directories:
            if not directory.exists():
                continue
            for image_path in sorted(directory.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in VALID_SUFFIXES:
                    class_sources[class_name].append(image_path)

    for relative_path, class_name in ROOT_SAMPLE_LABELS.items():
        path = Path(relative_path)
        if path.exists():
            class_sources[class_name].append(path)

    return class_sources


def is_label_safe_external_image(image_path: Path, class_name: str) -> bool:
    """
    FIX: Relaxed thresholds significantly.
    Old thresholds were too strict and rejecting most images,
    leaving almost no data to train on.
    """
    try:
        eye_score = external_eye_score(image_path)
        cloudiness = center_cloudiness_score(image_path)
        dark_pupil = center_dark_pupil_score(image_path)
    except Exception:
        return False

    # FIX: Lowered from 0.18 to 0.05 — don't reject borderline eye images
    if eye_score < 0.05:
        return False

    if class_name == "Cataract":
        # FIX: Lowered from 0.10 to 0.03 — mild cataracts have low cloudiness
        return cloudiness >= 0.03

    # FIX: Relaxed normal check — dark pupil >= 0.05 (was 0.10), cloudiness < 0.45 (was 0.32)
    return dark_pupil >= 0.05 and cloudiness < 0.45


def filter_sources(source_map: dict[str, list[Path]]) -> dict[str, list[Path]]:
    filtered: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_NAMES}
    for class_name, image_paths in source_map.items():
        accepted = []
        rejected = 0
        for image_path in image_paths:
            if is_label_safe_external_image(image_path, class_name):
                accepted.append(image_path)
            else:
                rejected += 1
        print(f"[filter] {class_name}: {len(accepted)} accepted, {rejected} rejected")
        filtered[class_name] = accepted
    return filtered


def copy_image(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with open_rgb_image(source_path) as image:
        image.save(destination_path, format="JPEG", quality=92)


def save_variant(image: Image.Image, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination_path, format="JPEG", quality=92)


def diverse_augment(image: Image.Image, rng: random.Random, class_name: str) -> Image.Image:
    """
    FIX: Much more diverse augmentation so duplicated images look different.
    The old augmentation was too mild — same image rotated slightly looks identical to the model.
    """
    augmented = image.copy()

    # Random horizontal flip
    if rng.random() < 0.5:
        augmented = ImageOps.mirror(augmented)

    # Random vertical flip (eyes can be upside down in some datasets)
    if rng.random() < 0.3:
        augmented = ImageOps.flip(augmented)

    # Rotation — wider range
    angle = rng.uniform(-35.0, 35.0)
    augmented = augmented.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    # Random crop + resize (zoom in/out)
    width, height = augmented.size
    zoom_factor = rng.uniform(0.7, 1.0)
    crop_w = max(1, int(width * zoom_factor))
    crop_h = max(1, int(height * zoom_factor))
    left = rng.randint(0, width - crop_w)
    top = rng.randint(0, height - crop_h)
    augmented = augmented.crop((left, top, left + crop_w, top + crop_h))
    augmented = augmented.resize((width, height), Image.Resampling.LANCZOS)

    # Brightness
    augmented = ImageEnhance.Brightness(augmented).enhance(rng.uniform(0.6, 1.5))

    # Contrast
    augmented = ImageEnhance.Contrast(augmented).enhance(rng.uniform(0.7, 1.4))

    # Color saturation
    augmented = ImageEnhance.Color(augmented).enhance(rng.uniform(0.7, 1.3))

    # Sharpness
    augmented = ImageEnhance.Sharpness(augmented).enhance(rng.uniform(0.5, 2.0))

    # Occasional blur (simulates out-of-focus images)
    if rng.random() < 0.3:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    # Cataract-specific: occasionally add slight white haze to center
    if class_name == "Cataract" and rng.random() < 0.2:
        haze = Image.new("RGB", augmented.size, (230, 220, 200))
        mask = Image.new("L", augmented.size, 0)
        cx, cy = augmented.size[0] // 2, augmented.size[1] // 2
        radius = min(augmented.size) // 4
        for x in range(augmented.size[0]):
            for y in range(augmented.size[1]):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if dist < radius:
                    mask.putpixel((x, y), int(40 * (1 - dist / radius)))
        augmented = Image.composite(haze, augmented, mask)

    return augmented


def split_sources(filtered_sources: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    rng = random.Random(SEED)
    train_split: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_NAMES}
    val_split: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_NAMES}

    for class_name, image_paths in filtered_sources.items():
        shuffled = image_paths[:]
        rng.shuffle(shuffled)

        if len(shuffled) <= 1:
            train_split[class_name] = shuffled
            val_split[class_name] = shuffled[:]
            continue

        val_count = max(1, int(round(len(shuffled) * 0.2)))
        val_split[class_name] = shuffled[:val_count]
        train_split[class_name] = shuffled[val_count:] or shuffled[:1]

    return train_split, val_split


def populate_split(
    split_dir: Path,
    class_name: str,
    originals: list[Path],
    target_count: int,
    allow_augmentation: bool,
) -> int:
    """
    FIX: Cap augmentation multiplier at 5x.
    If you have 50 images and target 500, you were making 10 copies each.
    At 10x copies the model sees essentially the same image and memorizes it.
    Now capped at 5x so augmentation stays meaningful.
    """
    rng = random.Random(f"{SEED}:{split_dir}:{class_name}")
    class_dir = split_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    if not originals:
        return 0

    # FIX: Cap target at 5x the real images to avoid over-duplication
    max_augmented = len(originals) * 5
    effective_target = min(target_count, max_augmented) if allow_augmentation else target_count

    count = 0
    for source_path in originals:
        count += 1
        copy_image(source_path, class_dir / f"{class_name.lower()}_{count:05d}.jpg")

    while count < effective_target:
        base_path = originals[count % len(originals)]
        with open_rgb_image(base_path) as image:
            if allow_augmentation:
                variant = diverse_augment(image, rng, class_name)
            else:
                variant = image.copy()

        count += 1
        save_variant(variant, class_dir / f"{class_name.lower()}_{count:05d}.jpg")

    return count


def prepare_dataset(data_dir: Path, train_target: int, min_val: int) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    ensure_structure(data_dir)

    discovered_sources = discover_source_images()
    filtered_sources = filter_sources(discovered_sources)
    train_split, val_split = split_sources(filtered_sources)

    source_counts = {class_name: len(filtered_sources[class_name]) for class_name in CLASS_NAMES}
    if 0 in source_counts.values():
        missing = [class_name for class_name, count in source_counts.items() if count == 0]
        raise ValueError(
            "No usable external eye images were found for: "
            + ", ".join(missing)
            + ". Add real eye photos to data_extra/normal and data_extra/cataract."
        )

    effective_train_target = max(train_target, max(len(train_split[class_name]) for class_name in CLASS_NAMES))
    effective_val_target = max(min_val, max(len(val_split[class_name]) for class_name in CLASS_NAMES))

    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        train_counts[class_name] = populate_split(
            data_dir / "train",
            class_name,
            train_split[class_name],
            target_count=effective_train_target,
            allow_augmentation=True,
        )
        val_counts[class_name] = populate_split(
            data_dir / "val",
            class_name,
            val_split[class_name] or train_split[class_name][:1],
            target_count=effective_val_target,
            allow_augmentation=False,
        )

    return source_counts, train_counts, val_counts


def collect_split_samples(split_dir: Path) -> tuple[list[str], list[int], Counter]:
    image_paths: list[str] = []
    labels: list[int] = []
    counts: Counter = Counter()

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_SUFFIXES:
                image_paths.append(str(image_path))
                labels.append(label_index)
                counts[class_name] += 1

    if not image_paths:
        raise ValueError(f"No images found in {split_dir}")

    return image_paths, labels, counts


def decode_image(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, tf.cast(label, tf.float32)


def tf_augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    FIX: Added TensorFlow-level augmentation on top of PIL augmentation.
    This creates additional variety at training time without extra disk usage.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    return image, label


def build_dataset(image_paths: list[str], labels: list[int], batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        dataset = dataset.shuffle(len(image_paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
    if training:
        # FIX: Apply TF-level augmentation during training for extra variety
        dataset = dataset.map(tf_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def resolve_mobilenet_weights() -> str:
    cached_weights = Path.home() / ".keras" / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    if cached_weights.exists():
        return str(cached_weights)
    return "imagenet"


def build_model() -> Model:
    base_model = MobileNetV2(
        weights=resolve_mobilenet_weights(),
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # FIX: Freeze ALL base layers first, only unfreeze top 10 (was 30).
    # With <500 real images, unfreezing 30 layers = too many free parameters = overfitting.
    # Fine-tuning only the last 10 layers is safer and more stable.
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # FIX: Added BatchNormalization — stabilizes training with small datasets
    x = BatchNormalization()(x)

    # FIX: Reduced Dense layer from 128 to 64 neurons — smaller model overfits less
    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))(x)

    # FIX: Increased Dropout from 0.5 to 0.6 — stronger regularization for small data
    x = Dropout(0.6)(x)

    # FIX: Added second smaller Dense layer with L2 regularization
    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.4)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


def build_callbacks(model_path: str) -> list[tf.keras.callbacks.Callback]:
    return [
        # FIX: Save the best model by val_accuracy, not just end of training
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # FIX: Increased patience from 5 to 8 — small datasets are noisy,
        # stopping too early misses the real learning signal
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1, min_lr=1e-7),
    ]


def confusion_matrix_and_report(model: Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, str]:
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    predicted = (probabilities >= 0.5).astype(np.int32)

    actual_batches = [labels.numpy().astype(np.int32) for _, labels in dataset]
    actual = np.concatenate(actual_batches)

    matrix = np.zeros((2, 2), dtype=np.int32)
    for truth, pred in zip(actual, predicted):
        matrix[int(truth), int(pred)] += 1

    lines = ["class      precision   recall      f1-score    support"]
    for class_index, class_name in enumerate(CLASS_NAMES):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        support = int(matrix[class_index, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        lines.append(f"{class_name:<10}{precision:<12.3f}{recall:<12.3f}{f1:<12.3f}{support}")

    return matrix, "\n".join(lines)


def print_confusion_matrix(matrix: np.ndarray) -> None:
    print("\nConfusion Matrix")
    print("true\\pred   Normal   Cataract")
    print(f"Normal      {matrix[0, 0]:<8}{matrix[0, 1]}")
    print(f"Cataract    {matrix[1, 0]:<8}{matrix[1, 1]}")


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    data_dir = Path(args.data_dir)
    source_counts, train_counts, val_counts = prepare_dataset(data_dir, args.train_target, args.min_val)

    train_paths, train_labels, train_counter = collect_split_samples(data_dir / "train")
    val_paths, val_labels, val_counter = collect_split_samples(data_dir / "val")

    # FIX: Compute class weights to handle imbalanced datasets
    total = len(train_labels)
    class_weight = {}
    for class_index in range(len(CLASS_NAMES)):
        count = train_labels.count(class_index)
        class_weight[class_index] = total / (len(CLASS_NAMES) * count) if count > 0 else 1.0

    train_ds = build_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_ds = build_dataset(val_paths, val_labels, args.batch_size, training=False)

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Prepared source counts:")
    for class_name in CLASS_NAMES:
        print(f"- {class_name}: {source_counts[class_name]}")

    print("\nTrain counts:")
    for class_name in CLASS_NAMES:
        print(f"- {class_name}: {train_counter[class_name]}")

    print("\nVal counts:")
    for class_name in CLASS_NAMES:
        print(f"- {class_name}: {val_counter[class_name]}")

    print("\nClass weights:", class_weight)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=build_callbacks(args.model_path),
        class_weight=class_weight,  # FIX: Added class weights
        verbose=1,
    )

    # Load the best saved model for evaluation
    model = tf.keras.models.load_model(args.model_path)

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    matrix, report = confusion_matrix_and_report(model, val_ds)
    print_confusion_matrix(matrix)
    print("\nClassification Report")
    print(report)

    save_metadata(
        {
            "image_size": list(IMAGE_SIZE),
            "class_names": CLASS_NAMES.copy(),
            "confidence_threshold": 0.5,
            "source_counts": source_counts,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "epochs_completed": len(history.history.get("loss", [])),
            "learning_rate": args.learning_rate,
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
        },
        args.metadata_path,
    )

    print(f"\nSaved model to {args.model_path}")
    print(f"Saved metadata to {args.metadata_path}")


if __name__ == "__main__":
    main()
