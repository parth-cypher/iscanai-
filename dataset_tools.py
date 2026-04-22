import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CANONICAL_CLASSES = {
    "normal": "1_normal",
    "cataract": "2_cataract",
    "glaucoma": "2_glaucoma",
    "retina": "3_retina_disease",
    "retina_disease": "3_retina_disease",
}


@dataclass
class ImageQualityResult:
    keep: bool
    reason: str
    blur_score: float
    mean_brightness: float
    eye_score: float


def canonical_class_dir(name: str) -> str | None:
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in CANONICAL_CLASSES:
        return CANONICAL_CLASSES[normalized]

    for key, value in CANONICAL_CLASSES.items():
        if key in normalized:
            return value
    return None


def ensure_extra_data_dirs(base_dir: str | Path = "data_extra") -> None:
    base_path = Path(base_dir)
    for folder in ("normal", "cataract", "glaucoma", "retina"):
        (base_path / folder).mkdir(parents=True, exist_ok=True)


def safe_open_rgb(image_path: str | Path) -> Image.Image | None:
    try:
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image)
            rgb = rgb.convert("RGB")
            rgb.load()
            return rgb.copy()
    except Exception:
        return None


def image_file_hash(image_path: str | Path) -> str:
    sha1 = hashlib.sha1()
    with open(image_path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def laplacian_variance(image: Image.Image) -> float:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    padded = np.pad(grayscale, ((1, 1), (1, 1)), mode="edge")
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    laplacian = (
        kernel[0, 0] * padded[:-2, :-2]
        + kernel[0, 1] * padded[:-2, 1:-1]
        + kernel[0, 2] * padded[:-2, 2:]
        + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 0] * padded[2:, :-2]
        + kernel[2, 1] * padded[2:, 1:-1]
        + kernel[2, 2] * padded[2:, 2:]
    )
    return float(np.var(laplacian))


def is_blurry(image: Image.Image, threshold: float = 8.0) -> bool:
    return laplacian_variance(image) < threshold


def mean_brightness(image: Image.Image) -> float:
    return float(np.asarray(image.convert("L"), dtype=np.float32).mean())


def is_bad_exposure(image: Image.Image, min_brightness: float = 20.0, max_brightness: float = 235.0) -> str | None:
    brightness = mean_brightness(image)
    if brightness < min_brightness:
        return "too_dark"
    if brightness > max_brightness:
        return "too_bright"
    return None


def estimate_eye_region_score(image: Image.Image) -> float:
    resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    grayscale = array.mean(axis=2)

    yy, xx = np.ogrid[:224, :224]
    cy, cx = 111.5, 111.5
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    center_disk = distance <= 80.0
    middle_ring = (distance > 80.0) & (distance <= 105.0)
    outer_ring = distance > 105.0

    center_mean = float(grayscale[center_disk].mean())
    middle_mean = float(grayscale[middle_ring].mean())
    outer_mean = float(grayscale[outer_ring].mean())
    contrast = center_mean - outer_mean

    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    warm_ratio = float((red > green * 0.92).mean())
    ring_darkness = float(1.0 - outer_mean)

    return float(0.45 * max(contrast, 0.0) + 0.30 * ring_darkness + 0.25 * warm_ratio + 0.10 * middle_mean)


def circular_iris_score(image: Image.Image) -> float:
    resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    grayscale = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0

    yy, xx = np.ogrid[:224, :224]
    cy, cx = 111.5, 111.5
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    pupil_disk = distance <= 26.0
    iris_ring = (distance > 26.0) & (distance <= 58.0)
    sclera_ring = (distance > 58.0) & (distance <= 92.0)
    outer_region = distance > 92.0

    pupil_mean = float(grayscale[pupil_disk].mean())
    iris_mean = float(grayscale[iris_ring].mean())
    sclera_mean = float(grayscale[sclera_ring].mean())
    outer_mean = float(grayscale[outer_region].mean())

    pupil_darkness = max(1.0 - pupil_mean, 0.0)
    iris_contrast = max(sclera_mean - iris_mean, 0.0)
    center_contrast = max(iris_mean - pupil_mean, 0.0)
    circular_separation = max(sclera_mean - outer_mean, 0.0)

    return float(
        0.32 * pupil_darkness
        + 0.28 * min(center_contrast / 0.30, 1.0)
        + 0.22 * min(iris_contrast / 0.25, 1.0)
        + 0.18 * min(circular_separation / 0.20, 1.0)
    )


def has_circular_iris_like_region(image: Image.Image, threshold: float = 0.26) -> bool:
    return circular_iris_score(image) >= threshold


def evaluate_image_quality(
    image: Image.Image,
    blur_threshold: float = 8.0,
    min_brightness: float = 20.0,
    max_brightness: float = 235.0,
    eye_threshold: float = 0.40,
    iris_threshold: float = 0.26,
) -> ImageQualityResult:
    blur_value = laplacian_variance(image)
    brightness = mean_brightness(image)
    generic_eye_score = estimate_eye_region_score(image)
    iris_score = circular_iris_score(image)
    eye_score = max(generic_eye_score, iris_score)

    if is_blurry(image, threshold=blur_threshold):
        return ImageQualityResult(False, "blurry", blur_value, brightness, eye_score)
    exposure_reason = is_bad_exposure(image, min_brightness=min_brightness, max_brightness=max_brightness)
    if exposure_reason is not None:
        return ImageQualityResult(False, exposure_reason, blur_value, brightness, eye_score)
    if generic_eye_score < eye_threshold and iris_score < iris_threshold:
        return ImageQualityResult(False, "non_eye_like", blur_value, brightness, eye_score)
    return ImageQualityResult(True, "ok", blur_value, brightness, eye_score)


def _clip_histogram(histogram: np.ndarray, clip_limit: int) -> np.ndarray:
    clipped = np.minimum(histogram, clip_limit)
    excess = int(histogram.sum() - clipped.sum())
    if excess > 0:
        clipped += excess // len(histogram)
        remainder = excess % len(histogram)
        if remainder > 0:
            clipped[:remainder] += 1
    return clipped


def clahe_enhance(image: Image.Image, tile_grid_size: int = 8, clip_limit: float = 2.0) -> Image.Image:
    ycbcr = image.convert("YCbCr")
    y_channel, cb_channel, cr_channel = ycbcr.split()
    y = np.asarray(y_channel, dtype=np.uint8)
    height, width = y.shape

    tiles_y = max(1, tile_grid_size)
    tiles_x = max(1, tile_grid_size)
    tile_height = math.ceil(height / tiles_y)
    tile_width = math.ceil(width / tiles_x)
    enhanced = np.zeros_like(y)

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            y0 = tile_y * tile_height
            y1 = min((tile_y + 1) * tile_height, height)
            x0 = tile_x * tile_width
            x1 = min((tile_x + 1) * tile_width, width)
            tile = y[y0:y1, x0:x1]

            histogram = np.bincount(tile.ravel(), minlength=256).astype(np.int32)
            tile_clip_limit = max(1, int(clip_limit * tile.size / 256.0))
            clipped = _clip_histogram(histogram, tile_clip_limit)
            cdf = clipped.cumsum()
            cdf_min = int(cdf[cdf > 0][0]) if np.any(cdf > 0) else 0
            denominator = max(int(cdf[-1] - cdf_min), 1)
            lookup = np.round((cdf - cdf_min) * 255.0 / denominator).clip(0, 255).astype(np.uint8)
            enhanced[y0:y1, x0:x1] = lookup[tile]

    enhanced_image = Image.fromarray(enhanced, mode="L")
    return Image.merge("YCbCr", (enhanced_image, cb_channel, cr_channel)).convert("RGB")


def sharpen_image(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))


def load_enhanced_rgb(image_path: str | Path) -> Image.Image | None:
    image = safe_open_rgb(image_path)
    if image is None:
        return None
    image = clahe_enhance(image)
    image = sharpen_image(image)
    return image


def split_samples(
    samples: list[tuple[Path, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[Path, str]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[Path, str]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    split_map = {"train": [], "val": [], "test": []}
    for class_name, class_samples in grouped.items():
        shuffled = class_samples[:]
        rng.shuffle(shuffled)

        total = len(shuffled)
        if total == 1:
            split_map["train"].extend(shuffled)
            continue

        desired_train = max(1, int(round(total * train_ratio)))
        desired_val = max(1, int(round(total * val_ratio)))
        desired_test = max(0, int(round(total * test_ratio)))

        while desired_train + desired_val + desired_test > total:
            if desired_test > 0:
                desired_test -= 1
            elif desired_val > 1:
                desired_val -= 1
            else:
                desired_train = max(1, desired_train - 1)

        if desired_train + desired_val > total:
            desired_val = max(1, total - desired_train)
        if desired_train + desired_val + desired_test < total:
            desired_train += total - (desired_train + desired_val + desired_test)

        train_end = desired_train
        val_end = desired_train + desired_val

        split_map["train"].extend(shuffled[:train_end])
        split_map["val"].extend(shuffled[train_end:val_end] or shuffled[-1:])
        if desired_test > 0:
            split_map["test"].extend(shuffled[val_end:val_end + desired_test] or shuffled[-1:])

    return split_map
