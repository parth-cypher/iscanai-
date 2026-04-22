import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    import cv2
except ImportError:
    cv2 = None

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Cataract"]
MOBILE_CLASS_NAMES = CLASS_NAMES.copy()
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_PATH = "cataract_external_model.h5"
MOBILE_MODEL_PATH = MODEL_PATH
RETINA_MODEL_PATH = MODEL_PATH
METADATA_PATH = "cataract_external_model_metadata.json"
DEFAULT_IMAGE_CANDIDATES = ["eye.jpg", "cleaneye.jpg", "eye1.jpg"]

MODEL_CACHE: dict[str, tf.keras.Model] = {}


def save_metadata(metadata: dict[str, Any], metadata_path: str | Path = METADATA_PATH) -> Path:
    output_path = Path(metadata_path)
    output_path.write_text(json.dumps(metadata, indent=2))
    return output_path


def clean_class_names(class_names: list[str]) -> list[str]:
    return [str(name).strip().title() for name in class_names]


def load_metadata(metadata_path: str | Path = METADATA_PATH) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        return {
            "image_size": list(IMAGE_SIZE),
            "class_names": CLASS_NAMES.copy(),
            "confidence_threshold": 0.5,  # FIX: lowered from 0.6
        }

    metadata = json.loads(path.read_text())
    return {
        "image_size": metadata.get("image_size", list(IMAGE_SIZE)),
        "class_names": metadata.get("class_names", CLASS_NAMES.copy()),
        "display_class_names": metadata.get("class_names", CLASS_NAMES.copy()),
        "confidence_threshold": metadata.get("confidence_threshold", 0.5),  # FIX: lowered from 0.6
        "train_counts": metadata.get("train_counts", {}),
        "val_counts": metadata.get("val_counts", {}),
    }


def resolve_default_image_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Image not found: {path}")

    for candidate in DEFAULT_IMAGE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError("No input image found. Provide an image path.")


def resolve_model_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Model not found: {path}")

    candidates = [MODEL_PATH, "mobile_model.h5", "cataract_model_v3_smoketest.h5", "cataract_model_v2.h5", "cataract_model.h5"]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(f"Model not found. Tried: {', '.join(candidates)}")


def load_model_safe(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


def get_model(explicit_model_path: str | Path | None = None) -> tf.keras.Model:
    resolved_path = resolve_model_path(explicit_model_path)
    cache_key = str(resolved_path.resolve())
    if cache_key not in MODEL_CACHE:
        MODEL_CACHE[cache_key] = load_model_safe(resolved_path)
    return MODEL_CACHE[cache_key]


def open_rgb_image(image_source: str | Path | Image.Image) -> Image.Image:
    if isinstance(image_source, Image.Image):
        image = image_source.copy()
    else:
        with Image.open(image_source) as opened:
            image = opened.copy()
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def prepare_upload_image(source_path: str | Path, destination_path: str | Path, max_side: int = 1600) -> Path:
    source = Path(source_path)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open_rgb_image(source) as image:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=92, optimize=True)

    return destination


def blur_score(image_source: str | Path | Image.Image) -> float:
    with open_rgb_image(image_source) as image:
        grayscale = np.asarray(image.convert("L"), dtype=np.float32)

    grad_x = np.diff(grayscale, axis=1)
    grad_y = np.diff(grayscale, axis=0)
    return float(np.var(grad_x) + np.var(grad_y))


def is_blurry_image(image_source: str | Path | Image.Image, threshold: float = 18.0) -> bool:
    return blur_score(image_source) < threshold


def external_eye_score(image_source: str | Path | Image.Image) -> float:
    with open_rgb_image(image_source) as image:
        resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0

    grayscale = array.mean(axis=2)
    yy, xx = np.ogrid[: IMAGE_SIZE[0], : IMAGE_SIZE[1]]
    cy, cx = (IMAGE_SIZE[0] - 1) / 2.0, (IMAGE_SIZE[1] - 1) / 2.0
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    center = distance <= 58.0
    middle = (distance > 58.0) & (distance <= 92.0)
    outer = distance > 92.0

    center_brightness = float(grayscale[center].mean())
    middle_brightness = float(grayscale[middle].mean())
    outer_brightness = float(grayscale[outer].mean())
    contrast = max(middle_brightness - center_brightness, 0.0)

    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    skin_like = float(
        ((red > 0.25) & (green > 0.16) & (blue > 0.12) & (red > green) & (green >= blue * 0.8)).mean()
    )
    sclera_like = float(
        ((grayscale > 0.52) & (np.abs(red - green) < 0.15) & (np.abs(green - blue) < 0.18)).mean()
    )
    dark_center = float((grayscale[center] < 0.22).mean())
    frame_balance = max(1.0 - abs(outer_brightness - middle_brightness), 0.0)

    score = (
        0.24 * contrast
        + 0.22 * skin_like
        + 0.18 * sclera_like
        + 0.18 * min(dark_center / 0.30, 1.0)
        + 0.18 * frame_balance
    )
    return float(score)


def center_cloudiness_score(image_source: str | Path | Image.Image) -> float:
    with open_rgb_image(image_source) as image:
        resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0

    grayscale = array.mean(axis=2)
    yy, xx = np.ogrid[: IMAGE_SIZE[0], : IMAGE_SIZE[1]]
    cy, cx = (IMAGE_SIZE[0] - 1) / 2.0, (IMAGE_SIZE[1] - 1) / 2.0
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    pupil_region = distance <= 34.0
    iris_region = (distance > 34.0) & (distance <= 64.0)

    pupil_brightness = float(grayscale[pupil_region].mean())
    iris_brightness = float(grayscale[iris_region].mean())
    white_ratio = float((grayscale[pupil_region] > 0.62).mean())
    low_texture = 1.0 - min(float(np.std(grayscale[pupil_region])) / 0.18, 1.0)

    return float(
        0.45 * max(pupil_brightness - iris_brightness + 0.20, 0.0)
        + 0.35 * white_ratio
        + 0.20 * low_texture
    )


def center_dark_pupil_score(image_source: str | Path | Image.Image) -> float:
    with open_rgb_image(image_source) as image:
        resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        grayscale = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0

    yy, xx = np.ogrid[: IMAGE_SIZE[0], : IMAGE_SIZE[1]]
    cy, cx = (IMAGE_SIZE[0] - 1) / 2.0, (IMAGE_SIZE[1] - 1) / 2.0
    distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    pupil_region = distance <= 34.0
    return float((grayscale[pupil_region] < 0.24).mean())


# FIX: Lowered threshold from 0.18 to 0.05 — consistent with training filter
def is_probable_external_eye(image_source: str | Path | Image.Image, threshold: float = 0.05) -> bool:
    return external_eye_score(image_source) >= threshold


def detect_image_type(image_source: str | Path | Image.Image) -> str:
    return "external" if is_probable_external_eye(image_source) else "unknown"


def load_and_preprocess_image(
    image_source: str | Path | Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
    add_batch: bool = False,
) -> np.ndarray:
    with open_rgb_image(image_source) as image:
        image = image.resize(image_size, Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32)

    processed = preprocess_input(array)
    if add_batch:
        processed = np.expand_dims(processed, axis=0)
    return processed


def top_predictions(probabilities: np.ndarray, class_names: list[str], limit: int = 2) -> list[dict[str, Any]]:
    scores = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    ranked_indices = np.argsort(scores)[::-1][:limit]
    return [
        {
            "class_name": class_names[int(index)],
            "confidence": round(float(scores[int(index)] * 100.0), 2),
        }
        for index in ranked_indices
    ]


def predict_external_cataract(
    image_path: str | Path,
    model_path: str | Path | None = None,
    metadata_path: str | Path = METADATA_PATH,
) -> dict[str, Any]:
    resolved_image_path = resolve_default_image_path(image_path)
    metadata = load_metadata(metadata_path)
    model = get_model(model_path)

    blurry = is_blurry_image(resolved_image_path)
    eye_score = external_eye_score(resolved_image_path)
    cloudiness = center_cloudiness_score(resolved_image_path)

    # Run AI model prediction
    image_batch = load_and_preprocess_image(resolved_image_path, add_batch=True)
    pred = float(model.predict(image_batch, verbose=0)[0][0])

    # pred = probability of being Cataract (index 1)
    cataract_prob = pred
    normal_prob = 1.0 - pred

    if cataract_prob >= 0.5:
        ml_label = "Cataract"
        ml_confidence = cataract_prob
    else:
        ml_label = "Normal"
        ml_confidence = normal_prob

    # FIX: REMOVED the visual rule override entirely.
    # The old detect_cataract_visual() just checked if center brightness > 135,
    # which is NOT a cataract indicator — it flagged any bright or overexposed image
    # as cataract and then OVERRODE the AI model. This was the main source of wrong
    # predictions. The AI model's output is now the single source of truth.
    final_label = ml_label
    confidence = ml_confidence

    # FIX: Only show "Uncertain" for genuinely bad images (blurry), not based on
    # eye_score which was rejecting valid images. Lowered eye_score threshold to 0.05.
    confidence_threshold = float(metadata.get("confidence_threshold", 0.5))
    warning = None

    if blurry:
        final_label = "Uncertain - Retake Image"
        warning = "Image looks blurry. Use a sharper close-up of the eye."
    elif eye_score < 0.05:
        # Only warn, don't override the prediction — the model may still be right
        warning = "Image may not be a clear eye photo. Results may be less reliable."
    elif confidence < confidence_threshold:
        warning = "Prediction confidence is low. Retake the image in better lighting."

    return {
        "image_path": str(resolved_image_path),
        "prediction": final_label,
        "predicted_class": final_label,
        "ai_prediction": ml_label,
        "ml_prediction": ml_label,
        "ml_confidence": round(ml_confidence * 100.0, 2),
        # FIX: visual_prediction now just reports cloudiness-based observation,
        # not used for overriding the final label
        "visual_prediction": "Cloudy" if cloudiness > 0.3 else "Clear",
        "visual_label": "Cloudy" if cloudiness > 0.3 else "Clear",
        "final_result": final_label,
        "confidence": round(confidence * 100.0, 2),
        "confidence_threshold": round(confidence_threshold * 100.0, 2),
        "cataract_probability": round(cataract_prob * 100.0, 2),
        "normal_probability": round(normal_prob * 100.0, 2),
        "cloudiness_score": round(cloudiness, 4),
        "eye_score": round(eye_score, 4),
        "warning": warning,
    }