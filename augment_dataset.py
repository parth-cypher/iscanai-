import argparse
import os
import random
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from model_utils import VALID_IMAGE_EXTENSIONS

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download optional mobile-eye datasets and physically augment them into output/train_augmented."
    )
    parser.add_argument("--source-dir", default="datasets/mobile_raw", help="Raw mobile image directory.")
    parser.add_argument("--prepared-dir", default="datasets/mobile_binary", help="Prepared Normal/Cataract dataset directory.")
    parser.add_argument("--output-dir", default="output/train_augmented", help="Destination for augmented class folders.")
    parser.add_argument("--target-total", type=int, default=2000, help="Target total image count after augmentation.")
    parser.add_argument(
        "--kaggle-dataset",
        action="append",
        default=[],
        help="Kaggle dataset handle like username/dataset-name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--roboflow-dataset",
        action="append",
        default=[],
        help="Roboflow dataset reference in workspace/project/version format. Can be passed multiple times.",
    )
    parser.add_argument("--roboflow-api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key.")
    parser.add_argument("--download-only", action="store_true", help="Only download and prepare the dataset, without augmentation.")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def download_from_kaggle(dataset_handle: str, download_root: Path) -> Path:
    destination = download_root / dataset_handle.replace("/", "_")
    destination.mkdir(parents=True, exist_ok=True)

    kaggle_path = shutil.which("kaggle")
    if kaggle_path:
        run_command([kaggle_path, "datasets", "download", "-d", dataset_handle, "-p", str(destination), "--unzip"])
        return destination

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle download requested, but neither the kaggle CLI nor kagglehub is installed."
        ) from exc

    downloaded_path = Path(kagglehub.dataset_download(dataset_handle))
    if downloaded_path.is_dir():
        for item in downloaded_path.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    else:
        shutil.copy2(downloaded_path, destination / downloaded_path.name)

    return destination


def download_from_roboflow(dataset_ref: str, download_root: Path, api_key: str | None) -> Path:
    if not api_key:
        raise RuntimeError("Roboflow download requested, but ROBOFLOW_API_KEY is missing.")

    destination = download_root / dataset_ref.replace("/", "_")
    destination.mkdir(parents=True, exist_ok=True)

    workspace, project, version = dataset_ref.split("/")

    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Roboflow download requested, but the roboflow Python package is not installed.") from exc

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(int(version))
    version_obj.download("folder", location=str(destination))
    return destination


def find_all_images(root_dir: Path) -> list[Path]:
    return [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def infer_binary_label(path: Path) -> str | None:
    lowered = " ".join(part.lower() for part in path.parts)

    if "cataract" in lowered:
        return "Cataract"

    normal_keywords = ["normal", "healthy", "control", "clear", "non cataract", "no cataract"]
    if any(keyword in lowered for keyword in normal_keywords):
        return "Normal"

    return None


def prepare_binary_dataset(source_dir: Path, prepared_dir: Path) -> dict[str, int]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    counts = {"Normal": 0, "Cataract": 0}

    for class_name in counts:
        (prepared_dir / class_name).mkdir(parents=True, exist_ok=True)

    for image_path in find_all_images(source_dir):
        label = infer_binary_label(image_path)
        if label is None:
            continue

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            destination_name = f"{image_path.stem}_{uuid.uuid4().hex[:8]}.jpg"
            destination_path = prepared_dir / label / destination_name
            image.save(destination_path, format="JPEG", quality=92)
            counts[label] += 1

    return counts


def normalize_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image.copy()


def apply_zoom(image: Image.Image) -> Image.Image:
    width, height = image.size
    zoom_factor = random.uniform(0.88, 1.0)
    crop_width = int(width * zoom_factor)
    crop_height = int(height * zoom_factor)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def apply_noise(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=random.uniform(4.0, 10.0), size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def augment_one_image(image: Image.Image) -> Image.Image:
    augmented = image.copy()

    if random.random() < 0.7:
        angle = random.uniform(-15, 15)
        augmented = augmented.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    if random.random() < 0.8:
        augmented = apply_zoom(augmented)

    if random.random() < 0.75:
        brightness = ImageEnhance.Brightness(augmented)
        augmented = brightness.enhance(random.uniform(0.8, 1.2))

    if random.random() < 0.5:
        augmented = ImageOps.mirror(augmented)

    if random.random() < 0.7:
        augmented = apply_noise(augmented)

    if random.random() < 0.6:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

    return augmented


def copy_originals(source_dir: Path, output_dir: Path) -> dict[str, list[Path]]:
    copied_by_class: dict[str, list[Path]] = {"Normal": [], "Cataract": []}

    for class_name in copied_by_class:
        class_source_dir = source_dir / class_name
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        if not class_source_dir.exists():
            continue

        for image_path in find_all_images(class_source_dir):
            image = normalize_image(image_path)
            destination = class_output_dir / f"{image_path.stem}_{uuid.uuid4().hex[:8]}.jpg"
            image.save(destination, format="JPEG", quality=92)
            copied_by_class[class_name].append(destination)

    return copied_by_class


def augment_dataset(prepared_dir: Path, output_dir: Path, target_total: int) -> dict[str, int]:
    random.seed(SEED)
    np.random.seed(SEED)

    copied_by_class = copy_originals(prepared_dir, output_dir)
    totals = {class_name: len(paths) for class_name, paths in copied_by_class.items()}
    current_total = sum(totals.values())

    if current_total == 0:
        raise ValueError("No prepared mobile images were found to augment.")

    if current_total >= target_total:
        return totals

    classes = [class_name for class_name in copied_by_class if copied_by_class[class_name]]
    if not classes:
        raise ValueError("No class images were available after preparation.")

    per_class_target = max(target_total // len(classes), max(totals.values()))

    for class_name in classes:
        class_output_dir = output_dir / class_name
        source_files = copied_by_class[class_name]

        while totals[class_name] < per_class_target:
            base_file = random.choice(source_files)
            image = normalize_image(base_file)
            augmented = augment_one_image(image)
            destination = class_output_dir / f"{Path(base_file).stem}_aug_{uuid.uuid4().hex[:8]}.jpg"
            augmented.save(destination, format="JPEG", quality=90)
            source_files.append(destination)
            totals[class_name] += 1

    while sum(totals.values()) < target_total:
        class_name = min(classes, key=lambda name: totals[name])
        class_output_dir = output_dir / class_name
        base_file = random.choice(copied_by_class[class_name])
        image = normalize_image(base_file)
        augmented = augment_one_image(image)
        destination = class_output_dir / f"{Path(base_file).stem}_extra_{uuid.uuid4().hex[:8]}.jpg"
        augmented.save(destination, format="JPEG", quality=90)
        copied_by_class[class_name].append(destination)
        totals[class_name] += 1

    return totals


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    download_root = source_dir

    if args.kaggle_dataset or args.roboflow_dataset:
        download_root.mkdir(parents=True, exist_ok=True)

    for dataset_handle in args.kaggle_dataset:
        print(f"Downloading Kaggle dataset: {dataset_handle}")
        download_from_kaggle(dataset_handle, download_root)

    for dataset_ref in args.roboflow_dataset:
        print(f"Downloading Roboflow dataset: {dataset_ref}")
        download_from_roboflow(dataset_ref, download_root, args.roboflow_api_key)

    prepared_counts = prepare_binary_dataset(source_dir, prepared_dir)
    print(f"Prepared dataset counts: {prepared_counts}")

    if args.download_only:
        print(f"Prepared binary dataset written to: {prepared_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    totals = augment_dataset(prepared_dir, output_dir, args.target_total)
    print(f"Augmented dataset counts: {totals}")
    print(f"Final total images: {sum(totals.values())}")
    print(f"Augmented dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
