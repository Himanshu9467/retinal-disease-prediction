"""Build a patient-safe 7,000-image binary retinal research dataset.

Source:
    research_aug_dataset/{train,val,test}/{normal,disease_detected}/

Output:
    research_binary_7000/
        train/{normal,disease}/
        val/{normal,disease}/
        test/{normal,disease}/
        reports/

The at-risk class is completely excluded. Patients with both normal and disease
labels are removed to avoid contradictory binary supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from tqdm import tqdm

SEED = 42
CLASSES = ("normal", "disease")
SPLITS = ("train", "val", "test")
SOURCE_CLASS_MAP = {
    "normal": "normal",
    "disease": "disease",
    "disease_detected": "disease",
}
TARGETS = {
    "train": {"normal": 2800, "disease": 2800},
    "val": {"normal": 350, "disease": 350},
    "test": {"normal": 350, "disease": 350},
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    filename: str
    patient_id: str
    class_name: str
    source_split: str
    sha256: str
    width: int
    height: int
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float
    entropy: float
    retinal_coverage: float
    vessel_visibility: float
    passed: bool
    removal_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("research_aug_dataset"))
    parser.add_argument("--output", type=Path, default=Path("research_binary_7000"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quality-threshold", type=float, default=0.58)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-quality",
        action="store_true",
        help="Ignore cached quality_statistics.csv and rescore all images.",
    )
    return parser.parse_args()


def setup_logging(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "preparation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def safe_prepare_output(output_root: Path, overwrite: bool) -> Path:
    resolved = output_root.resolve()
    if resolved.exists() and overwrite:
        if resolved == resolved.anchor or len(resolved.parts) < 3:
            raise ValueError(f"Refusing to remove unsafe output path: {resolved}")
        shutil.rmtree(resolved)
    for split in SPLITS:
        for class_name in CLASSES:
            (resolved / split / class_name).mkdir(parents=True, exist_ok=True)
    (resolved / "reports").mkdir(parents=True, exist_ok=True)
    return resolved


def extract_patient_id(filename: str) -> str:
    stem = Path(filename).stem.lower()

    # processed_dataset_600_left-600.jpg
    match = re.search(r'processed_dataset_(\d+)', stem)
    if match:
        return match.group(1)

    # prepared_dataset_123_right.jpg
    match = re.search(r'prepared_dataset_(\d+)', stem)
    if match:
        return match.group(1)

    stem = re.sub(r"^(prepared|processed)_dataset_", "", stem)
    stem = re.sub(r"^(normal|disease|disease_detected|at_risk)[_-]+", "", stem)

    hex_match = re.search(r"[a-f0-9]{10,64}", stem)
    if hex_match:
        return hex_match.group(0)[:16]

    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", stem)
        if token
        and token
        not in {
            "left",
            "right",
            "le",
            "re",
            "os",
            "od",
            "eye",
            "followup",
            "follow",
            "visit",
            "scan",
            "img",
            "image",
            "gf",
            "fa",
            "hbf",
            "hfa",
            "all",
        }
    ]

    if not tokens:
        return stem

    return tokens[0]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def score_image_quality(path: Path, class_name: str, source_split: str, threshold: float) -> ImageRecord:
    filename = path.name
    patient_id = extract_patient_id(filename)
    try:
        sha256 = sha256_file(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("opencv_read_failed")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        entropy = image_entropy(gray)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        retina_mask = (saturation > 28) & (value > 25) & (value < 248)
        retinal_coverage = float(retina_mask.mean())

        green = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        vessel_response = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
        vessel_visibility = float((vessel_response > np.percentile(vessel_response, 92)).mean())

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sharpness = float(np.sqrt(gx * gx + gy * gy).mean())

        bright_pixels = (gray > 185) & retina_mask
        optic_disc_visibility = float(bright_pixels.mean() / max(retinal_coverage, 1e-6))

        blur_score = normalize(laplacian_variance, 25.0, 180.0)
        contrast_score = normalize(contrast, 18.0, 58.0)
        brightness_score = 1.0 - min(abs(brightness - 118.0) / 95.0, 1.0)
        entropy_score = normalize(entropy, 3.2, 7.2)
        coverage_score = normalize(retinal_coverage, 0.20, 0.72)
        vessel_score = normalize(vessel_visibility, 0.010, 0.080)
        sharpness_score = normalize(sharpness, 7.0, 34.0)
        disc_score = normalize(optic_disc_visibility, 0.006, 0.060)

        quality_score = (
            0.18 * blur_score
            + 0.15 * contrast_score
            + 0.12 * brightness_score
            + 0.12 * entropy_score
            + 0.18 * coverage_score
            + 0.15 * vessel_score
            + 0.07 * sharpness_score
            + 0.03 * disc_score
        )

        reasons = []
        if laplacian_variance < 25:
            reasons.append("severe_blur")
        if brightness < 35:
            reasons.append("underexposure")
        if brightness > 225:
            reasons.append("overexposure")
        if contrast < 18:
            reasons.append("low_contrast")
        if entropy < 3.2:
            reasons.append("low_entropy")
        if retinal_coverage < 0.20:
            reasons.append("cropped_or_low_retinal_coverage")
        if retinal_coverage > 0.96:
            reasons.append("excessive_artifact_or_non_retinal_fill")
        if vessel_visibility < 0.010:
            reasons.append("unreadable_vessels")
        if optic_disc_visibility < 0.004:
            reasons.append("optic_disc_not_visible")
        if quality_score < threshold:
            reasons.append("quality_score_below_threshold")

        removal_reason = ";".join(reasons)
        return ImageRecord(
            path=path,
            filename=filename,
            patient_id=patient_id,
            class_name=class_name,
            source_split=source_split,
            sha256=sha256,
            width=width,
            height=height,
            quality_score=float(quality_score),
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            entropy=entropy,
            retinal_coverage=retinal_coverage,
            vessel_visibility=vessel_visibility,
            passed=not reasons,
            removal_reason=removal_reason,
        )
    except Exception as exc:
        return ImageRecord(
            path=path,
            filename=filename,
            patient_id=patient_id,
            class_name=class_name,
            source_split=source_split,
            sha256="",
            width=0,
            height=0,
            quality_score=0.0,
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            entropy=0.0,
            retinal_coverage=0.0,
            vessel_visibility=0.0,
            passed=False,
            removal_reason=f"processing_error:{exc}",
        )


def discover_images(source_root: Path) -> List[tuple[Path, str, str]]:
    discovered = []
    for split_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            class_name = SOURCE_CLASS_MAP.get(class_dir.name)
            if class_name not in CLASSES:
                continue
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    discovered.append((path, class_name, split_dir.name))
    return discovered


def records_from_cache(cache_path: Path) -> List[ImageRecord]:
    rows = pd.read_csv(cache_path)
    records = []
    for row in rows.to_dict("records"):
        records.append(
            ImageRecord(
                path=Path(row["path"]),
                filename=str(row["filename"]),
                patient_id=str(row["patient_id"]),
                class_name=str(row["class"]),
                source_split=str(row["source_split"]),
                sha256=str(row.get("sha256") or ""),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
                quality_score=float(row["quality_score"]),
                sharpness=float(row["sharpness"]),
                brightness=float(row["brightness"]),
                contrast=float(row["contrast"]),
                entropy=float(row["entropy"]),
                retinal_coverage=float(row["retinal_coverage"]),
                vessel_visibility=float(row["vessel_visibility"]),
                passed=bool(row["passed"]),
                removal_reason=str(row.get("removal_reason") or ""),
            )
        )
    return records


def process_quality(
    discovered: Sequence[tuple[Path, str, str]],
    report_dir: Path,
    threshold: float,
    workers: int,
    refresh: bool,
) -> List[ImageRecord]:
    cache_path = report_dir / "quality_statistics.csv"
    if cache_path.is_file() and not refresh:
        logging.info("Resuming from cached quality statistics: %s", cache_path)
        return records_from_cache(cache_path)

    records = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(score_image_quality, path, class_name, split, threshold)
            for path, class_name, split in discovered
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring image quality"):
            records.append(future.result())
    return records


def drop_duplicate_hashes(records: Sequence[ImageRecord]) -> tuple[List[ImageRecord], List[ImageRecord]]:
    best_by_hash: Dict[str, ImageRecord] = {}
    removed = []
    for record in records:
        if not record.sha256:
            removed.append(record)
            continue
        previous = best_by_hash.get(record.sha256)
        if previous is None or record.quality_score > previous.quality_score:
            if previous is not None:
                removed.append(
                    ImageRecord(**{**asdict(previous), "passed": False, "removal_reason": "duplicate_sha256"})
                )
            best_by_hash[record.sha256] = record
        else:
            removed.append(
                ImageRecord(**{**asdict(record), "passed": False, "removal_reason": "duplicate_sha256"})
            )
    return list(best_by_hash.values()), removed


def remove_mixed_label_patients(records: Sequence[ImageRecord]) -> tuple[List[ImageRecord], List[ImageRecord]]:
    labels_by_patient: Dict[str, set[str]] = defaultdict(set)
    for record in records:
        labels_by_patient[record.patient_id].add(record.class_name)

    mixed_patients = {
        patient_id for patient_id, labels in labels_by_patient.items() if len(labels) > 1
    }
    kept, removed = [], []
    for record in records:
        if record.patient_id in mixed_patients:
            removed.append(
                ImageRecord(
                    **{
                        **asdict(record),
                        "passed": False,
                        "removal_reason": "mixed_binary_patient_labels",
                    }
                )
            )
        else:
            kept.append(record)
    return kept, removed


def select_split_images(
    records: Sequence[ImageRecord], class_name: str, rng: random.Random
) -> Dict[str, List[ImageRecord]]:
    class_records = [record for record in records if record.class_name == class_name and record.passed]
    by_patient: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in class_records:
        by_patient[record.patient_id].append(record)
    for patient_records in by_patient.values():
        patient_records.sort(
            key=lambda item: (item.quality_score, item.retinal_coverage, item.sharpness, item.sha256),
            reverse=True,
        )

    patients = list(by_patient)
    rng.shuffle(patients)
    patients.sort(
        key=lambda patient_id: (
            np.mean([record.quality_score for record in by_patient[patient_id][:3]]),
            -len(by_patient[patient_id]),
            patient_id,
        ),
        reverse=True,
    )

    selected: Dict[str, List[ImageRecord]] = {split: [] for split in SPLITS}
    used_patients: set[str] = set()

    for split in ("test", "val", "train"):
        target = TARGETS[split][class_name]
        split_patients = []
        capacity = 0
        for patient_id in patients:
            if patient_id in used_patients:
                continue
            split_patients.append(patient_id)
            used_patients.add(patient_id)
            capacity += len(by_patient[patient_id])
            if capacity >= target:
                break
        if capacity < target:
            raise RuntimeError(
                f"Insufficient high-quality {class_name} images for {split}: "
                f"capacity={capacity}, target={target}"
            )
        split_records = [
            record for patient_id in split_patients for record in by_patient[patient_id]
        ]
        split_records.sort(
            key=lambda item: (item.quality_score, item.retinal_coverage, item.sharpness, item.sha256),
            reverse=True,
        )
        selected[split] = split_records[:target]

    return selected


def build_selection(records: Sequence[ImageRecord], seed: int) -> Dict[str, List[ImageRecord]]:
    rng = random.Random(seed)
    selected = {split: [] for split in SPLITS}
    for class_name in CLASSES:
        class_selection = select_split_images(records, class_name, rng)
        for split in SPLITS:
            selected[split].extend(class_selection[split])
    return selected


def destination_name(record: ImageRecord) -> str:
    suffix = record.path.suffix.lower() or ".jpg"
    return f"{record.patient_id}_{record.sha256[:12]}{suffix}"


def materialize_dataset(output_root: Path, selected: Dict[str, List[ImageRecord]]) -> List[dict]:
    manifest_rows = []
    for split in SPLITS:
        for record in tqdm(selected[split], desc=f"Copying {split}"):
            destination = output_root / split / record.class_name / destination_name(record)
            if not destination.exists():
                shutil.copy2(record.path, destination)
            manifest_rows.append(
                {
                    "split": split,
                    "class": record.class_name,
                    "patient_id": record.patient_id,
                    "filename": destination.name,
                    "image_path": str(destination.resolve()),
                    "source_path": str(record.path.resolve()),
                    "sha256": record.sha256,
                    "quality_score": record.quality_score,
                }
            )
    return manifest_rows


def write_reports(
    report_dir: Path,
    records: Sequence[ImageRecord],
    removed: Sequence[ImageRecord],
    manifest_rows: Sequence[dict],
) -> None:
    quality_rows = []
    for record in records:
        quality_rows.append(
            {
                "path": str(record.path),
                "filename": record.filename,
                "patient_id": record.patient_id,
                "class": record.class_name,
                "source_split": record.source_split,
                "sha256": record.sha256,
                "width": record.width,
                "height": record.height,
                "quality_score": record.quality_score,
                "sharpness": record.sharpness,
                "brightness": record.brightness,
                "contrast": record.contrast,
                "entropy": record.entropy,
                "retinal_coverage": record.retinal_coverage,
                "vessel_visibility": record.vessel_visibility,
                "passed": record.passed,
                "removal_reason": record.removal_reason,
            }
        )
    pd.DataFrame(quality_rows).to_csv(report_dir / "quality_statistics.csv", index=False)

    removed_rows = [
        {
            "filename": record.filename,
            "patient_id": record.patient_id,
            "class": record.class_name,
            "quality_score": record.quality_score,
            "removal_reason": record.removal_reason,
        }
        for record in removed
    ]
    pd.DataFrame(removed_rows).to_csv(report_dir / "removed_images.csv", index=False)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(report_dir / "dataset_manifest.csv", index=False)

    summary_rows = []
    patient_rows = []
    for split in SPLITS:
        split_rows = manifest[manifest["split"] == split]
        for class_name in CLASSES:
            rows = split_rows[split_rows["class"] == class_name]
            summary_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "images": int(len(rows)),
                    "patients": int(rows["patient_id"].nunique()),
                }
            )
            for patient_id, group in rows.groupby("patient_id"):
                patient_rows.append(
                    {
                        "patient_id": patient_id,
                        "split": split,
                        "class": class_name,
                        "image_count": int(len(group)),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(report_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(report_dir / "patient_distribution.csv", index=False)


def validate_selection(manifest_rows: Sequence[dict]) -> tuple[bool, bool]:
    manifest = pd.DataFrame(manifest_rows)
    patient_sets = {
        split: set(manifest.loc[manifest["split"] == split, "patient_id"]) for split in SPLITS
    }

    assert patient_sets["train"].isdisjoint(patient_sets["val"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["val"].isdisjoint(patient_sets["test"])

    for split in SPLITS:
        for class_name in CLASSES:
            observed = len(manifest[(manifest["split"] == split) & (manifest["class"] == class_name)])
            expected = TARGETS[split][class_name]
            if observed != expected:
                raise AssertionError(
                    f"{split}/{class_name}: expected {expected}, observed {observed}"
                )

    duplicate_sha = manifest["sha256"].duplicated().any()
    duplicate_paths = manifest["image_path"].duplicated().any()
    if duplicate_sha or duplicate_paths:
        raise AssertionError("Duplicate selected images detected")

    total = len(manifest)
    if total != 7000:
        raise AssertionError(f"Expected 7000 selected images, observed {total}")
    return True, True


def log_statistics(
    discovered_count: int,
    records: Sequence[ImageRecord],
    removed: Sequence[ImageRecord],
    manifest_rows: Sequence[dict],
    threshold: float,
) -> None:
    manifest = pd.DataFrame(manifest_rows)
    logging.info("Total images scanned: %s", discovered_count)
    logging.info("Total patients scanned: %s", len({record.patient_id for record in records}))
    logging.info("Images removed: %s", len(removed))
    logging.info("Patients removed: %s", len({record.patient_id for record in removed}))
    logging.info("Quality threshold: %.3f", threshold)
    logging.info("Raw class distribution: %s", dict(Counter(record.class_name for record in records)))
    logging.info(
        "Final class distribution: %s",
        dict(Counter(manifest["class"])) if not manifest.empty else {},
    )
    for split in SPLITS:
        split_rows = manifest[manifest["split"] == split]
        logging.info(
            "%s final statistics: images=%s patients=%s classes=%s",
            split,
            len(split_rows),
            split_rows["patient_id"].nunique(),
            dict(Counter(split_rows["class"])),
        )
    logging.info("Verification results: patient leakage passed; duplicate check passed")


def print_final_summary(manifest_rows: Sequence[dict]) -> None:
    manifest = pd.DataFrame(manifest_rows)

    def line_for(split: str, class_name: str) -> int:
        return int(len(manifest[(manifest["split"] == split) & (manifest["class"] == class_name)]))

    def patients_for(split: str) -> int:
        return int(manifest.loc[manifest["split"] == split, "patient_id"].nunique())

    print("\n==================================================")
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("==========================================")
    print("\nTrain:")
    print(f"Normal: {line_for('train', 'normal')}")
    print(f"Disease: {line_for('train', 'disease')}")
    print(f"Patients: {patients_for('train')}")
    print("\nValidation:")
    print(f"Normal: {line_for('val', 'normal')}")
    print(f"Disease: {line_for('val', 'disease')}")
    print(f"Patients: {patients_for('val')}")
    print("\nTest:")
    print(f"Normal: {line_for('test', 'normal')}")
    print(f"Disease: {line_for('test', 'disease')}")
    print(f"Patients: {patients_for('test')}")
    print("\nPatient Leakage Check: PASSED")
    print("Duplicate Check: PASSED")
    print("Quality Filtering: PASSED")
    print("=========================")


def main() -> None:
    args = parse_args()
    check_random_state(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_root = args.source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    output_root = safe_prepare_output(args.output, args.overwrite)
    report_dir = output_root / "reports"
    setup_logging(report_dir)

    logging.info("Starting binary retinal dataset preparation")
    logging.info("Source: %s", source_root)
    logging.info("Output: %s", output_root)

    discovered = discover_images(source_root)
    if not discovered:
        raise RuntimeError(f"No binary source images found under {source_root}")

    records = process_quality(
        discovered,
        report_dir,
        args.quality_threshold,
        args.workers,
        args.refresh_quality,
    )
    unique_records, duplicate_removed = drop_duplicate_hashes(records)
    passed_records = [record for record in unique_records if record.passed]
    non_quality_removed = [record for record in unique_records if not record.passed]
    single_label_records, mixed_removed = remove_mixed_label_patients(passed_records)
    removed = [*duplicate_removed, *non_quality_removed, *mixed_removed]

    logging.info("High-quality single-label candidate images: %s", len(single_label_records))
    selection = build_selection(single_label_records, args.seed)
    manifest_rows = materialize_dataset(output_root, selection)
    validate_selection(manifest_rows)
    write_reports(report_dir, records, removed, manifest_rows)
    log_statistics(
        len(discovered),
        records,
        removed,
        manifest_rows,
        args.quality_threshold,
    )
    print_final_summary(manifest_rows)
# -----------------------------
# TEST PATIENT ID EXTRACTION
# -----------------------------

tests = [
    "processed_dataset_600_left-600.jpg",
    "processed_dataset_600_right-600.jpg",
    "processed_dataset_601_left-601.jpg",
]

for t in tests:
    print(t, "->", extract_patient_id(t))


if __name__ == "__main__":
    main()
