"""
Patient-level safe retinal dataset splitter and leakage auditor.

This script fixes image-level split leakage by rebuilding train/val/test folders
from patient/image-family groups. It supports both common dataset layouts:

1. Flat class folders:
   dataset/
     normal/
     at_risk/
     disease_detected/

2. Existing split folders, which are collapsed before regrouping:
   dataset/
     train/normal/
     val/normal/
     test/normal/

Outputs:
  research_patient_safe_dataset/
    train|val|test/class_label/*.jpg
    dataset_manifest.csv
    patient_split_summary.csv
    leakage_audit.csv

The implementation is intentionally strict:
  - all images with the same patient/family key stay in the same split
  - duplicate SHA256 hashes may not cross splits
  - perceptual hashes are used to flag near-duplicate leakage
  - corrupted images are rejected from the rebuilt dataset
  - optional retina-quality heuristics flag suspicious non-retinal images
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageStat, UnidentifiedImageError

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional runtime feature
    cv2 = None
    np = None


CLASS_NAMES = ("normal", "at_risk", "disease_detected")
SPLIT_NAMES = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    filename: str
    class_label: str
    patient_id: str
    family_id: str
    eye: str
    image_hash: str
    perceptual_hash: str
    width: int
    height: int
    source_split: str
    retina_quality: str
    retina_score: float


@dataclass
class PatientGroup:
    group_id: str
    records: List[ImageRecord] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def class_counts(self) -> Counter:
        return Counter(record.class_label for record in self.records)

    @property
    def dominant_class(self) -> str:
        counts = self.class_counts
        return max(CLASS_NAMES, key=lambda cls: (counts[cls], cls))


def normalize_label(value: str) -> Optional[str]:
    """Return a canonical class label if a folder name represents a class."""
    label = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "normal": "normal",
        "healthy": "normal",
        "no_disease": "normal",
        "at_risk": "at_risk",
        "atrisk": "at_risk",
        "risk": "at_risk",
        "disease_detected": "disease_detected",
        "disease": "disease_detected",
        "detected": "disease_detected",
    }
    return aliases.get(label)


def normalize_split(value: str) -> Optional[str]:
    split = value.strip().lower()
    aliases = {"training": "train", "validation": "val", "valid": "val", "testing": "test"}
    split = aliases.get(split, split)
    return split if split in SPLIT_NAMES else None


def iter_image_candidates(input_root: Path) -> Iterable[Tuple[Path, str, str]]:
    """Yield image path, class label, and existing split if present."""
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel_parts = path.relative_to(input_root).parts
        class_label = None
        source_split = "unsplit"

        # Support dataset/split/class/image.ext.
        if len(rel_parts) >= 3 and normalize_split(rel_parts[0]) and normalize_label(rel_parts[1]):
            source_split = normalize_split(rel_parts[0]) or "unsplit"
            class_label = normalize_label(rel_parts[1])

        # Support dataset/class/image.ext.
        if class_label is None and len(rel_parts) >= 2 and normalize_label(rel_parts[0]):
            class_label = normalize_label(rel_parts[0])

        # Support accidental nested layouts by searching parents nearest first.
        if class_label is None:
            for parent in reversed(path.relative_to(input_root).parents):
                if not parent.parts:
                    continue
                maybe_label = normalize_label(parent.parts[-1])
                if maybe_label:
                    class_label = maybe_label
                    break

        if class_label:
            yield path, class_label, source_split


def strip_known_prefixes(stem: str) -> str:
    cleaned = stem.lower()
    cleaned = re.sub(
        r"^(prepared_dataset_|processed_dataset_|processed_|image_|img_|retina_|fundus_)+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"^aug_\d+_", "", cleaned)
    cleaned = re.sub(r"_aug_\d+$", "", cleaned)
    return cleaned


def extract_eye(stem: str) -> str:
    """Extract eye laterality when encoded in a filename."""
    lowered = stem.lower()
    eye_patterns = [
        (r"(?:^|[_\-.])left(?:$|[_\-.])", "left"),
        (r"(?:^|[_\-.])right(?:$|[_\-.])", "right"),
        (r"(?:^|[_\-.])os(?:$|[_\-.])", "left"),
        (r"(?:^|[_\-.])od(?:$|[_\-.])", "right"),
        (r"(?:^|[_\-.])l(?:$|[_\-.])", "left"),
        (r"(?:^|[_\-.])r(?:$|[_\-.])", "right"),
    ]
    for pattern, eye in eye_patterns:
        if re.search(pattern, lowered):
            return eye
    return "unknown"


def extract_patient_and_family(path: Path) -> Tuple[str, str, str]:
    """
    Robust patient/family extraction for retinal datasets.

    Goals:
    - keep ALL augmented siblings together
    - keep left/right eyes together
    - remove processing suffixes
    - prevent leakage across train/val/test
    """

    raw_stem = strip_known_prefixes(path.stem)
    raw_stem = raw_stem.lower()

    eye = extract_eye(raw_stem)

    # ------------------------------------------------------------------
    # STEP 1: normalize separators
    # ------------------------------------------------------------------
    cleaned = re.sub(r"[\-\.]+", "_", raw_stem)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    # ------------------------------------------------------------------
    # STEP 2: remove augmentation suffixes
    # Example:
    # prepared_dataset_15056_left_aug_612
    # -> 15056_left
    # ------------------------------------------------------------------
    cleaned = re.sub(
        r"_aug_\d+",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    # ------------------------------------------------------------------
    # STEP 3: remove processing suffixes
    #
    # Handles:
    # -600
    # -FA
    # -HBF
    # -GF
    # -ALL
    # chained forms:
    # -600-FA-HBF
    # _600_hfa_all
    # etc.
    # ------------------------------------------------------------------
    cleaned = re.sub(
        r"([_-](600|gf|fa|rb|hbf|hfa|hff|hb|fs|all))+",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    # ------------------------------------------------------------------
    # STEP 4: remove laterality tokens ONLY for patient ID generation
    # Keep them for family grouping initially.
    # ------------------------------------------------------------------
    patient_base = re.sub(
        r"(^|_)(left|right|od|os|l|r)($|_)",
        "_",
        cleaned,
        flags=re.IGNORECASE
    )

    patient_base = re.sub(r"_+", "_", patient_base).strip("_")

    # ------------------------------------------------------------------
    # STEP 5: explicit patient IDs
    # ------------------------------------------------------------------
    explicit = re.search(
        r"(?:patient|pt|pid|subject|subj|id)[_\- ]*([a-z0-9]+)",
        patient_base
    )
    explicit_patient_id = False

    if explicit:
        patient_id = explicit.group(1)
        explicit_patient_id = True

    else:
        tokens = [t for t in patient_base.split("_") if t]

        # anonymized/hash IDs
        mixed_tokens = [
            t for t in tokens
            if len(t) >= 8
            and re.search(r"[a-z]", t)
            and re.search(r"\d", t)
        ]

        if mixed_tokens:
            patient_id = mixed_tokens[0]

        else:
            numeric_tokens = re.findall(r"\d{2,}", patient_base)

            if numeric_tokens:
                patient_id = numeric_tokens[0]

            else:
                patient_id = patient_base

    # ------------------------------------------------------------------
    # FINAL SAFETY FALLBACK
    # ------------------------------------------------------------------
    if not patient_id:
        patient_id = hashlib.sha1(
            str(path.stem).encode("utf-8")
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # FAMILY ID
    #
    # IMPORTANT:
    # family_id keeps eye info removed too,
    # preventing left/right leakage.
    # ------------------------------------------------------------------
    family = re.sub(
        r"(^|_)(left|right|od|os|l|r)($|_)",
        "_",
        cleaned,
        flags=re.IGNORECASE
    )

    family = re.sub(r"_+", "_", family).strip("_")

    if explicit_patient_id:
        family = patient_id

    if not family:
        family = patient_id

    return patient_id, family, eye


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Small dependency-free perceptual hash using dHash."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append(1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def retina_quality_score(image: Image.Image) -> Tuple[str, float]:
    """
    Lightweight retinal heuristic.

    This does not replace a trained quality model. It flags likely non-retinal or
    unusable images by checking whether a large, roughly circular bright fundus
    region exists against a darker background.
    """
    if cv2 is None or np is None:
        stat = ImageStat.Stat(image.convert("L").resize((64, 64)))
        mean = float(stat.mean[0])
        score = max(0.0, min(1.0, mean / 128.0))
        return ("unchecked", score)

    rgb = np.array(image.convert("RGB").resize((256, 256)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "suspicious_non_retinal", 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    image_area = float(gray.shape[0] * gray.shape[1])
    area_ratio = area / max(image_area, 1.0)
    perimeter = float(cv2.arcLength(largest, True))
    circularity = 0.0 if perimeter == 0 else (4.0 * math.pi * area) / (perimeter * perimeter)

    score = max(0.0, min(1.0, 0.65 * min(area_ratio / 0.45, 1.0) + 0.35 * min(circularity, 1.0)))
    if area_ratio < 0.08 or circularity < 0.20:
        return "suspicious_non_retinal", score
    if area_ratio < 0.15 or circularity < 0.35:
        return "low_quality_or_cropped", score
    return "retina_like", score


def scan_dataset(input_root: Path, reject_non_retinal: bool = False) -> Tuple[List[ImageRecord], List[Dict[str, str]]]:
    records: List[ImageRecord] = []
    rejected: List[Dict[str, str]] = []

    for index, (path, class_label, source_split) in enumerate(iter_image_candidates(input_root), start=1):
        if index % 1000 == 0:
            print(f"  scanned {index:,} images...")

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                width, height = image.size
                phash = difference_hash(image)
                quality, quality_score = retina_quality_score(image)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            rejected.append({"image_path": str(path), "reason": f"corrupted_or_unreadable: {exc}"})
            continue

        if reject_non_retinal and quality == "suspicious_non_retinal":
            rejected.append({"image_path": str(path), "reason": f"non_retinal_heuristic: {quality_score:.4f}"})
            continue

        patient_id, family_id, eye = extract_patient_and_family(path)
        records.append(
            ImageRecord(
                source_path=path,
                filename=path.name,
                class_label=class_label,
                patient_id=patient_id,
                family_id=family_id,
                eye=eye,
                image_hash=sha256_file(path),
                perceptual_hash=phash,
                width=width,
                height=height,
                source_split=source_split,
                retina_quality=quality,
                retina_score=quality_score,
            )
        )

    return records, rejected


def load_records_from_manifest(path: Path) -> List[ImageRecord]:
    """Reload previously scanned image metadata to make reruns/resumes fast."""
    records: List[ImageRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_path = Path(row.get("source_path") or row.get("image_path") or "")
            if not source_path.exists():
                continue
            patient_id, family_id, eye = extract_patient_and_family(source_path)
            records.append(
                ImageRecord(
                    source_path=source_path,
                    filename=row.get("source_filename") or source_path.name,
                    class_label=row["class_label"],
                    patient_id=patient_id,
                    family_id=family_id,
                    eye=eye,
                    image_hash=row["image_hash"],
                    perceptual_hash=row.get("perceptual_hash") or "",
                    width=int(row["width"]),
                    height=int(row["height"]),
                    source_split=row.get("source_split") or "cached",
                    retina_quality=row.get("retina_quality") or "cached",
                    retina_score=float(row.get("retina_score") or 0.0),
                )
            )
    return records


def build_groups(records: Sequence[ImageRecord]) -> List[PatientGroup]:
    groups: Dict[str, PatientGroup] = {}

    for record in records:
        # Use patient ID as the hard split key. Family is audited separately.
        key = f"patient::{record.patient_id}"
        groups.setdefault(key, PatientGroup(key)).records.append(record)

    return list(groups.values())


def greedy_patient_split(groups: Sequence[PatientGroup], ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    """
    Deterministic stratified greedy split at patient-group level.

    The splitter sorts harder groups first: larger and class-mixed groups are
    assigned before simpler groups. Each candidate split is scored by how close
    the resulting image and class distributions remain to target ratios.
    """
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)

    total_images = sum(group.size for group in groups)
    total_class_counts = Counter()
    for group in groups:
        total_class_counts.update(group.class_counts)

    target_images = {split: total_images * ratios[split] for split in SPLIT_NAMES}
    target_class_counts = {
        split: {cls: total_class_counts[cls] * ratios[split] for cls in CLASS_NAMES}
        for split in SPLIT_NAMES
    }

    assigned_images = Counter()
    assigned_classes: Dict[str, Counter] = {split: Counter() for split in SPLIT_NAMES}
    assignments: Dict[str, str] = {}

    def difficulty(group: PatientGroup) -> Tuple[int, int, int]:
        return (group.size, len(group.class_counts), group.class_counts[group.dominant_class])

    def global_score() -> float:
        """Score the whole current split state, with strong penalties for imbalance."""
        score = 0.0
        for split in SPLIT_NAMES:
            image_target = max(target_images[split], 1.0)
            image_error = (assigned_images[split] - target_images[split]) / image_target
            score += image_error * image_error * 2.0

            for cls in CLASS_NAMES:
                class_target = max(target_class_counts[split][cls], 1.0)
                class_error = (assigned_classes[split][cls] - target_class_counts[split][cls]) / class_target
                score += class_error * class_error

                # Large class overfill in val/test is especially harmful for
                # research evaluation because it distorts per-class metrics.
                if split != "train" and assigned_classes[split][cls] > target_class_counts[split][cls] * 1.10:
                    score += 10.0 * (
                        (assigned_classes[split][cls] - target_class_counts[split][cls])
                        / class_target
                    ) ** 2

            if split != "train" and assigned_images[split] > target_images[split] * 1.08:
                score += 10.0 * ((assigned_images[split] - target_images[split]) / image_target) ** 2
        return score

    for group in sorted(groups, key=difficulty, reverse=True):
        best_split = None
        best_score = float("inf")

        for split in SPLIT_NAMES:
            assigned_images[split] += group.size
            assigned_classes[split].update(group.class_counts)
            score = global_score()
            assigned_images[split] -= group.size
            for cls, count in group.class_counts.items():
                assigned_classes[split][cls] -= count

            if score < best_score:
                best_score = score
                best_split = split

        assert best_split is not None
        assignments[group.group_id] = best_split
        assigned_images[best_split] += group.size
        assigned_classes[best_split].update(group.class_counts)

    return assignments


def prepare_output_dirs(output_root: Path, resume: bool) -> None:
    if output_root.exists() and not resume:
        shutil.rmtree(output_root)
    for split in SPLIT_NAMES:
        for cls in CLASS_NAMES:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)


def unique_destination(output_root: Path, split: str, record: ImageRecord) -> Path:
    destination = output_root / split / record.class_label / record.filename
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    short_hash = record.image_hash[:12]
    return destination.with_name(f"{stem}_{short_hash}{suffix}")


def copy_records(
    output_root: Path,
    groups: Sequence[PatientGroup],
    assignments: Dict[str, str],
    resume: bool,
) -> List[Dict[str, str]]:
    manifest_rows: List[Dict[str, str]] = []

    for group in groups:
        split = assignments[group.group_id]
        for record in group.records:
            destination = unique_destination(output_root, split, record)
            if not (resume and destination.exists() and destination.stat().st_size > 0):
                shutil.copy2(record.source_path, destination)

            manifest_rows.append(
                {
                    "image_path": str(destination),
                    "source_path": str(record.source_path),
                    "filename": destination.name,
                    "source_filename": record.filename,
                    "class_label": record.class_label,
                    "patient_id": record.patient_id,
                    "family_id": record.family_id,
                    "eye": record.eye,
                    "split": split,
                    "source_split": record.source_split,
                    "image_hash": record.image_hash,
                    "perceptual_hash": record.perceptual_hash,
                    "width": str(record.width),
                    "height": str(record.height),
                    "retina_quality": record.retina_quality,
                    "retina_score": f"{record.retina_score:.6f}",
                }
            )

    return manifest_rows


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def split_sets(rows: Sequence[Dict[str, str]], key: str) -> Dict[str, set]:
    values: Dict[str, set] = {split: set() for split in SPLIT_NAMES}
    for row in rows:
        values[row["split"]].add(row[key])
    return values


def pairwise_overlaps(values_by_split: Dict[str, set]) -> List[Tuple[str, str, set]]:
    overlaps = []
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1 :]:
            overlap = values_by_split[left] & values_by_split[right]
            if overlap:
                overlaps.append((left, right, overlap))
    return overlaps


def build_leakage_audit(rows: Sequence[Dict[str, str]], near_duplicate_threshold: int) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    audit_rows: List[Dict[str, str]] = []

    for key_name, issue_type in (
        ("patient_id", "patient_id_overlap"),
        ("family_id", "filename_family_overlap"),
        ("image_hash", "sha256_duplicate_overlap"),
    ):
        for left, right, overlap in pairwise_overlaps(split_sets(rows, key_name)):
            for value in sorted(overlap):
                audit_rows.append(
                    {
                        "issue_type": issue_type,
                        "left_split": left,
                        "right_split": right,
                        "key": value,
                        "detail": f"{key_name} appears in both {left} and {right}",
                    }
                )

    # Perceptual hash near-duplicate audit. For retinal images, loose perceptual
    # thresholds create many false positives because images share circular fundus
    # structure. The default is therefore exact dHash overlap (threshold 0).
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        phash = row.get("perceptual_hash", "")
        if phash:
            bucket_key = phash if near_duplicate_threshold == 0 else phash[:4]
            buckets[bucket_key].append(row)

    seen_pairs = set()
    for bucket_rows in buckets.values():
        for i, left in enumerate(bucket_rows):
            for right in bucket_rows[i + 1 :]:
                if left["split"] == right["split"]:
                    continue
                pair_key = tuple(sorted((left["image_hash"], right["image_hash"])))
                if pair_key in seen_pairs:
                    continue
                distance = hamming_hex(left["perceptual_hash"], right["perceptual_hash"])
                if distance <= near_duplicate_threshold:
                    seen_pairs.add(pair_key)
                    audit_rows.append(
                        {
                            "issue_type": "perceptual_near_duplicate_overlap",
                            "left_split": left["split"],
                            "right_split": right["split"],
                            "key": f"{left['perceptual_hash']}~{right['perceptual_hash']}",
                            "detail": (
                                f"hamming={distance}; "
                                f"{left['image_path']} <-> {right['image_path']}"
                            ),
                        }
                    )

    summary = Counter(row["issue_type"] for row in audit_rows)
    return audit_rows, dict(summary)


def build_summary(groups: Sequence[PatientGroup], assignments: Dict[str, str], rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    patient_counts = Counter(assignments.values())
    image_counts = Counter(row["split"] for row in rows)
    class_counts: Dict[str, Counter] = {split: Counter() for split in SPLIT_NAMES}
    for row in rows:
        class_counts[row["split"]][row["class_label"]] += 1

    summary_rows: List[Dict[str, str]] = []
    for split in SPLIT_NAMES:
        row = {
            "split": split,
            "patients": str(patient_counts[split]),
            "images": str(image_counts[split]),
        }
        for cls in CLASS_NAMES:
            row[f"{cls}_images"] = str(class_counts[split][cls])
        summary_rows.append(row)

    total = {
        "split": "total",
        "patients": str(len(groups)),
        "images": str(len(rows)),
    }
    total_classes = Counter(row["class_label"] for row in rows)
    for cls in CLASS_NAMES:
        total[f"{cls}_images"] = str(total_classes[cls])
    summary_rows.append(total)
    return summary_rows


def print_distribution(summary_rows: Sequence[Dict[str, str]]) -> None:
    print("\nSplit distribution")
    print("-" * 78)
    header = f"{'split':<8} {'patients':>10} {'images':>10} {'normal':>10} {'at_risk':>10} {'disease':>10}"
    print(header)
    print("-" * 78)
    for row in summary_rows:
        print(
            f"{row['split']:<8} {row['patients']:>10} {row['images']:>10} "
            f"{row['normal_images']:>10} {row['at_risk_images']:>10} {row['disease_detected_images']:>10}"
        )


def run(args: argparse.Namespace) -> int:
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_root}")

    print("\nPatient-safe retinal dataset splitter")
    print("=" * 78)
    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"Seed  : {args.seed}")

    rejected: List[Dict[str, str]] = []
    cache_manifest = Path(args.cache_manifest).resolve() if args.cache_manifest else None
    if cache_manifest and cache_manifest.exists():
        print(f"\nLoading cached scan metadata from {cache_manifest}...")
        records = load_records_from_manifest(cache_manifest)
    else:
        print("\nScanning images and computing integrity hashes...")
        records, rejected = scan_dataset(input_root, reject_non_retinal=args.reject_non_retinal)
    if not records:
        raise RuntimeError("No valid retinal images were found under the input dataset.")

    groups = build_groups(records)
    assignments = greedy_patient_split(groups, DEFAULT_SPLIT_RATIOS, args.seed)

    prepare_output_dirs(output_root, resume=args.resume)
    print("\nCopying images into patient-safe split folders...")
    manifest_rows = copy_records(output_root, groups, assignments, resume=args.resume)

    manifest_fields = [
        "image_path",
        "filename",
        "class_label",
        "patient_id",
        "split",
        "image_hash",
        "width",
        "height",
        "source_path",
        "source_filename",
        "family_id",
        "eye",
        "source_split",
        "perceptual_hash",
        "retina_quality",
        "retina_score",
    ]
    write_csv(output_root / "dataset_manifest.csv", manifest_rows, manifest_fields)

    summary_rows = build_summary(groups, assignments, manifest_rows)
    summary_fields = ["split", "patients", "images"] + [f"{cls}_images" for cls in CLASS_NAMES]
    write_csv(output_root / "patient_split_summary.csv", summary_rows, summary_fields)

    audit_rows, audit_summary = build_leakage_audit(manifest_rows, args.near_duplicate_threshold)
    audit_fields = ["issue_type", "left_split", "right_split", "key", "detail"]
    write_csv(output_root / "leakage_audit.csv", audit_rows, audit_fields)

    if rejected:
        write_csv(output_root / "rejected_images.csv", rejected, ["image_path", "reason"])

    print("\nDataset scan summary")
    print("-" * 78)
    print(f"Total patients/groups : {len(groups):,}")
    print(f"Total valid images    : {len(manifest_rows):,}")
    print(f"Rejected images       : {len(rejected):,}")
    print(f"Per-class counts      : {dict(Counter(row['class_label'] for row in manifest_rows))}")
    print_distribution(summary_rows)

    hard_issue_types = {
        "patient_id_overlap",
        "filename_family_overlap",
        "sha256_duplicate_overlap",
    }
    hard_issues = [row for row in audit_rows if row["issue_type"] in hard_issue_types]
    perceptual_review = [row for row in audit_rows if row["issue_type"] == "perceptual_near_duplicate_overlap"]

    print("\nFinal leakage report")
    print("-" * 78)
    if hard_issues:
        print("FAIL: confirmed split leakage found")
        for issue_type, count in sorted(audit_summary.items()):
            if issue_type in hard_issue_types:
                print(f"  {issue_type}: {count}")
        print(f"See: {output_root / 'leakage_audit.csv'}")
    else:
        print("HARD PASS: no patient ID, filename-family, or SHA256 duplicate overlap detected.")
        if perceptual_review:
            print(
                "REVIEW: perceptual-hash similarities were found. These are not treated as "
                "confirmed leakage because fundus dHash collisions are common; review the CSV "
                "if you want to manually inspect near-duplicate candidates."
            )
            print(f"  perceptual_near_duplicate_overlap: {len(perceptual_review)}")
            print(f"See: {output_root / 'leakage_audit.csv'}")
        else:
            print("No perceptual-hash review warnings detected.")

    print("\nGenerated files")
    print("-" * 78)
    print(output_root / "dataset_manifest.csv")
    print(output_root / "patient_split_summary.csv")
    print(output_root / "leakage_audit.csv")
    if rejected:
        print(output_root / "rejected_images.csv")

    return 0 if not hard_issues else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a patient-level safe retinal train/val/test split.")
    parser.add_argument("--input", default="dataset", help="Input dataset root.")
    parser.add_argument("--output", default="research_patient_safe_dataset", help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic random seed.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing copied files when possible.")
    parser.add_argument(
        "--cache-manifest",
        default="",
        help="Optional previous dataset_manifest.csv to reuse scan metadata and hashes.",
    )
    parser.add_argument(
        "--reject-non-retinal",
        action="store_true",
        help="Reject images that fail the lightweight retina heuristic.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=0,
        help=(
            "Maximum dHash Hamming distance considered a cross-split near duplicate. "
            "Default 0 means exact perceptual-hash overlap only; use 2-4 for a stricter "
            "manual review pass, but expect false positives on retinal fundus images."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
