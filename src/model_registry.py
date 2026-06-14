"""Central production model registry for backend inference and reporting."""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)

MODEL_KEYS = ("efficientnet", "resnet", "mobilenet", "cnn")
PRODUCTION_MODEL_KEYS = ("efficientnet", "resnet", "mobilenet", "cnn")
FALLBACK_ORDER = MODEL_KEYS
MODEL_LABELS = {
    "efficientnet": "EfficientNet-B3",
    "resnet": "ResNet50",
    "mobilenet": "MobileNetV3-Large",
    "cnn": "CNN",
}
METRIC_KEYS = ("accuracy", "macro_f1", "roc_auc")
DEFAULT_WEIGHTS = {
    "efficientnet": 0.50,
    "resnet": 0.20,
    "mobilenet": 0.20,
    "cnn": 0.10,
}


@dataclass(frozen=True)
class ModelRegistryEntry:
    name: str
    display_name: str
    checkpoint: str
    history: str
    priority: int
    status: str
    version: str
    metrics: Dict[str, float]
    weight: float
    sha256: str
    modified_utc: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def outputs_dir() -> Path:
    return Path(os.environ.get("MODEL_REGISTRY_DIR", project_root() / "research_training_outputs2")).resolve()


def _artifact_path(output_dir: Path, model_name: str, filename: str, legacy_filename: str = "") -> Path:
    nested = output_dir / model_name / filename
    if nested.exists():
        return nested
    if legacy_filename:
        flat = output_dir / legacy_filename
        if flat.exists():
            LOGGER.info(
                "Using flat production artifact for %s: %s. Preferred path is %s",
                model_name,
                flat,
                nested,
            )
            return flat
    return nested


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_history(path: Path) -> List[dict]:
    if not path.exists() or path.suffix.lower() != ".json":
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not parse history %s: %s", path, exc)
        return []
    return data if isinstance(data, list) else []


def _load_csv_metrics(path: Path, model_name: str) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        import csv

        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        LOGGER.warning("Could not parse CSV metrics %s: %s", path, exc)
        return {}

    if not rows:
        return {}

    if "model" in rows[0]:
        for row in rows:
            if str(row.get("model", "")).lower() == model_name:
                return {key: _safe_float(row.get(key)) for key in row if key != "model"}
        return {}

    def score(row: dict) -> float:
        return _safe_float(row.get("macro_f1", row.get("val_macro_f1", row.get("f1"))), -1.0)

    row = max(rows, key=score)
    return {
        "accuracy": _safe_float(row.get("val_accuracy", row.get("accuracy"))),
        "precision": _safe_float(row.get("precision", row.get("val_precision"))),
        "recall": _safe_float(row.get("recall", row.get("val_recall"))),
        "sensitivity": _safe_float(row.get("sensitivity", row.get("val_sensitivity"))),
        "specificity": _safe_float(row.get("specificity", row.get("val_specificity"))),
        "macro_f1": _safe_float(row.get("macro_f1", row.get("val_macro_f1", row.get("f1")))),
        "roc_auc": _safe_float(row.get("roc_auc", row.get("val_roc_auc"))),
        "pr_auc": _safe_float(row.get("pr_auc", row.get("val_pr_auc"))),
    }


def _load_metrics(path: Path, history: Iterable[dict]) -> Dict[str, float]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                metrics = {}
                for key, value in data.items():
                    normalized_key = key[5:] if key.startswith("test_") else key
                    if normalized_key in {
                        "accuracy",
                        "precision",
                        "recall",
                        "sensitivity",
                        "specificity",
                        "macro_f1",
                        "roc_auc",
                        "pr_auc",
                    }:
                        metrics[normalized_key] = _safe_float(value)
                return metrics
        except Exception as exc:
            LOGGER.warning("Could not parse metrics %s: %s", path, exc)
    return best_validation_metrics(history)


def best_validation_metrics(history: Iterable[dict]) -> Dict[str, float]:
    best = {}
    best_score = float("-inf")
    for row in history:
        val = row.get("val") if isinstance(row, dict) else None
        if not isinstance(val, dict):
            continue
        score = _safe_float(val.get("macro_f1", val.get("weighted_f1", val.get("f1"))), -1.0)
        if score > best_score:
            best_score = score
            best = dict(val)
            best["epoch"] = row.get("epoch")

    if not best:
        return {}

    if "macro_f1" not in best and "f1" in best:
        best["macro_f1"] = best["f1"]
    return {key: _safe_float(value) for key, value in best.items()}


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_score(metrics: Dict[str, float]) -> float:
    available = [_safe_float(metrics.get(key)) for key in METRIC_KEYS if metrics.get(key) is not None]
    if not available:
        return 0.0
    return sum(available) / len(available)


def compute_ensemble_weights(metrics_by_model: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Return validated production soft-voting weights for all four binary models."""
    del metrics_by_model
    return dict(DEFAULT_WEIGHTS)


def build_model_registry(strict: bool = True) -> Dict[str, ModelRegistryEntry]:
    output_dir = outputs_dir()
    registry: Dict[str, ModelRegistryEntry] = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    for priority, model_name in enumerate(MODEL_KEYS, start=1):
        checkpoint = _artifact_path(output_dir, model_name, "best_model.pth", f"best_{model_name}.pt")
        history_path = _artifact_path(output_dir, model_name, "history.json", f"metrics_{model_name}.csv")
        summary_path = _artifact_path(output_dir, model_name, "test_summary.json", f"test_summary_{model_name}.json")
        metrics_path = _artifact_path(output_dir, model_name, "metrics.json")
        if not checkpoint.exists():
            if strict and model_name == "efficientnet":
                raise FileNotFoundError(f"Missing production checkpoint: {checkpoint}")
            LOGGER.warning("Skipping missing checkpoint: %s", checkpoint)
            stat = None
        else:
            stat = checkpoint.stat()

        history = _load_history(history_path)
        metrics = (
            _load_metrics(summary_path, [])
            or _load_csv_metrics(output_dir / "all_models_summary.csv", model_name)
            or _load_metrics(metrics_path, history)
            or _load_csv_metrics(history_path, model_name)
        )
        metrics_by_model[model_name] = metrics
        registry[model_name] = ModelRegistryEntry(
            name=model_name,
            display_name=MODEL_LABELS[model_name],
            checkpoint=str(checkpoint) if checkpoint.exists() else "",
            history=str(history_path) if history_path.exists() else "",
            priority=priority,
            status="production" if model_name == "efficientnet" and checkpoint.exists() else "fallback" if checkpoint.exists() else "missing",
            version=(
                f"final-{datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime('%Y%m%d%H%M%S')}"
                if stat
                else "not-loaded"
            ),
            metrics=metrics,
            weight=0.0,
            sha256=_sha256(checkpoint) if checkpoint.exists() else "",
            modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else "",
        )

    configured = os.environ.get("ENSEMBLE_WEIGHTS_JSON")
    if configured:
        try:
            weights = {str(k).lower(): float(v) for k, v in json.loads(configured).items()}
        except Exception as exc:
            raise ValueError(f"Invalid ENSEMBLE_WEIGHTS_JSON: {exc}") from exc
        total = sum(v for k, v in weights.items() if k in registry and v > 0)
        weights = {k: v / total for k, v in weights.items() if k in registry and v > 0} if total > 0 else {}
    else:
        weights = compute_ensemble_weights(metrics_by_model)

    return {
        name: ModelRegistryEntry(**{**asdict(entry), "weight": float(weights.get(name, 0.0))})
        for name, entry in registry.items()
    }


def registry_as_dict(registry: Optional[Dict[str, ModelRegistryEntry]] = None) -> Dict[str, dict]:
    return {name: asdict(entry) for name, entry in (registry or build_model_registry()).items()}


def validate_registry_paths(registry: Dict[str, ModelRegistryEntry]) -> List[str]:
    issues = []
    allowed_root = outputs_dir()
    for name, entry in registry.items():
        checkpoint = Path(entry.checkpoint).resolve() if entry.checkpoint else None
        if checkpoint is None or not checkpoint.exists():
            if name == "efficientnet":
                issues.append(f"{name}: checkpoint missing: {checkpoint}")
            continue
        if allowed_root not in checkpoint.parents and checkpoint != allowed_root:
            issues.append(f"{name}: checkpoint is outside outputs: {checkpoint}")
        if name == "efficientnet" and entry.status != "production":
            issues.append(f"{name}: status is not production: {entry.status}")
    return issues
