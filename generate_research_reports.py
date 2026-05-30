"""Generate research-grade evaluation reports for final retinal models.

Outputs:
  research_reports/{efficientnet,resnet,vit,cnn}/...
  final_research_summary.md
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from model_registry import MODEL_KEYS, build_model_registry, registry_as_dict
from production_models import build_final_model

SEED = 42
CLASS_NAMES = ["at_risk", "disease_detected", "normal"]
DISPLAY_NAMES = ["At-Risk", "Disease Detected", "Normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_deterministic(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def val_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_dataset(dataset_root: Path, split: str, img_size: int):
    dataset = datasets.ImageFolder(str(dataset_root / split), transform=val_transform(img_size))
    if dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(f"Unexpected class mapping: {dataset.class_to_idx}; expected {CLASS_TO_IDX}")
    return dataset


def load_checkpoint(model_name: str, checkpoint_path: Path, device):
    model = build_final_model(model_name, num_classes=len(CLASS_NAMES))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"{model_name} checkpoint does not contain a state_dict: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"{model_name} checkpoint mismatch: missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    preds, targets, probs, paths = [], [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        batch_probs = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(batch_probs)
        preds.extend(batch_probs.argmax(axis=1).tolist())
        targets.extend(labels.numpy().tolist())
        paths.extend([sample[0] for sample in loader.dataset.samples[len(paths):len(paths) + len(labels)]])
    return np.asarray(preds), np.asarray(targets), np.vstack(probs), paths


def expected_calibration_error(targets, probs, bins: int = 15):
    confidence = probs.max(axis=1)
    correctness = (probs.argmax(axis=1) == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not np.any(mask):
            continue
        acc = float(correctness[mask].mean())
        conf = float(confidence[mask].mean())
        frac = float(mask.mean())
        ece += frac * abs(acc - conf)
        bin_rows.append({"low": float(low), "high": float(high), "accuracy": acc, "confidence": conf, "fraction": frac})
    return float(ece), bin_rows


def bootstrap_ci(targets, preds, probs, n_bootstrap: int = 1000):
    rng = np.random.default_rng(SEED)
    n = len(targets)
    metrics = {"accuracy": [], "weighted_f1": [], "roc_auc": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_true = targets[idx]
        y_pred = preds[idx]
        y_prob = probs[idx]
        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["weighted_f1"].append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        try:
            metrics["roc_auc"].append(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=list(range(3)))
            )
        except ValueError:
            pass
    return {
        key: {
            "mean": float(np.mean(values)) if values else 0.0,
            "ci95_low": float(np.percentile(values, 2.5)) if values else 0.0,
            "ci95_high": float(np.percentile(values, 97.5)) if values else 0.0,
        }
        for key, values in metrics.items()
    }


def per_class_specificity(cm):
    specs = {}
    total = cm.sum()
    for idx, name in enumerate(CLASS_NAMES):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        specs[name] = float(tn / max(tn + fp, 1))
    return specs


def brier_multiclass(targets, probs):
    y = label_binarize(targets, classes=list(range(len(CLASS_NAMES))))
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def save_confusion_matrix(cm, out_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES)
    plt.title("Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_curve(targets, probs, out_path):
    y_bin = label_binarize(targets, classes=list(range(3)))
    plt.figure(figsize=(7, 6))
    for idx, name in enumerate(DISPLAY_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], probs[:, idx])
        auc_val = roc_auc_score(y_bin[:, idx], probs[:, idx])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("One-vs-Rest ROC Curves", fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pr_curve(targets, probs, out_path):
    y_bin = label_binarize(targets, classes=list(range(3)))
    plt.figure(figsize=(7, 6))
    for idx, name in enumerate(DISPLAY_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], probs[:, idx])
        ap = average_precision_score(y_bin[:, idx], probs[:, idx])
        plt.plot(recall, precision, linewidth=2, label=f"{name} AP={ap:.3f}")
    plt.title("Precision-Recall Curves", fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_training_history(history_path, out_path):
    history = json.loads(Path(history_path).read_text(encoding="utf-8")) if history_path and Path(history_path).exists() else []
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, metric, title in zip(axes, ["loss", "accuracy", "weighted_f1"], ["Loss", "Accuracy", "Weighted F1"]):
        ax.plot(epochs, [row["train"].get(metric, np.nan) for row in history], label="Train", linewidth=2)
        ax.plot(epochs, [row["val"].get(metric, np.nan) for row in history], label="Validation", linewidth=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_calibration(targets, probs, out_path):
    confidence = probs.max(axis=1)
    correctness = (probs.argmax(axis=1) == targets).astype(int)
    prob_true, prob_pred = calibration_curve(correctness, confidence, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6.5, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    plt.title("Calibration Curve", fontweight="bold")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_image_panel(rows, out_path, title):
    if not rows:
        return
    n = min(len(rows), 9)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, rows[:n]):
        img = Image.open(row["path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"T: {DISPLAY_NAMES[row['target']]}\nP: {DISPLAY_NAMES[row['pred']]}\nC: {row['confidence']:.2f}",
            fontsize=8,
        )
    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def gradcam_or_saliency(model, model_name, image_path, out_path, device, img_size):
    raw = Image.open(image_path).convert("RGB")
    tensor = val_transform(img_size)(raw).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    cls = int(logits.argmax(dim=1).item())
    logits[0, cls].backward()
    saliency = tensor.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    plt.figure(figsize=(5, 5))
    plt.imshow(raw.resize((img_size, img_size)))
    plt.imshow(saliency, cmap="magma", alpha=0.45)
    plt.axis("off")
    plt.title(f"{model_name} explanation: {DISPLAY_NAMES[cls]}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def failure_rows(targets, preds, probs, paths):
    rows = []
    for target, pred, prob, path in zip(targets, preds, probs, paths):
        rows.append(
            {
                "path": str(path),
                "target": int(target),
                "pred": int(pred),
                "confidence": float(prob[pred]),
                "true_probability": float(prob[target]),
                "margin": float(np.partition(prob, -2)[-1] - np.partition(prob, -2)[-2]),
            }
        )
    wrong = [row for row in rows if row["target"] != row["pred"]]
    return {
        "most_confident_wrong": sorted(wrong, key=lambda r: r["confidence"], reverse=True)[:20],
        "hardest_samples": sorted(rows, key=lambda r: r["true_probability"])[:20],
        "class_confusions": sorted(wrong, key=lambda r: (r["target"], r["pred"], -r["confidence"]))[:30],
    }


def evaluate_one(model_name, entry, loader, report_root, device, img_size, bootstrap):
    out_dir = report_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction_samples").mkdir(exist_ok=True)
    (out_dir / "gradcam_examples").mkdir(exist_ok=True)

    model = load_checkpoint(model_name, Path(entry.checkpoint), device)
    preds, targets, probs, paths = predict(model, loader, device)
    cm = confusion_matrix(targets, preds, labels=list(range(3)))
    report = classification_report(
        targets, preds, labels=list(range(3)), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    ece, calibration_bins = expected_calibration_error(targets, probs)
    metrics = {
        "model": model_name,
        "checkpoint": entry.checkpoint,
        "version": entry.version,
        "accuracy": float(accuracy_score(targets, preds)),
        "precision_weighted": float(precision_score(targets, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(targets, preds, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "roc_auc_weighted_ovr": float(roc_auc_score(targets, probs, multi_class="ovr", average="weighted", labels=list(range(3)))),
        "ece": ece,
        "brier_score": brier_multiclass(targets, probs),
        "support_counts": {CLASS_NAMES[i]: int(cm[i].sum()) for i in range(3)},
        "specificity": per_class_specificity(cm),
        "classification_report": report,
        "bootstrap_ci": bootstrap_ci(targets, preds, probs, bootstrap),
        "calibration_bins": calibration_bins,
    }

    failures = failure_rows(targets, preds, probs, paths)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "failure_cases.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(targets, preds, labels=list(range(3)), target_names=CLASS_NAMES, zero_division=0),
        encoding="utf-8",
    )
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    save_roc_curve(targets, probs, out_dir / "roc_curve.png")
    save_pr_curve(targets, probs, out_dir / "precision_recall_curve.png")
    save_training_history(entry.history, out_dir / "training_history.png")
    shutil.copyfile(out_dir / "training_history.png", out_dir / "loss_curve.png")
    shutil.copyfile(out_dir / "training_history.png", out_dir / "accuracy_curve.png")
    save_calibration(targets, probs, out_dir / "calibration_curve.png")
    save_image_panel(failures["most_confident_wrong"], out_dir / "prediction_samples" / "most_confident_wrong.png", "Most Confident Wrong Predictions")
    save_image_panel(failures["hardest_samples"], out_dir / "prediction_samples" / "hardest_samples.png", "Hardest Samples")
    save_image_panel(failures["class_confusions"], out_dir / "prediction_samples" / "class_confusions.png", "Class Confusion Examples")

    for idx, row in enumerate(failures["hardest_samples"][:3]):
        gradcam_or_saliency(model, model_name, row["path"], out_dir / "gradcam_examples" / f"explanation_{idx + 1}.png", device, img_size)

    return metrics


def write_summary(metrics_by_model, registry, out_path):
    best = max(metrics_by_model, key=lambda m: (m["weighted_f1"], m["roc_auc_weighted_ovr"]))
    lines = [
        "# Final Research Summary",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy | Weighted F1 | ROC-AUC | ECE | Brier | Ensemble Weight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in sorted(metrics_by_model, key=lambda m: m["model"]):
        entry = registry[metrics["model"]]
        lines.append(
            f"| {metrics['model']} | {metrics['accuracy']:.4f} | {metrics['weighted_f1']:.4f} | "
            f"{metrics['roc_auc_weighted_ovr']:.4f} | {metrics['ece']:.4f} | {metrics['brier_score']:.4f} | {entry.weight:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Model Selection",
            "",
            f"Best single model by weighted F1 then ROC-AUC: **{best['model']}**.",
            "",
            "## Overfitting Analysis",
            "",
            "See each model's `training_history.png` for train-validation divergence. A large validation plateau with rising train performance should be treated as overfitting evidence.",
            "",
            "## Generalization Discussion",
            "",
            "These reports evaluate the final checkpoints against the configured held-out split. External-site validation is still required before clinical claims.",
            "",
            "## ROC-AUC Comparison",
            "",
            "Weighted one-vs-rest ROC-AUC is included in `metrics.json` and visualized in each `roc_curve.png`.",
            "",
            "## Confusion Matrix Interpretation",
            "",
            "Inspect off-diagonal cells for clinically important confusions, especially disease_detected predicted as normal or at_risk.",
            "",
            "## GradCAM Interpretation",
            "",
            "The saved explanation overlays are screening aids only. They should highlight retinal structures rather than borders, labels, or acquisition artifacts.",
            "",
            "## Failure-Case Analysis",
            "",
            "Each model folder includes `failure_cases.json` and image panels for most confident wrong, hardest, and class-confusion examples.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(ROOT / "research_patient_safe_dataset"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--reports-dir", default=str(ROOT / "research_reports"))
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_deterministic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = build_model_registry(strict=True)
    report_root = Path(args.reports_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "model_registry.json").write_text(json.dumps(registry_as_dict(registry), indent=2), encoding="utf-8")

    dataset = load_dataset(Path(args.dataset_root), args.split, args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics_by_model = []
    for model_name in MODEL_KEYS:
        metrics_by_model.append(
            evaluate_one(model_name, registry[model_name], loader, report_root, device, args.img_size, args.bootstrap)
        )
    write_summary(metrics_by_model, registry, ROOT / "final_research_summary.md")
    print(f"Research reports written to {report_root}")
    print("Summary written to final_research_summary.md")


if __name__ == "__main__":
    main()
