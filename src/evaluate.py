"""
evaluate.py — Model Evaluation with Classification Metrics
Metrics: Accuracy | Precision | Recall | F1-Score | Confusion Matrix
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import RetinalDataset, get_val_transforms, CLASS_NAMES
from models import build_model
from train import compute_metrics, compute_per_class_metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval().to(device)
    all_preds, all_targets, all_probs, all_paths = [], [], [], []

    for images, labels, paths in loader:
        images  = images.to(device, non_blocking=True)
        logits  = model(images)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
        all_probs.extend(probs)
        all_paths.extend(paths)

    return (np.array(all_preds), np.array(all_targets),
            np.array(all_probs), all_paths)


def evaluate_model(model, test_csv, img_dir, img_size=224, batch_size=32):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RetinalDataset(test_csv, img_dir, get_val_transforms(img_size))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    preds, targets, probs, paths = run_inference(model, loader, device)
    metrics    = compute_metrics(preds, targets, probs)
    per_class  = compute_per_class_metrics(preds, targets)

    print("\n─── Test Set Evaluation ───")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k:12s}: {v:.4f}")
    print("\nPer-Class Report:")
    for cls in CLASS_NAMES:
        m = per_class.get(cls, {})
        print(f"  {cls:20s} | P: {m.get('precision',0):.4f}  "
              f"R: {m.get('recall',0):.4f}  F1: {m.get('f1-score',0):.4f}")

    return preds, targets, probs, paths, metrics


# ─────────────────────────────────────────────
#  Multi-Model Comparison
# ─────────────────────────────────────────────

def compare_models(model_configs, test_csv, img_dir,
                   img_size=224, save_dir="./outputs/comparison"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RetinalDataset(test_csv, img_dir, get_val_transforms(img_size))
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    comparison    = []
    all_preds_dict = {}
    all_probs_dict = {}

    for cfg in model_configs:
        name, ckpt = cfg["model_name"], cfg["checkpoint_path"]
        print(f"\n[Evaluating] {name}")
        model = build_model(name, img_size=img_size)
        try:
            checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint does not contain a state_dict: {ckpt}")
        model.load_state_dict(state)

        preds, targets, probs, _ = run_inference(model, loader, device)
        metrics = compute_metrics(preds, targets, probs)
        metrics["model"] = name
        comparison.append(metrics)
        all_preds_dict[name] = preds
        all_probs_dict[name] = probs

    df = pd.DataFrame(comparison).set_index("model")
    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)
    print(df[["accuracy", "precision", "recall", "f1"]].to_string())
    df.to_csv(Path(save_dir) / "model_comparison.csv")

    _plot_comparison(df, all_preds_dict, all_probs_dict, targets, save_dir)
    return df


# ─────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────

def _plot_comparison(df, preds_dict, probs_dict, targets, save_dir):
    save_dir = Path(save_dir)
    colors   = plt.cm.Set2(np.linspace(0, 1, len(df)))

    # ── 1. Bar chart: all 4 metrics ─────────────────────────
    metrics_list = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, metrics_list):
        bars = ax.bar(df.index, df[metric], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.suptitle("Model Comparison — CVD Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Confusion matrices ────────────────────────────────
    n_models = len(preds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(targets, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. ROC Curves (one-vs-rest) ──────────────────────────
    targets_bin = label_binarize(targets, classes=[0, 1, 2])
    fig, axes   = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, probs), color in zip(axes, probs_dict.items(), colors):
        for i, cls in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
        ax.plot([0,1],[0,1],"k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} — ROC Curves", fontweight="bold")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[Plots saved] → {save_dir}")


# ─────────────────────────────────────────────
#  Training History Plot
# ─────────────────────────────────────────────

def plot_training_history(history_csv_paths, labels=None, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_csv_paths)))

    for i, (path, color) in enumerate(zip(history_csv_paths, colors)):
        df    = pd.read_csv(path)
        label = labels[i] if labels else Path(path).stem
        axes[0].plot(df["train_loss"], color=color, linestyle="--", label=f"{label} Train")
        axes[0].plot(df["val_loss"],   color=color, linestyle="-",  label=f"{label} Val")
        axes[1].plot(df["train_acc"],  color=color, linestyle="--", label=f"{label} Train")
        axes[1].plot(df["val_acc"],    color=color, linestyle="-",  label=f"{label} Val")
        axes[2].plot(df["train_f1"],   color=color, linestyle="--", label=f"{label} Train")
        axes[2].plot(df["val_f1"],     color=color, linestyle="-",  label=f"{label} Val")

    for ax, title in zip(axes, ["Loss", "Accuracy", "F1-Score"]):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
