"""
FINAL PATIENT-SAFE RETINAL TRAINING PIPELINE
============================================

FINAL STABLE RESEARCH VERSION

MODELS:
✅ EfficientNet-B3
✅ ResNet50
✅ Vision Transformer (ViT-B16)
✅ Custom CNN

FEATURES:
✅ Patient-safe dataset
✅ Augmentation-safe dataset
✅ Weighted CrossEntropy
✅ Label Smoothing
✅ Mixed Precision
✅ Early Stopping
✅ ReduceLROnPlateau
✅ Gradient Clipping
✅ ROC-AUC
✅ Weighted F1
✅ Macro F1
✅ Confusion Matrix Saving
✅ Stable RTX 3050Ti Training
✅ Reduced Overfitting
✅ Reduced Underfitting
✅ Safer AMP
✅ Safer Best-Model Recovery
✅ Stable Windows Dataloader

DATASET:
research_patient_safe_aug_dataset
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import json
import copy
import random
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

import matplotlib.pyplot as plt

from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================================================
# SEED
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(PROJECT_ROOT, "research_patient_safe_aug_dataset")
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "outputs")
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    "epochs": 35,

    "weight_decay": 1e-3,

    "patience": 5,

    "num_classes": 3,

    "label_smoothing": 0.05
}

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
AMP_ENABLED = DEVICE.type == "cuda"

print(f"\nUsing Device: {DEVICE}")

# =========================================================
# CUSTOM CNN
# =========================================================

class RetinalCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(128, 3)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# =========================================================
# MODEL FACTORY
# =========================================================

def build_model(model_name):

    # -----------------------------------------------------
    # EFFICIENTNET
    # -----------------------------------------------------

    if model_name == "efficientnet":

        model = models.efficientnet_b3(

            weights=models.EfficientNet_B3_Weights.DEFAULT
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                in_features,
                CONFIG["num_classes"]
            )
        )

        lr = 7e-5

        batch_size = 6

        img_size = 300

    # -----------------------------------------------------
    # RESNET
    # -----------------------------------------------------

    elif model_name == "resnet":

        model = models.ResNet34(

            weights=models.ResNet34_Weights.DEFAULT
        )

        in_features = model.fc.in_features

        model.fc = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                in_features,
                CONFIG["num_classes"]
            )
        )

        lr = 7e-5

        batch_size = 6

        img_size = 300

    # -----------------------------------------------------
    # VIT
    # -----------------------------------------------------

    elif model_name == "vit":

        model = models.vit_b_16(

            weights=models.ViT_B_16_Weights.DEFAULT
        )

        in_features = model.heads.head.in_features

        model.heads.head = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(
                in_features,
                CONFIG["num_classes"]
            )
        )

        lr = 1e-5

        batch_size =4

        img_size = 224

    # -----------------------------------------------------
    # CNN
    # -----------------------------------------------------

    elif model_name == "cnn":

        model = RetinalCNN()

        lr = 1e-4

        batch_size = 6

        img_size = 300

    else:

        raise ValueError("Unknown model")

    return (
        model.to(DEVICE),
        lr,
        batch_size,
        img_size
    )

# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    all_labels,
    all_preds,
    all_probs,
    loss
):

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            all_labels,
            np.array(all_probs),
            multi_class="ovr"
        )

    except Exception:

        roc_auc = 0.0

    metrics = {

        "loss": loss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "weighted_f1": weighted_f1,

        "macro_f1": macro_f1,

        "roc_auc": roc_auc
    }

    return metrics

# =========================================================
# RUN EPOCH
# =========================================================

def run_epoch(

    model,
    loader,
    optimizer,
    criterion,
    scaler,
    train=True
):

    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    valid_batches = 0
    skipped_batches = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader):

        images = images.to(DEVICE, non_blocking=AMP_ENABLED)
        labels = labels.to(DEVICE, non_blocking=AMP_ENABLED)

        if train:

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):

                outputs = model(images)

                if not torch.isfinite(outputs).all():
                    skipped_batches += 1
                    continue

                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            with torch.no_grad():

                with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):

                    outputs = model(images)

                    if not torch.isfinite(outputs).all():
                        skipped_batches += 1
                        continue

                    loss = criterion(outputs, labels)

                    if not torch.isfinite(loss):
                        skipped_batches += 1
                        continue

        running_loss += loss.item()
        valid_batches += 1

        probs = torch.softmax(
            outputs.float(),
            dim=1
        )

        probs = torch.nan_to_num(
            probs,
            nan=1e-7
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_preds.extend(
            preds.cpu().numpy().tolist()
        )

        all_probs.extend(
            probs.detach().cpu().numpy().tolist()
        )

    if valid_batches == 0:
        raise RuntimeError("No valid batches were processed; check images, labels, and numerical stability.")

    if skipped_batches:
        print(f"WARNING: skipped {skipped_batches} non-finite batch(es)")

    epoch_loss = running_loss / valid_batches

    metrics = calculate_metrics(
        all_labels,
        all_preds,
        all_probs,
        epoch_loss
    )

    return metrics

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(model_name):

    print("\n" + "#" * 70)
    print(f"TRAINING: {model_name.upper()}")
    print("#" * 70)

    model, lr, batch_size, img_size = build_model(model_name)

    # =====================================================
    # TRANSFORMS
    # =====================================================

    train_transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(15),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.92, 1.08)
        ),

        transforms.ColorJitter(
            brightness=0.22,
            contrast=0.22,
            saturation=0.08
        ),

        transforms.ToTensor(),
        
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.08)
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------------------------------

    val_transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =====================================================
    # DATASETS
    # =====================================================

    train_dataset = datasets.ImageFolder(

        os.path.join(DATASET_PATH, "train"),

        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(

        os.path.join(DATASET_PATH, "val"),

        transform=val_transform
    )

    test_dataset = datasets.ImageFolder(

        os.path.join(DATASET_PATH, "test"),

        transform=val_transform
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    num_workers = 4
    persistent_flag = True
    pin_memory_flag = True

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=pin_memory_flag,

        persistent_workers=persistent_flag,
        
        drop_last=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory_flag,

        persistent_workers=persistent_flag
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory_flag,

        persistent_workers=persistent_flag
    )

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    targets = [

        label for _, label
        in train_dataset.samples
    ]

    class_weights = compute_class_weight(

        class_weight="balanced",

        classes=np.unique(targets),

        y=targets
    )

    class_weights = class_weights ** 0.75

    class_weights = torch.tensor(

        class_weights,

        dtype=torch.float32
    ).to(DEVICE)

    print("\nClass Weights:")
    print(class_weights)

    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        label_smoothing=CONFIG["label_smoothing"]
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(

        model.parameters(),

        lr=lr,

        weight_decay=CONFIG["weight_decay"]
    )

    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=2,

        min_lr=1e-6
    )

    # =====================================================
    # AMP
    # =====================================================

    scaler = GradScaler("cuda", enabled=AMP_ENABLED)

    # =====================================================
    # BEST MODEL SAFETY
    # =====================================================

    best_f1 = 0.0

    patience_counter = 0

    best_model = copy.deepcopy(
        model.state_dict()
    )

    history = []

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(CONFIG["epochs"]):

        print(
            f"\nEpoch "
            f"[{epoch+1}/{CONFIG['epochs']}]"
        )

        train_metrics = run_epoch(

            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            train=True
        )

        val_metrics = run_epoch(

            model,
            val_loader,
            optimizer,
            criterion,
            scaler,
            train=False
        )

        scheduler.step(
            val_metrics["weighted_f1"]
        )

        print(

            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "

            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "

            f"Val Acc: "
            f"{val_metrics['accuracy']*100:.2f}% | "

            f"Weighted F1: "
            f"{val_metrics['weighted_f1']*100:.2f}% | "

            f"Macro F1: "
            f"{val_metrics['macro_f1']*100:.2f}% | "

            f"ROC-AUC: "
            f"{val_metrics['roc_auc']:.4f}"
        )

        history.append({

            "epoch": epoch + 1,

            "train": train_metrics,

            "val": val_metrics
        })

        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if val_metrics["weighted_f1"] > best_f1:

            best_f1 = val_metrics["weighted_f1"]

            patience_counter = 0

            best_model = copy.deepcopy(
                model.state_dict()
            )

            torch.save(

                best_model,

                os.path.join(
                    OUTPUT_DIR,
                    f"{model_name}_best.pth"
                )
            )

            print("✅ Best model updated")

        else:

            patience_counter += 1

        # =================================================
        # EARLY STOPPING
        # =================================================

        if patience_counter >= CONFIG["patience"]:

            print("\n⛔ Early stopping triggered")

            break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    model.load_state_dict(best_model)

    # =====================================================
    # FINAL TEST
    # =====================================================

    test_metrics = run_epoch(

        model,
        test_loader,
        optimizer,
        criterion,
        scaler,
        train=False
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE, non_blocking=AMP_ENABLED)

            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):

                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_preds.extend(
                preds.cpu().numpy().tolist()
            )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    class_names = [
        "at_risk",
        "disease_detected",
        "normal"
    ]

    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title(
        f"{model_name.upper()} Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=15
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    for i in range(len(class_names)):

        for j in range(len(class_names)):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_confusion_matrix.png"
        )
    )

    plt.close()

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy    : "
        f"{test_metrics['accuracy']*100:.2f}%"
    )

    print(
        f"Weighted F1 : "
        f"{test_metrics['weighted_f1']*100:.2f}%"
    )

    print(
        f"Macro F1    : "
        f"{test_metrics['macro_f1']*100:.2f}%"
    )

    print(
        f"ROC-AUC     : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    # =====================================================
    # SAVE HISTORY
    # =====================================================

    with open(

        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_history.json"
        ),

        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    MODELS = [
        "resnet",
        "efficientnet",
        "vit"
    ]

    for model_name in MODELS:

        torch.cuda.empty_cache()

        train_model(model_name)

    print("\n🎉 ALL TRAINING COMPLETED!")
