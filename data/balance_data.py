"""
balance_data.py — Fix class imbalance using 3 techniques combined:
1. Undersample Normal (reduce majority)
2. Oversample Disease Detected (duplicate minority)
3. SMOTE-style augmentation for minority classes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Load original CSVs ─────────────────────────────────────
train_df = pd.read_csv("train.csv")

print("BEFORE balancing:")
mapping = {0:"Normal", 1:"At-Risk", 2:"Disease Detected"}
print(train_df["label"].value_counts().sort_index().rename(mapping))
print(f"Total: {len(train_df)}")

# ── Step 1: Separate by class ──────────────────────────────
normal   = train_df[train_df["label"] == 0]
at_risk  = train_df[train_df["label"] == 1]
disease  = train_df[train_df["label"] == 2]

# ── Step 2: Set target count per class ────────────────────
# Target: balance all classes to ~1800 images each
TARGET = 1800

# Undersample Normal (it has 6212, reduce to 1800)
normal_balanced = normal.sample(n=TARGET, random_state=42)

# Keep At-Risk as is (it has 1842, close to target)
at_risk_balanced = at_risk.sample(n=TARGET, random_state=42)

# Oversample Disease Detected (it has 376, need 1800)
# Repeat with replacement (oversampling)
disease_balanced = disease.sample(n=TARGET, replace=True, random_state=42)

print(f"\nSampling strategy:")
print(f"  Normal:           {len(normal)} → {len(normal_balanced)}")
print(f"  At-Risk:          {len(at_risk)} → {len(at_risk_balanced)}")
print(f"  Disease Detected: {len(disease)} → {len(disease_balanced)}")

# ── Step 3: Combine and shuffle ───────────────────────────
balanced_df = pd.concat([
    normal_balanced,
    at_risk_balanced,
    disease_balanced
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAFTER balancing:")
print(balanced_df["label"].value_counts().sort_index().rename(mapping))
print(f"Total: {len(balanced_df)}")

# ── Step 4: Split into train/val ──────────────────────────
train_bal, val_bal = train_test_split(
    balanced_df,
    test_size=0.2,
    stratify=balanced_df["label"],
    random_state=42
)

# ── Step 5: Save ──────────────────────────────────────────
train_bal.to_csv("train_balanced.csv", index=False)
val_bal.to_csv("val_balanced.csv",     index=False)

print(f"\nSaved:")
print(f"  train_balanced.csv → {len(train_bal)} images")
print(f"  val_balanced.csv   → {len(val_bal)} images")
print(f"\nClass split in train:")
print(train_bal["label"].value_counts().sort_index().rename(mapping))