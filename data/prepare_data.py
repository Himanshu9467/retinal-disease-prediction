import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
LABELS  = "trainLabels.csv"
IMG_DIR = "Images"
SAMPLE  = 0.30  # use 30% to fit in 8GB RAM

# Load & map labels
df = pd.read_csv(LABELS)
label_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
df["label"]      = df["level"].map(label_map)
df["image_path"] = df["image"] + ".jpeg"

# Sample 30%
df = df.sample(frac=SAMPLE, random_state=42)
print(f"Using {len(df)} images")

# Keep only images that actually exist
df = df[df["image_path"].apply(
    lambda x: os.path.exists(os.path.join(IMG_DIR, x))
)]
print(f"Found {len(df)} valid images")

# Split 80/20
train_df, val_df = train_test_split(
    df[["image_path", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv",     index=False)

print(f"\nTrain : {len(train_df)} images")
print(f"Val   : {len(val_df)} images")
mapping = {0:"Normal", 1:"At-Risk", 2:"Disease Detected"}
print(train_df["label"].value_counts().sort_index().rename(mapping))