import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

CSV_15 = r"data/labels/trainLabels15.csv"
CSV_19 = r"data/labels/trainLabels19.csv"

IMAGE_DIR = r"data/resized_traintest15_train19"

OUTPUT_DIR = r"prepared_dataset"

# ============================================================
# LABEL MAPPING
# ============================================================

def map_label(level):

    level = int(level)

    if level == 0:
        return "normal"

    elif level in [1, 2]:
        return "at_risk"

    else:
        return "disease_detected"

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

splits = ["train", "val", "test"]

classes = [
    "normal",
    "at_risk",
    "disease_detected"
]

for split in splits:

    for cls in classes:

        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )

# ============================================================
# LOAD CSV FILES
# ============================================================

df15 = pd.read_csv(CSV_15)
df19 = pd.read_csv(CSV_19)

# ============================================================
# MERGE DATAFRAMES
# ============================================================

df = pd.concat([df15, df19], ignore_index=True)

print("Columns:", df.columns)

# ============================================================
# FIX DIFFERENT COLUMN NAMES
# ============================================================

# Merge label columns
df["label"] = df["level"].fillna(df["diagnosis"])

# Merge image name columns
df["img_name"] = df["image"].fillna(df["id_code"])

# Remove missing rows
df = df.dropna(subset=["label", "img_name"])

print("Total records:", len(df))

# ============================================================
# CREATE NEW CLASS COLUMN
# ============================================================

df["class"] = df["label"].apply(map_label)

print("\nClass Distribution:")
print(df["class"].value_counts())

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["class"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["class"],
    random_state=42
)

print("\nDataset Split:")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(dataframe, split_name):

    copied = 0
    missing = 0

    for _, row in dataframe.iterrows():

        image_name = str(row["img_name"]) + ".jpg"

        label = row["class"]

        src = os.path.join(
            IMAGE_DIR,
            image_name
        )

        dst = os.path.join(
            OUTPUT_DIR,
            split_name,
            label,
            image_name
        )

        if os.path.exists(src):

            shutil.copy2(src, dst)

            copied += 1

        else:

            missing += 1

    print(f"\n{split_name.upper()}")
    print(f"Copied Images : {copied}")
    print(f"Missing Images: {missing}")

# ============================================================
# COPY DATA
# ============================================================

copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("\nDataset preparation completed successfully!")