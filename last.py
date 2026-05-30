"""
Messidor / 5-Class Retinal Dataset
→ 3-Class DR Dataset Converter

Creates:

processed_messidor_dataset/

├── train/
├── val/
└── test/

with:
- normal
- at_risk
- disease_detected

Mapping:
0 -> normal
1,2 -> at_risk
3,4 -> disease_detected

Validation split:
15% from training set

Safe for:
- your laptop
- merged DR pipeline
- EfficientNet training
"""

import os
import shutil
import random

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =========================================================
# RANDOM SEED
# =========================================================

SEED = 42

random.seed(SEED)

# =========================================================
# PATHS
# =========================================================

BASE_PATH = r"C:\Retinal-image-prediction"

INPUT_PATH = os.path.join(
    BASE_PATH,
    r"data\preprocessed dataset"
)

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "processed_messidor_dataset"
)

# =========================================================
# INPUT STRUCTURE
# =========================================================

TRAIN_INPUT = os.path.join(
    INPUT_PATH,
    "training"
)

TEST_INPUT = os.path.join(
    INPUT_PATH,
    "testing"
)

# =========================================================
# OUTPUT CLASSES
# =========================================================

CLASSES = [

    "normal",

    "at_risk",

    "disease_detected"
]

# =========================================================
# CREATE OUTPUT STRUCTURE
# =========================================================

for split in ["train", "val", "test"]:

    for cls in CLASSES:

        os.makedirs(

            os.path.join(
                OUTPUT_PATH,
                split,
                cls
            ),

            exist_ok=True
        )

# =========================================================
# CLASS MAPPING
# =========================================================

def map_class(original_class):

    original_class = int(original_class)

    # --------------------------------------------
    # NORMAL
    # --------------------------------------------

    if original_class == 0:

        return "normal"

    # --------------------------------------------
    # AT RISK
    # --------------------------------------------

    elif original_class in [1, 2]:

        return "at_risk"

    # --------------------------------------------
    # DISEASE DETECTED
    # --------------------------------------------

    elif original_class in [3, 4]:

        return "disease_detected"

    return None

# =========================================================
# GET TRAIN FILES
# =========================================================

train_samples = []

print("\n" + "=" * 70)
print("SCANNING TRAINING DATA")
print("=" * 70)

for class_folder in os.listdir(TRAIN_INPUT):

    class_path = os.path.join(
        TRAIN_INPUT,
        class_folder
    )

    if not os.path.isdir(class_path):

        continue

    mapped_class = map_class(class_folder)

    if mapped_class is None:

        continue

    files = [

        f for f in os.listdir(class_path)

        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        f"Class {class_folder} "
        f"-> {mapped_class}: "
        f"{len(files)} images"
    )

    for file in files:

        train_samples.append({

            "file":
                os.path.join(
                    class_path,
                    file
                ),

            "class":
                mapped_class,

            "name":
                file
        })

# =========================================================
# STRATIFIED TRAIN/VAL SPLIT
# =========================================================

labels = [

    sample["class"]

    for sample in train_samples
]

train_data, val_data = train_test_split(

    train_samples,

    test_size=0.15,

    stratify=labels,

    random_state=SEED
)

print("\n" + "=" * 70)
print("TRAIN / VAL SPLIT")
print("=" * 70)

print(f"Train Images: {len(train_data)}")

print(f"Val Images  : {len(val_data)}")

# =========================================================
# COPY FUNCTION
# =========================================================

def copy_samples(samples, split_name):

    counts = {

        "normal": 0,

        "at_risk": 0,

        "disease_detected": 0
    }

    for sample in tqdm(samples):

        src = sample["file"]

        target_class = sample["class"]

        filename = sample["name"]

        # --------------------------------------------
        # UNIQUE FILENAME
        # --------------------------------------------

        new_name = (
            f"messidor_{filename}"
        )

        dst = os.path.join(

            OUTPUT_PATH,

            split_name,

            target_class,

            new_name
        )

        try:

            shutil.copy2(src, dst)

            counts[target_class] += 1

        except Exception:

            continue

    return counts

# =========================================================
# PROCESS TRAIN
# =========================================================

print("\n" + "=" * 70)
print("PROCESSING TRAIN")
print("=" * 70)

train_counts = copy_samples(
    train_data,
    "train"
)

# =========================================================
# PROCESS VAL
# =========================================================

print("\n" + "=" * 70)
print("PROCESSING VAL")
print("=" * 70)

val_counts = copy_samples(
    val_data,
    "val"
)

# =========================================================
# PROCESS TEST
# =========================================================

print("\n" + "=" * 70)
print("PROCESSING TEST")
print("=" * 70)

test_counts = {

    "normal": 0,

    "at_risk": 0,

    "disease_detected": 0
}

for class_folder in os.listdir(TEST_INPUT):

    class_path = os.path.join(
        TEST_INPUT,
        class_folder
    )

    if not os.path.isdir(class_path):

        continue

    mapped_class = map_class(class_folder)

    if mapped_class is None:

        continue

    files = [

        f for f in os.listdir(class_path)

        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(
        f"Class {class_folder} "
        f"-> {mapped_class}: "
        f"{len(files)} images"
    )

    for file in tqdm(files):

        src = os.path.join(
            class_path,
            file
        )

        new_name = (
            f"messidor_{file}"
        )

        dst = os.path.join(

            OUTPUT_PATH,

            "test",

            mapped_class,

            new_name
        )

        try:

            shutil.copy2(src, dst)

            test_counts[mapped_class] += 1

        except Exception:

            continue

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print("\nTRAIN")

for cls, count in train_counts.items():

    print(f"{cls}: {count}")

print("\nVAL")

for cls, count in val_counts.items():

    print(f"{cls}: {count}")

print("\nTEST")

for cls, count in test_counts.items():

    print(f"{cls}: {count}")

print("\n🎉 MESSIDOR PROCESSING COMPLETED!")
