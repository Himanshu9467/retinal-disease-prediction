"""
FINAL RETINAL DATASET MERGER
--------------------------------

Merges:

1. prepared_dataset
2. processed_dataset
3. processed_idrid_dataset
4. processed_messidor_dataset

into:

final_dataset/

Expected structure:

final_dataset/

├── train/
│   ├── normal/
│   ├── at_risk/
│   └── disease_detected/
│
├── val/
│   ├── normal/
│   ├── at_risk/
│   └── disease_detected/
│
└── test/
    ├── normal/
    ├── at_risk/
    └── disease_detected/

SAFE FEATURES:
✅ no overwriting
✅ keeps split separation
✅ keeps class separation
✅ handles duplicate filenames
✅ supports jpg/png/jpeg
"""

import os
import shutil
from tqdm import tqdm

# =========================================================
# BASE PATH
# =========================================================

BASE_PATH = r"C:\Retinal-image-prediction"

# =========================================================
# SOURCE DATASETS
# =========================================================

SOURCE_DATASETS = [

    "prepared_dataset",

    "processed_dataset",

    "processed_idrid_dataset",

    "processed_messidor_dataset"
]

# =========================================================
# OUTPUT DATASET
# =========================================================

FINAL_DATASET = os.path.join(
    BASE_PATH,
    "final_dataset"
)

# =========================================================
# SPLITS + CLASSES
# =========================================================

SPLITS = [

    "train",

    "val",

    "test"
]

CLASSES = [

    "normal",

    "at_risk",

    "disease_detected"
]

# =========================================================
# CREATE FINAL STRUCTURE
# =========================================================

print("\n" + "=" * 70)
print("CREATING FINAL DATASET STRUCTURE")
print("=" * 70)

for split in SPLITS:

    for cls in CLASSES:

        path = os.path.join(
            FINAL_DATASET,
            split,
            cls
        )

        os.makedirs(
            path,
            exist_ok=True
        )

# =========================================================
# COUNTERS
# =========================================================

stats = {

    "train": {

        "normal": 0,

        "at_risk": 0,

        "disease_detected": 0
    },

    "val": {

        "normal": 0,

        "at_risk": 0,

        "disease_detected": 0
    },

    "test": {

        "normal": 0,

        "at_risk": 0,

        "disease_detected": 0
    }
}

# =========================================================
# PROCESS EACH DATASET
# =========================================================

for dataset_name in SOURCE_DATASETS:

    print("\n" + "=" * 70)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 70)

    dataset_path = os.path.join(
        BASE_PATH,
        dataset_name
    )

    # -----------------------------------------------------
    # VERIFY DATASET EXISTS
    # -----------------------------------------------------

    if not os.path.exists(dataset_path):

        print(f"❌ Missing dataset: {dataset_name}")

        continue

    # -----------------------------------------------------
    # PROCESS SPLITS
    # -----------------------------------------------------

    for split in SPLITS:

        split_path = os.path.join(
            dataset_path,
            split
        )

        if not os.path.exists(split_path):

            print(
                f"⚠ Missing split: "
                f"{dataset_name}/{split}"
            )

            continue

        # -------------------------------------------------
        # PROCESS CLASSES
        # -------------------------------------------------

        for cls in CLASSES:

            class_path = os.path.join(
                split_path,
                cls
            )

            if not os.path.exists(class_path):

                print(
                    f"⚠ Missing class: "
                    f"{dataset_name}/{split}/{cls}"
                )

                continue

            # ---------------------------------------------
            # GET FILES
            # ---------------------------------------------

            files = [

                f for f in os.listdir(class_path)

                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            print(
                f"{split}/{cls}: "
                f"{len(files)} images"
            )

            # ---------------------------------------------
            # COPY FILES
            # ---------------------------------------------

            for file in tqdm(files):

                src = os.path.join(
                    class_path,
                    file
                )

                # -----------------------------------------
                # UNIQUE NAME
                # -----------------------------------------

                new_name = (
                    f"{dataset_name}_{file}"
                )

                dst = os.path.join(

                    FINAL_DATASET,

                    split,

                    cls,

                    new_name
                )

                # -----------------------------------------
                # HANDLE DUPLICATES
                # -----------------------------------------

                counter = 1

                base, ext = os.path.splitext(
                    new_name
                )

                while os.path.exists(dst):

                    dst = os.path.join(

                        FINAL_DATASET,

                        split,

                        cls,

                        f"{base}_{counter}{ext}"
                    )

                    counter += 1

                # -----------------------------------------
                # COPY
                # -----------------------------------------

                try:

                    shutil.copy2(src, dst)

                    stats[split][cls] += 1

                except Exception:

                    continue

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

for split in SPLITS:

    print(f"\n{split.upper()}")

    total = 0

    for cls in CLASSES:

        count = stats[split][cls]

        total += count

        print(f"{cls}: {count}")

    print(f"TOTAL: {total}")

# =========================================================
# EXPECTED OUTPUT INFO
# =========================================================

print("\n" + "=" * 70)
print("EXPECTED FINAL STRUCTURE")
print("=" * 70)

print("""

final_dataset/

├── train/
│   ├── normal/
│   ├── at_risk/
│   └── disease_detected/

├── val/
│   ├── normal/
│   ├── at_risk/
│   └── disease_detected/

└── test/
    ├── normal/
    ├── at_risk/
    └── disease_detected/

""")

print("🎉 FINAL DATASET MERGING COMPLETED!")