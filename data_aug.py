"""
MINOR RESEARCH-LEVEL AUGMENTATION
=================================

GOAL:
✅ ONLY augment disease_detected
✅ Small controlled augmentation
✅ Research-safe retinal augmentation
✅ No fake balancing
✅ No unrealistic transformations

INPUT:
research_dataset/

OUTPUT:
research_dataset_augmented/

FINAL TARGET:
disease_detected:
~5300 → ~7500

SAFE AUGMENTATIONS:
✅ mild rotation
✅ slight brightness
✅ slight contrast
✅ slight zoom
✅ horizontal flip

NO:
❌ vertical flip
❌ elastic transform
❌ strong saturation
❌ extreme rotation

"""

import os
import cv2
import shutil
import random
import numpy as np

from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================

BASE_PATH = r"C:\Retinal-image-prediction"

INPUT_DATASET = os.path.join(
    BASE_PATH,
    "research_dataset"
)

OUTPUT_DATASET = os.path.join(
    BASE_PATH,
    "research_aug_dataset"
)

# =========================================================
# SETTINGS
# =========================================================

TARGET_DISEASE_COUNT = 7500

IMG_EXTENSIONS = (

    ".jpg",

    ".jpeg",

    ".png"
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
# CREATE OUTPUT STRUCTURE
# =========================================================

print("\n" + "=" * 70)
print("CREATING AUGMENTED DATASET")
print("=" * 70)

for split in SPLITS:

    for cls in CLASSES:

        os.makedirs(

            os.path.join(
                OUTPUT_DATASET,
                split,
                cls
            ),

            exist_ok=True
        )

# =========================================================
# COPY NORMAL + AT_RISK DIRECTLY
# =========================================================

print("\n" + "=" * 70)
print("COPYING EXISTING DATA")
print("=" * 70)

for split in SPLITS:

    for cls in CLASSES:

        src_dir = os.path.join(
            INPUT_DATASET,
            split,
            cls
        )

        dst_dir = os.path.join(
            OUTPUT_DATASET,
            split,
            cls
        )

        files = [

            f for f in os.listdir(src_dir)

            if f.lower().endswith(
                IMG_EXTENSIONS
            )
        ]

        for file in tqdm(files):

            shutil.copy2(

                os.path.join(src_dir, file),

                os.path.join(dst_dir, file)
            )

# =========================================================
# AUGMENTATION FUNCTIONS
# =========================================================

def random_horizontal_flip(img):

    if random.random() < 0.5:

        img = cv2.flip(img, 1)

    return img

# ---------------------------------------------------------

def random_rotation(img):

    angle = random.uniform(-5, 5)

    h, w = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(

        (w // 2, h // 2),

        angle,

        1.0
    )

    rotated = cv2.warpAffine(

        img,

        matrix,

        (w, h),

        borderMode=cv2.BORDER_REFLECT
    )

    return rotated

# ---------------------------------------------------------

def random_brightness_contrast(img):

    alpha = random.uniform(0.95, 1.05)

    beta = random.randint(-8, 8)

    adjusted = cv2.convertScaleAbs(

        img,

        alpha=alpha,

        beta=beta
    )

    return adjusted

# ---------------------------------------------------------

def random_zoom(img):

    scale = random.uniform(0.95, 1.05)

    h, w = img.shape[:2]

    resized = cv2.resize(

        img,

        None,

        fx=scale,

        fy=scale
    )

    rh, rw = resized.shape[:2]

    # crop center
    if scale > 1.0:

        start_x = (rw - w) // 2

        start_y = (rh - h) // 2

        cropped = resized[
            start_y:start_y+h,
            start_x:start_x+w
        ]

        return cropped

    # pad center
    else:

        canvas = np.zeros_like(img)

        start_x = (w - rw) // 2

        start_y = (h - rh) // 2

        canvas[
            start_y:start_y+rh,
            start_x:start_x+rw
        ] = resized

        return canvas

# ---------------------------------------------------------

def augment_image(img):

    img = random_horizontal_flip(img)

    img = random_rotation(img)

    img = random_brightness_contrast(img)

    img = random_zoom(img)

    return img

# =========================================================
# AUGMENT ONLY TRAIN/disease_detected
# =========================================================

print("\n" + "=" * 70)
print("AUGMENTING disease_detected")
print("=" * 70)

train_disease_dir = os.path.join(

    OUTPUT_DATASET,

    "train",

    "disease_detected"
)

files = [

    f for f in os.listdir(train_disease_dir)

    if f.lower().endswith(
        IMG_EXTENSIONS
    )
]

current_count = len(files)

print(f"\nCurrent Count : {current_count}")

needed = TARGET_DISEASE_COUNT - current_count

print(f"Target Count  : {TARGET_DISEASE_COUNT}")

print(f"Need To Add   : {needed}")

if needed > 0:

    for i in tqdm(range(needed)):

        file = random.choice(files)

        src_path = os.path.join(
            train_disease_dir,
            file
        )

        img = cv2.imread(src_path)

        if img is None:

            continue

        aug = augment_image(img)

        name, ext = os.path.splitext(file)

        save_name = f"{name}_aug_{i}{ext}"

        save_path = os.path.join(

            train_disease_dir,

            save_name
        )

        cv2.imwrite(
            save_path,
            aug
        )

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL AUGMENTATION COMPLETE")
print("=" * 70)

final_files = [

    f for f in os.listdir(train_disease_dir)

    if f.lower().endswith(
        IMG_EXTENSIONS
    )
]

print(f"\nFinal disease_detected Count : {len(final_files)}")

print("\n🎉 RESEARCH-SAFE AUGMENTATION COMPLETED!")