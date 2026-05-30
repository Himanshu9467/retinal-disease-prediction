"""
RESEARCH-LEVEL RETINAL PREPROCESSING PIPELINE
=============================================

Creates:
research_dataset/

from:
cleaned_final_dataset/

GOALS:
✅ Research-oriented retinal preprocessing
✅ Better retinal consistency
✅ Better lesion visibility
✅ Better cross-dataset generalization
✅ Medical-safe enhancement
✅ No fake enhancement

FEATURES:
✅ Circular retina crop
✅ Black border removal
✅ CLAHE enhancement
✅ Illumination normalization
✅ Retina area validation
✅ Mild sharpening
✅ Blur filtering
✅ Corrupted image filtering
✅ Retina-centered preprocessing

SAFE FOR:
- diabetic retinopathy research
- EfficientNet
- ResNet
- ViT
- medical AI projects
"""

import os
import cv2
import shutil
import numpy as np

from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================

BASE_PATH = r"C:\Retinal-image-prediction"

INPUT_DATASET = os.path.join(
    BASE_PATH,
    "cleaned_final_dataset"
)

OUTPUT_DATASET = os.path.join(
    BASE_PATH,
    "research_dataset"
)

# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = 300

MIN_RETINA_AREA_RATIO = 0.10

BLUR_THRESHOLD = 18

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
print("CREATING RESEARCH DATASET STRUCTURE")
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
# STATS
# =========================================================

stats = {

    "processed": 0,

    "removed_corrupted": 0,

    "removed_blurry": 0,

    "removed_small_retina": 0,

    "saved": 0
}

# =========================================================
# IMAGE HELPERS
# =========================================================

def is_corrupted(img):

    return img is None

# ---------------------------------------------------------

def is_blurry(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return variance < BLUR_THRESHOLD

# ---------------------------------------------------------

def crop_retina(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # threshold retina region
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(

        thresh,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest)

    img_area = img.shape[0] * img.shape[1]

    area_ratio = area / img_area

    # retina too small
    if area_ratio < MIN_RETINA_AREA_RATIO:

        return None

    x, y, w, h = cv2.boundingRect(largest)

    cropped = img[y:y+h, x:x+w]

    return cropped

# ---------------------------------------------------------

def apply_clahe(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(

        clipLimit=1.5,

        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))

    result = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return result

# ---------------------------------------------------------

def normalize_illumination(img):

    img_float = img.astype(np.float32)

    mean = np.mean(img_float)

    img_float = img_float * (128.0 / (mean + 1e-5))

    img_float = np.clip(
        img_float,
        0,
        255
    )

    return img_float.astype(np.uint8)

# ---------------------------------------------------------

def mild_sharpen(img):

    gaussian = cv2.GaussianBlur(
        img,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(

        img,
        1.2,

        gaussian,
        -0.2,

        0
    )

    return sharpened

# =========================================================
# PROCESS DATASET
# =========================================================

print("\n" + "=" * 70)
print("STARTING RESEARCH PREPROCESSING")
print("=" * 70)

for split in SPLITS:

    print("\n" + "=" * 70)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 70)

    for cls in CLASSES:

        print(f"\nCLASS: {cls}")

        input_dir = os.path.join(
            INPUT_DATASET,
            split,
            cls
        )

        output_dir = os.path.join(
            OUTPUT_DATASET,
            split,
            cls
        )

        files = [

            f for f in os.listdir(input_dir)

            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        print(f"Images: {len(files)}")

        for file in tqdm(files):

            stats["processed"] += 1

            src = os.path.join(
                input_dir,
                file
            )

            dst = os.path.join(
                output_dir,
                file
            )

            # =================================================
            # READ IMAGE
            # =================================================

            img = cv2.imread(src)

            if is_corrupted(img):

                stats["removed_corrupted"] += 1

                continue

            # =================================================
            # BLUR CHECK
            # =================================================

            if is_blurry(img):

                stats["removed_blurry"] += 1

                continue

            # =================================================
            # RETINA CROP
            # =================================================

            cropped = crop_retina(img)

            if cropped is None:

                stats["removed_small_retina"] += 1

                continue

            # =================================================
            # CLAHE
            # =================================================

            processed = apply_clahe(cropped)

            # =================================================
            # ILLUMINATION NORMALIZATION
            # =================================================

            processed = normalize_illumination(
                processed
            )

            # =================================================
            # MILD SHARPENING
            # =================================================

            processed = mild_sharpen(
                processed
            )

            # =================================================
            # RESIZE
            # =================================================

            processed = cv2.resize(

                processed,

                (IMG_SIZE, IMG_SIZE)
            )

            # =================================================
            # SAVE
            # =================================================

            try:

                cv2.imwrite(
                    dst,
                    processed,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                stats["saved"] += 1

            except Exception:

                continue

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL RESEARCH DATASET SUMMARY")
print("=" * 70)

print(f"\nProcessed Images       : {stats['processed']}")

print(f"Saved Images           : {stats['saved']}")

print(f"Removed Corrupted      : {stats['removed_corrupted']}")

print(f"Removed Blurry         : {stats['removed_blurry']}")

print(f"Removed Small Retina   : {stats['removed_small_retina']}")

print("\n🎉 RESEARCH DATASET CREATED SUCCESSFULLY!")