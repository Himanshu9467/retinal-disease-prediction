import os
import random
import shutil

base_path = "cleaned_dataset"

output_base = "balanced_dataset"

limits = {
    "train": 1555,
    "val": 250,
    "test": 250
}

classes = [
    "normal",
    "at_risk",
    "disease_detected"
]

valid_ext = (
    ".jpg",
    ".jpeg",
    ".png"
)

for split in ["train", "val", "test"]:

    print(f"\nPROCESSING {split.upper()}")

    for cls in classes:

        src_folder = os.path.join(
            base_path,
            split,
            cls
        )

        dst_folder = os.path.join(
            output_base,
            split,
            cls
        )

        os.makedirs(
            dst_folder,
            exist_ok=True
        )

        files = [

            f for f in os.listdir(src_folder)

            if f.lower().endswith(valid_ext)
        ]

        random.shuffle(files)

        limit = min(
            limits[split],
            len(files)
        )

        selected = files[:limit]

        for file in selected:

            src = os.path.join(
                src_folder,
                file
            )

            dst = os.path.join(
                dst_folder,
                file
            )

            shutil.copy2(src, dst)

        print(
            f"{cls}: {len(selected)} images copied"
        )

print("\n✅ Rebalancing Complete!")