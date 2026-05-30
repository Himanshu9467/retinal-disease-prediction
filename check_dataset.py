import os

base = "research_patient_safe_aug_dataset"

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    split_path = os.path.join(base, split)

    for cls in os.listdir(split_path):

        cls_path = os.path.join(split_path, cls)

        count = len(os.listdir(cls_path))

        print(f"{cls}: {count}")