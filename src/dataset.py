"""
ImageFolder dataloading for RetinaRisk AI.

Expected class mapping:
0 = at_risk, 1 = disease_detected, 2 = normal
"""

import os
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASS_NAMES = ["at_risk", "disease_detected", "normal"]
DISPLAY_CLASS_NAMES = ["At-Risk", "Disease Detected", "Normal"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RetinalDataset(Dataset):
    """CSV-backed retinal dataset used by the standalone evaluation script."""

    IMAGE_COLUMNS = ("image_path", "path", "filename", "file", "img_name", "image", "id_code")
    LABEL_COLUMNS = ("class", "label", "diagnosis", "level")

    def __init__(self, csv_path, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.data = pd.read_csv(csv_path).dropna(how="all").reset_index(drop=True)
        self.image_column = self._first_existing_column(self.IMAGE_COLUMNS)
        self.label_column = self._first_existing_column(self.LABEL_COLUMNS)

    def _first_existing_column(self, candidates):
        for column in candidates:
            if column in self.data.columns:
                return column
        raise ValueError(
            f"CSV must contain one of these columns: {', '.join(candidates)}"
        )

    def _label_to_index(self, value):
        if pd.isna(value):
            raise ValueError("Missing label in evaluation CSV")

        label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if label in CLASS_TO_IDX:
            return CLASS_TO_IDX[label]

        numeric = int(float(value))
        if numeric == 0:
            return CLASS_TO_IDX["normal"]
        if numeric in {1, 2}:
            return CLASS_TO_IDX["at_risk"]
        return CLASS_TO_IDX["disease_detected"]

    def _image_path(self, value):
        raw = str(value).strip()
        path = Path(raw)
        if not path.suffix:
            path = path.with_suffix(".jpg")
        if not path.is_absolute():
            path = self.image_dir / path
        return path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self._image_path(row[self.image_column])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self._label_to_index(row[self.label_column])
        return image, label, str(image_path)


def get_train_transforms(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize((img_size + 24, img_size + 24)),
            transforms.RandomResizedCrop(img_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.08),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.18, contrast=0.22, saturation=0.12, hue=0.02),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.20),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ]
    )


def get_val_transforms(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _validate_split(root: Path, split: str) -> Path:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")
    missing = [class_name for class_name in CLASS_NAMES if not (split_dir / class_name).is_dir()]
    if missing:
        raise FileNotFoundError(f"{split_dir} is missing class folder(s): {', '.join(missing)}")
    return split_dir


def _image_folder(split_dir: Path, transform):
    dataset = datasets.ImageFolder(str(split_dir), transform=transform)
    if dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Unexpected class mapping for {split_dir}: {dataset.class_to_idx}. "
            f"Expected {CLASS_TO_IDX}."
        )
    return dataset


def _loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def build_dataloaders(
    dataset_root="balanced_dataset",
    img_size=224,
    batch_size=16,
    num_workers=None,
):
    root = Path(dataset_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    if num_workers is None:
        num_workers = min(2, max(0, (os.cpu_count() or 2) - 1))

    train_dataset = _image_folder(_validate_split(root, "train"), get_train_transforms(img_size))
    val_dataset = _image_folder(_validate_split(root, "val"), get_val_transforms(img_size))
    test_dataset = _image_folder(_validate_split(root, "test"), get_val_transforms(img_size))

    loaders = {
        "train": _loader(train_dataset, batch_size, True, num_workers),
        "val": _loader(val_dataset, batch_size, False, num_workers),
        "test": _loader(test_dataset, batch_size, False, num_workers),
        "class_to_idx": train_dataset.class_to_idx,
        "classes": CLASS_NAMES,
    }

    print("\nDataset loaded")
    print(f"Root: {root}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    return loaders
