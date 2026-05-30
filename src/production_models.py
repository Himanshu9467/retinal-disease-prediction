"""Production model definitions matching the outputs/ checkpoints."""

import torch.nn as nn
from torchvision import models


def build_final_model(model_name: str, num_classes: int = 3):
    """Build the exact architecture used by run_training.py final checkpoints."""
    name = str(model_name).strip().lower()

    if name == "efficientnet":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name == "resnet":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name == "vit":
        model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name in {"cnn", "custom_cnn", "customcnn"}:
        return FinalRetinalCNN(num_classes=num_classes)

    raise ValueError(f"Unknown production model: {model_name}")


class FinalRetinalCNN(nn.Module):
    """Custom CNN architecture kept visible for registry compatibility."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
