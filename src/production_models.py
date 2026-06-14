"""Production model definitions for research_training_outputs2 binary checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.skip(x))


class ProductionCustomCNN(nn.Module):
    def __init__(self, dropout: float = 0.4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 64, stride=1, dropout=0.05),
            ResidualBlock(64, 128, stride=2, dropout=0.10),
            ResidualBlock(128, 128, stride=1, dropout=0.10),
            ResidualBlock(128, 256, stride=2, dropout=0.15),
            ResidualBlock(256, 256, stride=1, dropout=0.15),
            ResidualBlock(256, 384, stride=2, dropout=0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(384),
            nn.Dropout(p=dropout),
            nn.Linear(384, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_final_model(model_name: str, num_classes: int = 1) -> nn.Module:
    del num_classes
    name = str(model_name).strip().lower()

    if name == "efficientnet":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(in_features, 1))
        return model

    if name == "resnet":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, 1))
        return model

    if name == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        if isinstance(model.classifier[2], nn.Dropout):
            model.classifier[2] = nn.Dropout(p=0.4, inplace=True)
        return model

    if name in {"cnn", "custom_cnn", "customcnn"}:
        return ProductionCustomCNN(dropout=0.4)

    raise ValueError(f"Unknown production model: {model_name}")
