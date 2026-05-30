"""
Classification models for RetinaRisk AI.

Shared class order:
0 = at_risk, 1 = disease_detected, 2 = normal
"""

import warnings

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    efficientnet_b3,
    resnet50,
    vit_b_16,
)

NUM_CLASSES = 3


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """A CPU-friendly CNN compatible with existing custom_cnn checkpoints."""

    def __init__(self, img_size=224, dropout=0.4, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        feat_dim = 512 * (img_size // 32) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResNetClassifier(nn.Module):
    def __init__(
        self,
        dropout=0.4,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        use_pretrained=True,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"ResNet pretrained weights unavailable ({exc}); using random init.",
                RuntimeWarning,
            )
            backbone = resnet50(weights=None)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        else:
            for name, p in backbone.named_parameters():
                p.requires_grad = any(x in name for x in ["layer3", "layer4", "fc"])

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        dropout=0.4,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        use_pretrained=True,
    ):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            backbone = efficientnet_b3(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"EfficientNet pretrained weights unavailable ({exc}); using random init.",
                RuntimeWarning,
            )
            backbone = efficientnet_b3(weights=None)

        if freeze_backbone:
            for p in backbone.features.parameters():
                p.requires_grad = False
        else:
            for block in list(backbone.features.children())[:-3]:
                for p in block.parameters():
                    p.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


class ViTClassifier(nn.Module):
    def __init__(
        self,
        dropout=0.4,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        use_pretrained=True,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            backbone = vit_b_16(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"ViT pretrained weights unavailable ({exc}); using random init.",
                RuntimeWarning,
            )
            backbone = vit_b_16(weights=None)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        else:
            fine_tune_layers = [
                f"encoder.layers.encoder_layer_{i}" for i in range(8, 12)
            ] + ["heads"]
            for name, p in backbone.named_parameters():
                p.requires_grad = any(layer in name for layer in fine_tune_layers)

        in_features = backbone.heads.head.in_features
        backbone.heads = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


class EnsembleClassifier(nn.Module):
    def __init__(self, resnet, efficientnet, vit, dropout=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        self.resnet = resnet
        self.efficientnet = efficientnet
        self.vit = vit

        for m in [self.resnet, self.efficientnet, self.vit]:
            for p in m.parameters():
                p.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            r = self.resnet(x)
            e = self.efficientnet(x)
            v = self.vit(x)
        combined = torch.cat([r, e, v], dim=1)
        return self.fusion(combined)


def build_model(
    name: str,
    img_size=224,
    dropout=0.4,
    num_classes=NUM_CLASSES,
    freeze_backbone=False,
    use_pretrained=True,
):
    name = str(name).lower().strip()
    if name in {"cnn", "custom_cnn", "customcnn"}:
        return CustomCNN(img_size=img_size, dropout=dropout, num_classes=num_classes)
    if name == "resnet":
        return ResNetClassifier(
            dropout=dropout,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            use_pretrained=use_pretrained,
        )
    if name == "efficientnet":
        return EfficientNetClassifier(
            dropout=dropout,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            use_pretrained=use_pretrained,
        )
    if name == "vit":
        return ViTClassifier(
            dropout=dropout,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            use_pretrained=use_pretrained,
        )
    if name == "ensemble":
        r = ResNetClassifier(dropout=dropout, num_classes=num_classes, use_pretrained=use_pretrained)
        e = EfficientNetClassifier(dropout=dropout, num_classes=num_classes, use_pretrained=use_pretrained)
        v = ViTClassifier(dropout=dropout, num_classes=num_classes, use_pretrained=use_pretrained)
        return EnsembleClassifier(r, e, v, dropout=dropout, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / max(total, 1)
    print(f"  Params - Total: {total:,} | Trainable: {trainable:,} ({pct:.1f}%)")
