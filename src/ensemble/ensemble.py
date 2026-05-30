"""Weighted inference ensemble for RetinaRisk AI."""

import base64
import io
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM, ViTAttentionRollout, get_target_layer, overlay_cam
from model_registry import (
    build_model_registry,
    registry_as_dict,
    validate_registry_paths,
)
from production_models import build_final_model

LOGGER = logging.getLogger(__name__)

CLASS_NAMES = ["At-Risk", "Disease Detected", "Normal"]
CLASS_KEYS = ["at_risk", "disease_detected", "normal"]
DEFAULT_WEIGHTS = {
    "efficientnet": 1.4,
    "resnet": 1.4,
    "vit": 1.1,
    "cnn": 0.0,
}
TEMPERATURE = {
    "efficientnet": 1.3,
    "resnet": 1.4,
    "vit": 1.5,
    "cnn": 1.0,
}
MODEL_TEMPERATURES = TEMPERATURE
ACTIVE_ENSEMBLE_MODELS = {"efficientnet", "resnet", "vit"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class RetinalRiskEnsemble:
    def __init__(
        self,
        img_size=224,
        model_weights: Optional[Dict[str, float]] = None,
        device=None,
        output_dir=None,
        enable_explainability=True,
        registry=None,
        strict_registry=True,
    ):
        self.img_size = img_size
        self.device = torch.device(device or "cpu")
        self.enable_explainability = enable_explainability
        self.output_dir = Path(output_dir) if output_dir else _project_root() / "outputs"
        self.registry = registry or build_model_registry(strict=strict_registry)
        issues = validate_registry_paths(self.registry)
        if issues:
            raise RuntimeError("Invalid production model registry: " + "; ".join(issues))
        registry_weights = {name: entry.weight for name, entry in self.registry.items()}
        self.model_weights = self._normalize_weights(model_weights or registry_weights or DEFAULT_WEIGHTS)
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.models = self._load_models()

    def _normalize_weights(self, weights):
        canonical = {}
        for key, value in weights.items():
            name = "cnn" if key in {"custom_cnn", "customcnn"} else key
            if name in ACTIVE_ENSEMBLE_MODELS or name == "cnn":
                canonical[name] = max(float(value), 0.0)
        canonical["cnn"] = 0.0
        total = sum(v for name, v in canonical.items() if name in ACTIVE_ENSEMBLE_MODELS and v > 0)
        if total <= 0:
            canonical = dict(DEFAULT_WEIGHTS)
            total = sum(canonical[name] for name in ACTIVE_ENSEMBLE_MODELS)
        return {
            k: (v / total if k in ACTIVE_ENSEMBLE_MODELS and v > 0 else 0.0)
            for k, v in canonical.items()
        }

    def _load_state_dict(self, model, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {checkpoint_path}: "
                f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
            )

    def _load_model(self, model_name, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint_path}")
        model = build_final_model(model_name, num_classes=len(CLASS_NAMES))
        self._load_state_dict(model, checkpoint_path)
        model.to(self.device)
        model.eval()
        return model

    def _load_models(self):
        loaded = {}
        ordered_names = sorted(
            [
                name
                for name, weight in self.model_weights.items()
                if weight > 0 and name in ACTIVE_ENSEMBLE_MODELS and name in self.registry
            ],
            key=lambda name: self.registry[name].priority,
        )
        for model_name in ordered_names:
            entry = self.registry.get(model_name)
            if not entry:
                LOGGER.warning("No registry entry found for %s", model_name)
                continue
            if not entry.checkpoint:
                LOGGER.warning("No checkpoint configured for %s; skipping", model_name)
                continue
            checkpoint = Path(entry.checkpoint)
            try:
                loaded[model_name] = self._load_model(model_name, checkpoint)
                LOGGER.info(
                    "Loaded %s %s from %s with ensemble weight %.4f",
                    model_name,
                    entry.version,
                    checkpoint,
                    self.model_weights.get(model_name, 0.0),
                )
            except Exception as exc:
                LOGGER.exception("Failed to load %s from %s: %s", model_name, checkpoint, exc)
                raise RuntimeError(f"Registered production model failed to load: {model_name}") from exc
        missing_registered = [name for name in ordered_names if name not in loaded]
        if missing_registered:
            raise RuntimeError(f"Registered production models were not loaded: {missing_registered}")
        return loaded

    def preprocess_image(self, image):
        try:
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                raise TypeError("image must be a file path or PIL image")
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image input: {exc}") from exc
        return self.transform(pil_image).unsqueeze(0), pil_image

    def preprocess_with_clahe(self, image):
        try:
            if isinstance(image, (str, Path)):
                rgb = np.array(Image.open(image).convert("RGB"))
            elif isinstance(image, Image.Image):
                rgb = np.array(image.convert("RGB"))
            else:
                raise TypeError("image must be a file path or PIL image")
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image input: {exc}") from exc
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        pil_image = Image.fromarray(enhanced_rgb)
        return self.transform(pil_image).unsqueeze(0), pil_image

    def _risk_band(self, prediction, confidence):
        if prediction == "Disease Detected":
            return "high"
        if prediction == "At-Risk" or confidence < 0.60:
            return "medium"
        return "low"

    def _agreement(self, model_outputs):
        if not model_outputs:
            return "Low", 0.0
        votes = {}
        for row in model_outputs.values():
            votes[row["prediction"]] = votes.get(row["prediction"], 0) + 1
        score = max(votes.values()) / len(model_outputs)
        if score >= 0.75:
            return "High", score
        if score >= 0.5:
            return "Medium", score
        return "Low", score

    def _gradcam(self, model_name, model, tensor, pil_image, target_class):
        if not self.enable_explainability:
            return None
        try:
            if model_name == "vit":
                rollout = ViTAttentionRollout(model)
                heatmap = rollout.generate(tensor.clone().to(self.device))
                rollout.remove_hooks()
            else:
                target_layer = get_target_layer(model, model_name)
                cam = GradCAM(model, target_layer)
                heatmap, _ = cam.generate(tensor.clone().to(self.device), target_class=target_class)
                cam.remove_hooks()
            return _to_data_url(overlay_cam(pil_image, heatmap))
        except Exception as exc:
            LOGGER.warning("GradCAM unavailable for %s: %s", model_name, exc)
            return None

    @torch.no_grad()
    def predict_tensor(self, image_tensor):
        if not self.models:
            raise RuntimeError("No model checkpoints are loaded for ensemble inference.")

        image_tensor = image_tensor.to(self.device)
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise RuntimeError(f"Invalid image tensor shape: {tuple(image_tensor.shape)}")
        if not torch.isfinite(image_tensor).all():
            raise RuntimeError("Invalid image tensor contains NaN or Inf values")

        weighted_sum = torch.zeros((1, len(CLASS_NAMES)), device=self.device)
        used_weight = 0.0
        individual = {}

        for name, model in self.models.items():
            weight = float(self.model_weights.get(name, 0.0))
            if weight <= 0:
                continue

            logits = model(image_tensor)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite logits produced by {name}")

            temperature = MODEL_TEMPERATURES.get(name, 1.0)
            if temperature <= 0:
                raise RuntimeError(f"Invalid calibration temperature for {name}: {temperature}")

            calibrated_logits = logits.float() / float(temperature)

            probs = F.softmax(calibrated_logits, dim=1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            if not torch.isfinite(probs).all():
                raise RuntimeError(f"Non-finite probabilities produced by {name}")

            weighted_sum += probs * weight

            used_weight += weight

            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs.max().item())
            individual[name] = {
                "model_name": name,
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 2),
                "weight": weight,
                "probabilities": {
                    CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(len(CLASS_NAMES))
                },
            }

        if used_weight <= 0:
            raise RuntimeError("Loaded models have zero ensemble weight.")

        final_probs = weighted_sum / used_weight
        final_probs = torch.nan_to_num(final_probs, nan=0.0, posinf=1.0, neginf=0.0)
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if not torch.isfinite(final_probs).all():
            raise RuntimeError("Non-finite ensemble probabilities produced")
        final_idx = int(final_probs.argmax(dim=1).item())
        final_confidence = float(final_probs.max().item())
        return final_idx, final_confidence, final_probs, individual

    def predict(self, image):
        tensor, pil_image = self.preprocess_image(image)
        return self._predict_from_tensor(tensor, pil_image)

    def _predict_from_tensor(self, tensor, pil_image):
        final_idx, final_confidence, final_probs, individual = self.predict_tensor(tensor)
        explanation = {}
        for name, model in self.models.items():
            data_url = self._gradcam(name, model, tensor, pil_image, final_idx)
            if data_url:
                explanation[f"{name}_gradcam"] = data_url

        final_prediction = CLASS_NAMES[final_idx]
        agreement, agreement_score = self._agreement(individual)
        probabilities = {
            CLASS_NAMES[i]: float(final_probs[0, i].item()) for i in range(len(CLASS_NAMES))
        }

        return {
            "prediction": final_prediction,
            "confidence": round(final_confidence * 100, 2),
            "confidence_score": final_confidence,
            "final_prediction": final_prediction,
            "final_confidence": round(final_confidence * 100, 2),
            "models": individual,
            "individual_models": individual,
            "ensemble_prediction": final_prediction,
            "ensemble_confidence": final_confidence,
            "ensemble_probabilities": probabilities,
            "individual_model_predictions": individual,
            "class_probabilities": probabilities,
            "probability_risk_band": self._risk_band(final_prediction, final_confidence),
            "risk_score": 1.0 - probabilities.get("Normal", 0.0),
            "model_agreement": agreement,
            "model_agreement_score": agreement_score,
            "loaded_models": list(self.models.keys()),
            "excluded_models": [
                name for name, weight in self.model_weights.items() if weight <= 0 and name in self.registry
            ],
            "model_registry": registry_as_dict(self.registry),
            "ensemble_weights": self.model_weights,
            "explainability": explanation,
            "target": CLASS_KEYS[final_idx],
            "target_description": "Retinal image severity prediction from the trained dataset classes.",
            "note": "This screening output supports clinical triage and is not a standalone diagnosis.",
            "bp_category": None,
            "cholesterol": None,
            "vessel_features": {},
            "retinal_severity": final_prediction,
            "cvd_risk": final_prediction,
        }

    def ensemble_predict(self, image, use_clahe=False):
        tensor, pil_image = self.preprocess_with_clahe(image) if use_clahe else self.preprocess_image(image)
        return self._predict_from_tensor(tensor, pil_image)
