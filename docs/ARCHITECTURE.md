# System Architecture

## Overview

RetinaRisk follows a straightforward inference architecture:

```text
Frontend
  -> FastAPI Backend
    -> Ensemble Engine
      -> ResNet34 + EfficientNet-B3 + ViT-B16
        -> Temperature Calibration
        -> Weighted Probability Fusion
      -> GradCAM / Attention Explainability
    -> Prediction Response
```

## Components

### Frontend

The React frontend provides authentication, patient management, retinal image upload, prediction rendering, confidence display, GradCAM display through secured media endpoints, prediction history, and analytics views.

### FastAPI Backend

The backend validates uploads, checks patient context, loads the model registry from `outputs/`, serves health and model metadata, runs inference, stores predictions, and exposes generated GradCAM images.

### Ensemble Engine

The ensemble engine loads active checkpoints from the production registry:

- EfficientNet-B3
- ResNet34
- ViT-B16

The Custom CNN remains visible in the registry but has zero ensemble weight.

### Calibration and Probability Flow

Each model produces logits. The logits are divided by a model-specific temperature, passed through softmax, sanitized for non-finite values, normalized, and weighted. The final ensemble probability vector is normalized before prediction.

### GradCAM

GradCAM is generated for compatible CNN-style models. ViT attention rollout is attempted where attention maps are available. The prediction response includes explainability data and the backend stores a retrievable image for the selected prediction.

## Data Flow

1. User selects a patient and uploads a retinal image.
2. Frontend sends `multipart/form-data` to `/predict` with the bearer token.
3. Backend validates image type, size, and basic image integrity.
4. Backend stores the uploaded image in `uploads/`.
5. Ensemble engine preprocesses the image and runs each active model.
6. Calibrated probabilities are combined by normalized weights.
7. Backend saves the prediction and explanation path in SQLite.
8. Frontend renders prediction, confidence, class probabilities, model agreement, and GradCAM.

## Production Artifact Source

The source of truth for inference is the current `outputs/` directory. The registry validates checkpoint paths and refuses missing active production models.
