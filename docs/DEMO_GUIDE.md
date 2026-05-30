# Demo Preparation Guide

## Demo Checklist

- Confirm `outputs/efficientnet_best.pth`, `outputs/resnet_best.pth`, and `outputs/vit_best.pth` exist.
- Start backend with `python -m uvicorn api.app:app --host 127.0.0.1 --port 8001`.
- Start frontend from `frontend/` with `npm run dev`.
- Open `http://127.0.0.1:5173`.
- Register or log in as a Doctor user.
- Create a demo patient.
- Upload a retinal image from the patient-safe test dataset.
- Show prediction class, confidence, probability distribution, model agreement, and GradCAM.
- Open Analytics and show model registry metrics.
- Visit `/health` and `/model-info` if asked about backend readiness.

## Testing Checklist

- Backend imports without syntax errors.
- `/health` returns `ok`.
- `/model-info` lists active checkpoints and weights.
- Auth registration/login works.
- Patient creation works.
- Retinal image upload returns a prediction.
- Ensemble probabilities sum to `1.0`.
- GradCAM endpoint returns an image.
- Frontend build passes.
- Frontend lint passes.

## Viva Questions and Answers

**What is the main objective of the project?**  
To classify retinal fundus images into severity categories and provide explainable screening support through a full-stack application.

**Why use a patient-safe split?**  
It prevents images from the same patient appearing in multiple splits, reducing data leakage and making validation more credible.

**Why use an ensemble?**  
EfficientNet, ResNet, and ViT capture different visual patterns. Weighted ensembling improves robustness compared with relying on one model.

**What is temperature calibration?**  
It rescales model logits before softmax to reduce overconfident probability estimates.

**How is GradCAM useful?**  
It highlights image regions that contributed to the prediction, helping reviewers understand what the model attended to.

**Is this a diagnostic system?**  
No. It is a screening and triage prototype. Final diagnosis must come from qualified clinicians.

**Which model performed best?**  
ViT-B16 has the highest weighted F1 from the latest validation histories, while EfficientNet-B3 has the highest ROC-AUC.

**Why is the CNN excluded?**  
Its validation metrics were weaker than the three production models, so it is retained for transparency with zero ensemble weight.

## Presentation Talking Points

- Problem: scalable retinal screening support.
- Research care: patient-safe split and leakage audit.
- Modeling: three complementary architectures plus calibrated ensemble.
- Explainability: GradCAM for visual trust and review.
- Engineering: FastAPI backend, React dashboard, registry, health checks, and prediction history.
- Results: production ensemble members have validation weighted F1 around `0.817` to `0.820`.
- Limitations: not a clinical diagnosis, needs broader external validation, stronger deployment hardening, and formal clinical review before real use.
