# Final QA Audit Report

Date: 2026-05-30

## Backend Audit

| Area | Result | Evidence |
| ---- | ------ | -------- |
| Startup | Pass | TestClient startup loaded EfficientNet, ResNet, and ViT |
| Health endpoint | Pass | `GET /health` returned `200`, status `ok` |
| Model-info endpoint | Pass | `GET /model-info` returned registry and ensemble weights |
| Prediction endpoint | Pass | `POST /predict` returned `200`, prediction `At-Risk`, confidence `0.7446` |
| GradCAM endpoint | Pass | `GET /gradcam/14` returned `200`, content type `image/png` |

## Frontend Audit

| Area | Result | Evidence |
| ---- | ------ | -------- |
| Upload image workflow | Pass by code path and backend smoke | `ScreeningView` posts `FormData` to `/predict`; backend smoke validated upload |
| Prediction rendering | Pass by code review | UI renders outcome, probabilities, model agreement, and guidance |
| Confidence rendering | Pass by code review | Confidence is formatted as percent and displayed in screening/history/analytics |
| GradCAM rendering | Pass by code review and backend smoke | Frontend retrieves media/GradCAM paths; backend served `image/png` |
| Error handling | Pass | API wrapper handles unreachable backend, non-JSON errors, and HTTP failures |

## Ensemble Audit

| Area | Result | Evidence |
| ---- | ------ | -------- |
| Weights | Pass | EfficientNet `0.358974`, ResNet `0.358974`, ViT `0.282051`, CNN `0.0` |
| Calibration | Pass | Temperatures: EfficientNet `1.3`, ResNet `1.4`, ViT `1.5` |
| Probability normalization | Pass | Backend smoke returned ensemble probability sum `1.0` |
| Active models | Pass | Loaded models: `efficientnet`, `resnet`, `vit` |

## Bugs Found

- Local `venv` is inconsistent: `venv\\Scripts\\python.exe` cannot import FastAPI and its pip import is broken. System Python has the required backend packages and passed the smoke test.
- Production warnings correctly flag that `JWT_SECRET` should be set before deployment.
- Registry uses flat `outputs/<model>_best.pth` artifacts and logs a preference for nested artifact paths. This is non-blocking because the current flat paths are the production source.
- ViT attention rollout did not produce a GradCAM image during smoke testing, but EfficientNet GradCAM was generated and served successfully.

## Fixes Applied

- Added professional README and release documentation.
- Added `.gitignore` for generated/runtime files.
- Cleaned only verified generated QA artifacts and Python bytecode caches.

## Remaining Risks

- No isolated automated backend test suite exists yet.
- SQLite database contains local demo/runtime state and should not be published with real patient data.
- Current upload storage is local filesystem storage, suitable for academic demo but not hardened clinical deployment.
