# CardioVision AI:
# Cardiovascular Disease Prediction Using Retinal Fundus Images and Deep Learning

CardioVision AI is an AI-powered healthcare platform for cardiovascular disease risk screening from retinal fundus images. It combines a FastAPI backend, React/Vite frontend, SQLite database, JWT authentication, and PyTorch production models.

## Overview

### Problem Statement

Cardiovascular disease can progress silently, and early risk screening is often limited by access to specialist diagnostics. Retinal fundus images contain vascular biomarkers that can support non-invasive risk assessment and clinical triage.

### Objective

Build a full-stack clinical AI platform that accepts retinal fundus images, performs binary disease risk prediction, stores patient and prediction history, and presents analytics through a secure dashboard.

### Solution

The system uses EfficientNet-B3 as the active production predictor because it has the best registered test metrics. ResNet50, MobileNetV3-Large, CNN, and a weighted ensemble reference remain available for comparison and review.

## Features

- Secure login and registration with JWT authentication
- Retinal image upload and live capture support
- AI prediction for binary Normal/Disease classification
- Best-model production prediction with model comparison metrics
- Dashboard analytics and prediction history
- Patient management and clinical follow-up checklist
- Explainability support through Grad-CAM artifacts
- Model registry and production model metrics
- Responsive React frontend for clinical workflows

## Technologies Used

### Frontend

- React
- Vite
- lucide-react

### Backend

- FastAPI
- Uvicorn
- Pydantic
- PyJWT

### AI/ML

- PyTorch
- Torchvision
- EfficientNet-B3
- ResNet50
- MobileNetV3-Large
- CNN
- EfficientNet-B3 production inference
- Ensemble soft voting reference

### Database

- SQLite

## Dataset

The training pipeline uses patient-level splitting to reduce leakage risk and balanced binary classes for Normal and Disease classification.

### Dataset Statistics

| Split | Normal | Disease | Total |
| --- | ---: | ---: | ---: |
| Train | 2800 | 2800 | 5600 |
| Validation | 350 | 350 | 700 |
| Test | 350 | 350 | 700 |

## Model Performance

Final test metrics are stored in `research_training_outputs2/test_summary_<model>.json`.

| Model | Accuracy | Precision | Recall | Macro F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNet-B3 | 95.14% | 98.77% | 91.43% | 95.14% | 99.03% | 99.22% |
| ResNet50 | 93.14% | 96.60% | 89.43% | 93.13% | 98.57% | 98.78% |
| MobileNetV3-Large | 93.43% | 98.41% | 88.29% | 93.41% | 98.49% | 98.62% |
| CNN | 86.57% | 95.07% | 77.14% | 86.45% | 95.01% | 95.62% |
| Weighted ensemble reference | 93.54% | 97.89% | 88.97% | 93.52% | 99.13% | |

## Production Model Selection

EfficientNet-B3 is the active production model because it is higher than the weighted ensemble reference on the current registered accuracy and macro F1 shown by the dashboard.

Weighted ensemble reference configuration:

```json
{
  "efficientnet": 0.50,
  "resnet": 0.20,
  "mobilenet": 0.20,
  "cnn": 0.10
}
```

The API reports production predictions with `"prediction_source": "efficientnet"` and `"production_model": "EfficientNet-B3"`.

Expected loaded model keys:

```json
[
  "efficientnet",
  "resnet",
  "mobilenet",
  "cnn"
]
```

Runtime verification confirmed:

- All four checkpoint files exist in `research_training_outputs2`.
- All four models load successfully.
- EfficientNet-B3 is used for the final production probability.
- The weighted ensemble probability is retained as a secondary comparison signal.

Example verification output:

| Model | Disease Probability |
| --- | ---: |
| EfficientNet-B3 | 0.3473 |
| ResNet50 | 0.4276 |
| MobileNetV3-Large | 0.7002 |
| CNN | 0.8340 |
| Weighted Ensemble Reference | 0.5773 |

Manual weighted average: `0.5773`

## Project Structure

```text
Retinal-image-prediction/
├── api/
│   ├── app.py
│   ├── database/
│   │   └── cvd_system.db
│   └── uploads/
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── labels.csv
│   └── prepare_data.py
├── database/
│   ├── database.py
│   └── cvd_system.db
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       ├── main.jsx
│       └── components/
│           └── dashboard/
├── research_binary_7000/
├── research_patient_safe_aug_dataset/
├── research_training_outputs2/
│   ├── best_efficientnet.pt
│   ├── best_resnet.pt
│   ├── best_mobilenet.pt
│   ├── best_cnn.pt
│   ├── test_summary_efficientnet.json
│   ├── test_summary_resnet.json
│   ├── test_summary_mobilenet.json
│   ├── test_summary_cnn.json
│   └── all_models_summary.csv
├── src/
│   ├── ensemble/
│   │   └── ensemble.py
│   ├── capture.py
│   ├── dataset.py
│   ├── gradcam.py
│   ├── model_registry.py
│   ├── production_models.py
│   └── train.py
├── uploads/
├── requirements.txt
└── README.md
```

## Screenshots

Add presentation screenshots here:

- Login page: `docs/screenshots/login.png`
- Dashboard: `docs/screenshots/dashboard.png`
- Prediction workflow: `docs/screenshots/prediction.png`
- Analytics: `docs/screenshots/analytics.png`

## API Endpoints

Backend base URL: `http://127.0.0.1:8001`

| Method | Endpoint | Purpose | Auth |
| --- | --- | --- | --- |
| `POST` | `/auth/login` | Login and receive JWT token | No |
| `POST` | `/auth/register` | Create user account | No |
| `GET` | `/auth/me` | Validate current token | Yes |
| `POST` | `/predict` | Upload retinal image and generate prediction | Yes |
| `GET` | `/predictions` | Prediction history | Yes |
| `DELETE` | `/predictions/{prediction_id}` | Delete prediction | Yes |
| `GET` | `/stats` | Dashboard statistics | Yes |
| `GET` | `/models` | Production analytics model list | Yes |
| `POST` | `/models/sync` | Sync model metrics into database | Doctor |
| `GET` | `/model-info` | Model registry metadata | No |
| `GET` | `/health` | Backend health status | No |

Swagger UI: `http://127.0.0.1:8001/docs`

## Security

### JWT Secret Configuration (Required)

The backend requires a `JWT_SECRET` environment variable. **The server will not start without it.**

#### Step 1 — Generate a Strong Secret

**PowerShell:**

```powershell
[guid]::NewGuid().ToString() + [guid]::NewGuid().ToString()
```

**Python:**

```python
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

#### Step 2 — Create `.env` File

Copy the example file and set your generated secret:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder:

```env
JWT_SECRET=your_generated_secret_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
```

#### Step 3 — Start Backend

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8001 --reload
```

#### Step 4 — Verify Authentication

Open Swagger UI at `http://127.0.0.1:8001/docs` and test login.

> **⚠️ WARNING: Never commit your `.env` file or real secrets to version control.**
>
> **⚠️ WARNING: Never use placeholder or development secrets in production.**
>
> The `.gitignore` file is configured to exclude `.env` automatically.

### Security Details

- Authentication uses JWT bearer tokens.
- Protected endpoints require `Authorization: Bearer <token>`.
- `JWT_SECRET` is **mandatory** — the backend fails at startup if it is missing.
- JWT algorithm and token expiration are configurable via `JWT_ALGORITHM` and `JWT_EXPIRE_MINUTES`.
- API responses include no-store cache headers for most endpoints.
- Media paths are validated to stay inside approved upload/output directories.

### Production Environment Example

```powershell
$env:APP_ENV="production"
$env:JWT_SECRET="replace-with-a-strong-64-byte-minimum-secret"
```

## How To Run Project

### Backend

1. Create `.env` from the example file:

```powershell
Copy-Item .env.example .env
```

   Edit `.env` and set a strong `JWT_SECRET` (see Security section above).

2. Create virtual environment:

```powershell
python -m venv venv
```

3. Activate on Windows:

```powershell
venv\Scripts\activate
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Start backend:

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8001 --reload
```

Backend URL:

```text
http://127.0.0.1:8001
```

Swagger:

```text
http://127.0.0.1:8001/docs
```

### Frontend

1. Enter frontend:

```powershell
cd frontend
```

2. Install packages:

```powershell
npm install
```

3. Start frontend:

```powershell
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

If the backend runs on a different URL, set `VITE_API_BASE` before starting Vite.

## Validation Checklist

Completed final validation:

- Model registry contains EfficientNet-B3, ResNet50, MobileNetV3-Large, and CNN.
- Ensemble weights use Configuration A: EfficientNet-B3 `0.50`, ResNet50 `0.20`, MobileNetV3-Large `0.20`, CNN `0.10`.
- Backend startup smoke check passed.
- Backend `/health` returned `ok` after lifespan startup.
- `/models` returned only EfficientNet-B3, ResNet50, MobileNetV3-Large, and CNN.
- Frontend lint passed.
- Frontend production build passed.
- `requirements.txt`, `frontend/package.json`, and `frontend/package-lock.json` are present.

## Remaining Production Notes

- Set `JWT_SECRET` before production deployment.
- Keep model checkpoints in `research_training_outputs2` or set `MODEL_REGISTRY_DIR`.
- The current registry supports flat artifact filenames such as `best_efficientnet.pt`; nested model folders are optional.
- Add final UI screenshots before presentation submission.
- Run deployment-specific network, HTTPS, and backup checks in the target hosting environment.

## Disclaimer

CardioVision AI is a clinical decision-support and screening tool. It is not a standalone diagnosis system. Final medical interpretation, treatment, and follow-up decisions must be made by qualified clinicians.
