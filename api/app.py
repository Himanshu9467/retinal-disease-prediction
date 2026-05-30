"""
api/app.py — FastAPI Backend with Live Retinal Capture
"""

import os, sys, base64, uuid, json, logging, io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "database"))

from ensemble import RetinalRiskEnsemble
from model_registry import (
    build_model_registry,
    registry_as_dict,
    validate_registry_paths,
)
from capture import capture_single_frame, generate_frames
import database as db

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("retinarisk.api")

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
DEFAULT_JWT_SECRET = "change-me-dev-retinarisk"
SECRET_KEY   = os.environ.get("JWT_SECRET", DEFAULT_JWT_SECRET)
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 60 * 24
MODEL_NAME   = os.environ.get("MODEL_NAME", "ensemble")
IMG_SIZE     = int(os.environ.get("IMG_SIZE", 224))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
GRADCAM_DIR  = OUTPUT_DIR / "gradcam"
CAPTURE_DIR  = UPLOAD_DIR / "captures"
APP_ENV      = os.environ.get("APP_ENV", "development").lower()
SEED_DEMO_USERS = os.environ.get("SEED_DEMO_USERS", "0") == "1"
RETINAL_REJECTION_ENABLED = os.environ.get("RETINAL_REJECTION_ENABLED", "1") == "1"
DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
]
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
    if origin.strip()
]

MODEL_LABELS = {
    "cnn": "Custom CNN",
    "custom_cnn": "Custom CNN",
    "resnet": "ResNet",
    "efficientnet": "EfficientNet",
    "vit": "ViT",
    "ensemble": "Ensemble",
}

MODEL_REGISTRY = build_model_registry(strict=True)
REGISTRY_ISSUES = validate_registry_paths(MODEL_REGISTRY)
if REGISTRY_ISSUES:
    raise RuntimeError("Invalid production model registry: " + "; ".join(REGISTRY_ISSUES))
ENSEMBLE_WEIGHTS = {name: entry.weight for name, entry in MODEL_REGISTRY.items()}

MODEL_NAME_TO_KEY = {
    "customcnn": "cnn",
    "cnn": "cnn",
    "custom_cnn": "cnn",
    "custom cnn": "cnn",
    "resnet": "resnet",
    "efficientnet": "efficientnet",
    "vit": "vit",
    "vision transformer": "vit",
    "ensemble": "ensemble",
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_history_metrics(history):
    best = None
    best_score = float("-inf")
    for row in history:
        val = row.get("val") if isinstance(row, dict) else None
        if not isinstance(val, dict):
            continue
        score = _safe_float(val.get("weighted_f1", val.get("f1")), -1.0)
        if score > best_score:
            best = val
            best_score = score
    return best


def _canonical_model_key(name: str) -> str:
    normalized = str(name or "").strip().lower().replace("-", " ")
    normalized = " ".join(normalized.split())
    return MODEL_NAME_TO_KEY.get(normalized, normalized.replace(" ", "_"))


def _registry_metric_payload(entry):
    metrics = entry.metrics or {}
    return {
        "accuracy": _safe_float(metrics.get("accuracy")),
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "f1_score": _safe_float(metrics.get("weighted_f1", metrics.get("f1"))),
    }


def _save_data_url_image(data_url: str, prefix: str) -> Optional[str]:
    if not data_url or ";base64," not in data_url:
        return None
    try:
        encoded = data_url.split(";base64,", 1)[1]
        data = base64.b64decode(encoded)
        out_path = GRADCAM_DIR / f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        out_path.write_bytes(data)
        return str(out_path)
    except Exception:
        return None


def _validate_image_bytes(data: bytes, filename: str) -> str:
    if not data:
        raise HTTPException(400, "Uploaded image is empty")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Uploaded image is larger than 20 MB")

    ext = Path(filename or "").suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        raise HTTPException(400, "Unsupported image format")

    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid or corrupted image upload")

    if RETINAL_REJECTION_ENABLED:
        quality = _retinal_image_quality(data)
        if quality["reject"]:
            raise HTTPException(
                422,
                {
                    "message": "Image does not look like a usable retinal fundus photograph.",
                    "quality": quality,
                },
            )
    return ".jpg" if ext == ".jpeg" else ext


def _retinal_image_quality(data: bytes):
    """Conservative image gate for obviously invalid or non-retinal uploads."""
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
    arr = np.asarray(image).astype(np.float32) / 255.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    red_green_balance = float((arr[..., 0].mean() + arr[..., 1].mean()) / (arr[..., 2].mean() + 1e-6))
    gray = arr.mean(axis=2)
    mask = gray > 0.08
    content_ratio = float(mask.mean())
    ys, xs = np.where(mask)
    roundness = 0.0
    if len(xs) > 50:
        width = max(xs.max() - xs.min(), 1)
        height = max(ys.max() - ys.min(), 1)
        roundness = float(min(width, height) / max(width, height))
    score = (
        0.30 * min(content_ratio / 0.45, 1.0)
        + 0.25 * min(contrast / 0.18, 1.0)
        + 0.25 * min(roundness / 0.70, 1.0)
        + 0.20 * min(red_green_balance / 1.10, 1.0)
    )
    reject = score < 0.20 or brightness < 0.03 or contrast < 0.03
    return {
        "score": round(float(score), 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "content_ratio": round(content_ratio, 4),
        "roundness": round(roundness, 4),
        "red_green_balance": round(red_green_balance, 4),
        "reject": bool(reject),
    }


def _safe_media_path(raw_path: str) -> Path:
    if not raw_path:
        raise HTTPException(404, "File not found")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    allowed_roots = [
        (PROJECT_ROOT / "uploads").resolve(),
        OUTPUT_DIR.resolve(),
        (PROJECT_ROOT / "api" / "uploads").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise HTTPException(403, "File is outside allowed media folders")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(404, "File not found")
    return resolved


def _pdf_escape(value) -> str:
    return str(value or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(lines):
    y = 800
    text_ops = ["BT", "/F1 12 Tf", "50 820 Td"]
    for line in lines:
        safe = _pdf_escape(line)
        text_ops.append(f"0 -18 Td ({safe}) Tj")
        y -= 18
        if y < 80:
            break
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_at = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF".encode()
    )
    return bytes(pdf)


def _upsert_ensemble_registry_row():
    weighted = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    used_weight = 0.0

    for key, entry in MODEL_REGISTRY.items():
        weight = float(ENSEMBLE_WEIGHTS.get(key, 0.0))
        if weight <= 0:
            continue
        payload = _registry_metric_payload(entry)
        used_weight += weight
        weighted["accuracy"] += weight * payload["accuracy"]
        weighted["precision"] += weight * payload["precision"]
        weighted["recall"] += weight * payload["recall"]
        weighted["f1"] += weight * payload["f1_score"]

    if used_weight <= 0:
        return

    for metric in weighted:
        weighted[metric] /= used_weight

    db.upsert_model(
        algorithm_name="Ensemble",
        model_version="v1.0",
        accuracy=weighted["accuracy"],
        precision=weighted["precision"],
        recall=weighted["recall"],
        f1_score=weighted["f1"],
        checkpoint_path="ensemble:efficientnet,resnet,vit",
    )


def sync_models_from_outputs():
    """
    Sync the Model table from the active outputs/ production registry.
    """
    synced = 0
    current_ids = {}

    for model_key, entry in MODEL_REGISTRY.items():
        payload = _registry_metric_payload(entry)
        result = db.upsert_model(
            algorithm_name=MODEL_LABELS.get(model_key, model_key.replace("_", " ").title()),
            model_version=entry.version,
            accuracy=payload["accuracy"],
            precision=payload["precision"],
            recall=payload["recall"],
            f1_score=payload["f1_score"],
            checkpoint_path=entry.checkpoint,
        )
        current_ids[model_key] = result["ModelID"]
        synced += 1

    _merge_stale_model_rows(current_ids)
    _upsert_ensemble_registry_row()
    ensemble = db.get_model_by_name("Ensemble")
    if ensemble:
        current_ids["ensemble"] = ensemble["ModelID"]
    _merge_stale_model_rows(current_ids)
    return synced


def _merge_stale_model_rows(current_ids):
    """
    Keep metadata rows anchored to the current outputs/ registry.
    Older non-current rows are merged so predictions keep a valid ModelID.
    """
    for row in db.get_all_models():
        key = _canonical_model_key(row.get("AlgorithmName"))
        target_id = current_ids.get(key)
        if not target_id or row["ModelID"] == target_id:
            continue
        try:
            db.merge_model_records(row["ModelID"], target_id)
        except Exception as exc:
            logger.warning("Could not merge stale model %s -> %s: %s", row["ModelID"], target_id, exc)


def _dedupe_models_by_algorithm():
    """
    Backward-compatible wrapper for older callers.
    Current behavior keeps rows anchored to active outputs/ registry IDs.
    """
    current_ids = {}
    for model_key, entry in MODEL_REGISTRY.items():
        row = db.get_model_by_name(MODEL_LABELS.get(model_key, model_key))
        if row and row.get("ModelVersion") == entry.version:
            current_ids[model_key] = row["ModelID"]
    ensemble = db.get_model_by_name("Ensemble")
    if ensemble:
        current_ids["ensemble"] = ensemble["ModelID"]
    _merge_stale_model_rows(current_ids)

for d in [UPLOAD_DIR, GRADCAM_DIR, CAPTURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────
app = FastAPI(title="CVD Retinal Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def add_no_store_headers(request, call_next):
    response = await call_next(request)
    if request.url.path not in {"/capture/stream"}:
        response.headers["Cache-Control"] = "no-store"
    return response

# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
_ensemble_service = None


def get_ensemble():
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = RetinalRiskEnsemble(
            img_size=IMG_SIZE,
            model_weights=ENSEMBLE_WEIGHTS,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            output_dir=OUTPUT_DIR,
            enable_explainability=True,
            registry=MODEL_REGISTRY,
            strict_registry=True,
        )
    return _ensemble_service

# ─────────────────────────────────────────────
#  Auth
# ─────────────────────────────────────────────
security = HTTPBearer()

def create_token(user_id, email, role):
    payload = {"sub": str(user_id), "email": email, "role": role,
               "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

def require_doctor(token=Depends(verify_token)):
    if token.get("role") != "Doctor":
        raise HTTPException(403, "Doctor access required")
    return token

# ─────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    name: str; email: str; password: str; role: str

class PatientCreate(BaseModel):
    name: str; age: int; gender: str; email: str; phone: str

class PatientUpdate(BaseModel):
    name: str; age: int; gender: str; email: str; phone: str

class ChecklistUpdate(BaseModel):
    FollowUpAdvised: bool = False
    DoctorReviewed: bool = False
    LifestyleCounseling: bool = False
    RescreenScheduled: bool = False

# ─────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup():
    if APP_ENV == "production" and SECRET_KEY == DEFAULT_JWT_SECRET:
        raise RuntimeError("JWT_SECRET must be set in production.")
    if SECRET_KEY == DEFAULT_JWT_SECRET:
        logger.warning("Using development JWT secret. Set JWT_SECRET before deployment.")
    db.init_db()
    try:
        get_ensemble()
    except Exception as exc:
        logger.warning("Model ensemble did not preload: %s", exc)

    if SEED_DEMO_USERS:
        doctor_password = os.environ.get("DEMO_DOCTOR_PASSWORD")
        staff_password = os.environ.get("DEMO_STAFF_PASSWORD")
        if not doctor_password or not staff_password:
            raise RuntimeError("Set DEMO_DOCTOR_PASSWORD and DEMO_STAFF_PASSWORD to seed demo users.")
        db.create_user("Dr. Sharma", "doctor@rit.edu", doctor_password, "Doctor")
        db.create_user("Nurse Priya", "staff@rit.edu",  staff_password,  "Medical Staff")

    synced_count = sync_models_from_outputs()
    if synced_count:
        print(f"[API] Synced {synced_count} production model record(s).")
    else:
        print("[API] No production model metrics found to sync.")

    print("[API] Server ready.")

# ─────────────────────────────────────────────
#  Auth Endpoints
# ─────────────────────────────────────────────
@app.post("/auth/register")
def register(req: RegisterRequest):
    if req.role not in ["Doctor", "Medical Staff"]:
        raise HTTPException(400, "Invalid role")
    if len(req.password) < 10:
        raise HTTPException(400, "Password must be at least 10 characters")
    result = db.create_user(req.name, req.email, req.password, req.role)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result

@app.post("/auth/login")
def login(req: LoginRequest):
    result = db.authenticate_user(req.email, req.password)
    if not result["success"]:
        raise HTTPException(401, "Invalid email or password")
    user  = result["user"]
    token = create_token(user["UserID"], user["Email"], user["Role"])
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user["UserID"], "name": user["Name"],
                     "email": user["Email"], "role": user["Role"]}}

@app.get("/auth/me")
def get_me(token=Depends(verify_token)):
    return token

# ─────────────────────────────────────────────
#  Patient Endpoints
# ─────────────────────────────────────────────
@app.get("/patients")
def list_patients(token=Depends(verify_token)):
    return db.get_all_patients()

@app.get("/patients/search")
def search_patients(q: str = "", token=Depends(verify_token)):
    return db.search_patients(q)

@app.post("/patients")
def create_patient(patient: PatientCreate, token=Depends(verify_token)):
    if not patient.email.strip():
        raise HTTPException(400, "Patient email is required")
    if not patient.phone.strip():
        raise HTTPException(400, "Patient phone number is required")
    return db.create_patient(patient.name, patient.age, patient.gender,
                             patient.email.strip(), patient.phone.strip(), int(token["sub"]))

@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, token=Depends(verify_token)):
    p = db.get_patient(patient_id)
    if not p: raise HTTPException(404, "Patient not found")
    p["predictions"] = db.get_patient_predictions(patient_id)
    p["checklist"] = db.get_patient_checklist(patient_id)
    return p

@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, patient: PatientUpdate, token=Depends(verify_token)):
    if not patient.email.strip():
        raise HTTPException(400, "Patient email is required")
    if not patient.phone.strip():
        raise HTTPException(400, "Patient phone number is required")
    result = db.update_patient(
        patient_id,
        patient.name.strip(),
        patient.age,
        patient.gender,
        patient.email.strip(),
        patient.phone.strip(),
    )
    if not result["success"]:
        raise HTTPException(404, "Patient not found")
    return result

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, token=Depends(verify_token)):
    result = db.delete_patient(patient_id)
    if not result["success"]:
        raise HTTPException(404, "Patient not found")
    return result

@app.get("/predictions/history/{patient_id}")
def patient_prediction_history(patient_id: int, token=Depends(verify_token)):
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    return {
        "patient": patient,
        "predictions": db.get_patient_predictions(patient_id),
        "checklist": db.get_patient_checklist(patient_id),
    }

@app.get("/patients/{patient_id}/checklist")
def get_patient_checklist(patient_id: int, token=Depends(verify_token)):
    if not db.get_patient(patient_id):
        raise HTTPException(404, "Patient not found")
    return db.get_patient_checklist(patient_id)

@app.put("/patients/{patient_id}/checklist")
def update_patient_checklist(
    patient_id: int,
    checklist: ChecklistUpdate,
    token=Depends(verify_token),
):
    result = db.save_patient_checklist(patient_id, checklist.dict())
    if not result["success"]:
        raise HTTPException(404, "Patient not found")
    return result

# ─────────────────────────────────────────────
#  Inference Helper
# ─────────────────────────────────────────────
def run_prediction(img_path: str, patient_id: int):
    try:
        ensemble = get_ensemble()
        inference = ensemble.ensemble_predict(img_path)
    except Exception as exc:
        logger.exception("Prediction failed for patient %s image %s", patient_id, img_path)
        raise HTTPException(500, f"Prediction failed: {exc}")

    final_risk = inference.get("retinal_severity") or inference.get("prediction")
    confidence = float(inference.get("ensemble_confidence") or inference.get("confidence_score") or 0.0)

    explanation_path = None
    for key in ("efficientnet_gradcam", "resnet_gradcam", "vit_gradcam"):
        maybe_data_url = inference.get("explainability", {}).get(key)
        if maybe_data_url:
            explanation_path = _save_data_url_image(maybe_data_url, prefix=key)
            if explanation_path:
                break

    ensemble_model = db.get_model_by_name("Ensemble")
    model_id = ensemble_model["ModelID"] if ensemble_model else None
    if model_id is None:
        best_model_row = db.get_best_model()
        model_id = best_model_row["ModelID"] if best_model_row else None

    stored = db.save_prediction(
        patient_id,
        model_id,
        final_risk,
        confidence,
        explanation_path,
    )

    return {
        "prediction": final_risk,
        "confidence_percent": round(confidence * 100, 2),
        "models": inference.get("models", inference.get("individual_model_predictions", {})),
        "prediction_id": stored["PredictionID"],
        "patient_id": patient_id,
        "result": final_risk,
        "confidence": round(confidence, 4),
        "class_probabilities": inference["ensemble_probabilities"],
        "ensemble_confidence": inference["ensemble_confidence"],
        "ensemble_probabilities": inference["ensemble_probabilities"],
        "ensemble_probability": inference["ensemble_confidence"],
        "ensemble_prediction": inference["ensemble_prediction"],
        "individual_model_predictions": inference.get("individual_model_predictions", {}),
        "bp_category": inference.get("bp_category"),
        "cholesterol": inference.get("cholesterol"),
        "cvd_risk": final_risk,
        "retinal_severity": final_risk,
        "probability_risk_band": inference.get("probability_risk_band"),
        "risk_score": inference.get("risk_score"),
        "model_agreement": inference.get("model_agreement"),
        "model_agreement_score": inference.get("model_agreement_score"),
        "loaded_models": inference.get("loaded_models", []),
        "ensemble_weights": inference.get("ensemble_weights", {}),
        "model_registry": inference.get("model_registry", {}),
        "gradcam_available": explanation_path is not None,
        "explainability": inference.get("explainability", {}),
        "vessel_features": inference.get("vessel_features", {}),
        "target": inference.get("target"),
        "target_description": inference.get("target_description"),
        "note": inference.get("note"),
        "timestamp": datetime.utcnow().isoformat(),
    }

# ─────────────────────────────────────────────
#  Predict from Upload
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(
    patient_id: int = Form(...),
    image: UploadFile = File(...),
    symptoms: Optional[str]   = Form(None),
    medical_history: Optional[str] = Form(None),
    blood_pressure: Optional[str]  = Form(None),
    cholesterol: Optional[str]   = Form(None),
    token=Depends(verify_token),
):
    if not db.get_patient(patient_id):
        raise HTTPException(404, "Patient not found")

    data = await image.read()
    ext = _validate_image_bytes(data, image.filename)
    img_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(img_path, "wb") as f:
        f.write(data)

    db.add_clinical_data(patient_id, str(img_path), symptoms,
                         medical_history, blood_pressure, cholesterol)
    return run_prediction(str(img_path), patient_id)


# ─────────────────────────────────────────────
#  🔴 LIVE CAPTURE ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/capture/stream")
def stream_camera(camera_index: int = 0, token: Optional[str] = None):
    """
    Streams live MJPEG camera feed with retinal guide overlay.
    Accepts token as query param because browser <img> tags cannot
    send Authorization headers.
    """
    # Validate token from query param
    if not token:
        raise HTTPException(401, "Token required")
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

    return StreamingResponse(
        generate_frames(camera_index),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache"}
    )


@app.post("/capture/snap")
def snap_frame(camera_index: int = 0, token=Depends(verify_token)):
    """
    Silently captures a single frame from the camera.
    Returns the saved image path.
    """
    try:
        path = capture_single_frame(camera_index)
        # Return base64 preview too
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {
            "success": True,
            "image_path": path,
            "preview": f"data:image/jpeg;base64,{b64}"
        }
    except Exception as e:
        raise HTTPException(500, f"Camera error: {str(e)}")


@app.post("/capture/predict")
async def capture_and_predict(
    patient_id: int  = Form(...),
    camera_index: int = Form(0),
    token=Depends(verify_token),
):
    """
    One-step: Capture from camera → Enhance → Predict retinal severity.
    """
    if not db.get_patient(patient_id):
        raise HTTPException(404, "Patient not found")

    # Capture & enhance via OpenCV
    try:
        img_path = capture_single_frame(camera_index)
    except Exception as e:
        raise HTTPException(500, f"Camera capture failed: {str(e)}")

    # Save clinical data
    db.add_clinical_data(patient_id, img_path)

    # Run model prediction
    return run_prediction(img_path, patient_id)


@app.get("/capture/check")
def check_camera(camera_index: int = 0, token=Depends(verify_token)):
    """Check if camera is available."""
    import cv2
    cap = cv2.VideoCapture(camera_index)
    available = cap.isOpened()
    cap.release()
    return {"camera_available": available, "camera_index": camera_index}


# ─────────────────────────────────────────────
#  Stats & Models
# ─────────────────────────────────────────────
@app.get("/stats")
def get_stats(token=Depends(verify_token)):
    stats = db.get_prediction_stats()
    stats["models"]   = len(db.get_all_models())
    stats["patients"] = len(db.get_all_patients())
    return stats

@app.get("/predictions")
def list_predictions(limit: int = 50, token=Depends(verify_token)):
    return db.get_all_predictions(limit)

@app.get("/media")
def get_media(path: str, token: Optional[str] = None):
    if not token:
        raise HTTPException(401, "Token required")
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
    resolved = _safe_media_path(path)
    return FileResponse(resolved)


@app.get("/gradcam/{prediction_id}")
def get_gradcam(prediction_id: int, token=Depends(verify_token)):
    prediction = db.get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(404, "Prediction not found")
    explanation_path = prediction.get("ExplanationPath")
    if not explanation_path:
        raise HTTPException(404, "GradCAM image not available for this prediction")
    resolved = _safe_media_path(explanation_path)
    return FileResponse(resolved)

@app.delete("/predictions")
def clear_predictions(token=Depends(verify_token)):
    return db.delete_all_predictions()

@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int, token=Depends(verify_token)):
    result = db.delete_prediction(prediction_id)
    if not result["success"]:
        raise HTTPException(404, "Prediction not found")
    return result

@app.get("/report/generate/{patient_id}")
def generate_patient_report(patient_id: int, token=Depends(verify_token)):
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    predictions = db.get_patient_predictions(patient_id)
    latest = predictions[0] if predictions else {}
    risk = latest.get("PredictionResult", "Not screened")
    confidence = latest.get("ConfidenceScore")
    guidance = {
        "Disease Detected": "Arrange prompt clinician or ophthalmology review and confirm the retinal screening result.",
        "At-Risk": "Schedule follow-up retinal review and assess relevant clinical risk factors.",
        "Normal": "Continue routine screening and healthy prevention habits.",
    }.get(risk, "Run retinal screening to generate a recommendation.")
    lines = [
        "RetinaRisk Retinal Severity Screening Patient Report",
        f"Generated: {datetime.utcnow().isoformat()} UTC",
        "Model target: retinal image severity mapped from dataset labels.",
        "Not a CVD, blood-pressure, or cholesterol diagnosis.",
        "",
        f"Patient ID: {patient.get('PatientID')}",
        f"Name: {patient.get('Name')}",
        f"Age: {patient.get('Age')}",
        f"Gender: {patient.get('Gender')}",
        f"Email: {patient.get('Email')}",
        f"Phone: {patient.get('Phone')}",
        "",
        f"Latest prediction: {risk}",
        f"Confidence score: {round(float(confidence or 0) * 100, 2) if confidence is not None else 'N/A'}%",
        f"Model: {latest.get('AlgorithmName') or 'N/A'}",
        f"Screened on: {latest.get('Timestamp') or 'N/A'}",
        "",
        f"Recommendation: {guidance}",
        "",
        "Previous screenings:",
    ]
    for row in predictions[:10]:
        lines.append(
            f"- #{row.get('PredictionID')} {row.get('PredictionResult')} "
            f"({round(float(row.get('ConfidenceScore') or 0) * 100, 2)}%) {row.get('Timestamp')}"
        )
    pdf = _build_simple_pdf(lines)
    filename = f"retinarisk_patient_{patient_id}_report.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.get("/models")
def list_models(token=Depends(verify_token)):
    sync_models_from_outputs()
    return db.get_all_models()


@app.get("/model-info")
def model_info():
    return {
        "status": "ready" if _ensemble_service and _ensemble_service.models else "configured",
        "class_order": ["At-Risk", "Disease Detected", "Normal"],
        "registry": registry_as_dict(MODEL_REGISTRY),
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "loaded_models": list(_ensemble_service.models.keys()) if _ensemble_service else [],
        "excluded_models": [
            name for name, weight in ENSEMBLE_WEIGHTS.items() if weight <= 0 and name in MODEL_REGISTRY
        ],
        "model_output_dir": str(OUTPUT_DIR),
        "metrics_source": "outputs/history.json or outputs/<model>_history.json",
        "active_checkpoints": {
            name: entry.checkpoint for name, entry in MODEL_REGISTRY.items() if entry.weight > 0
        },
    }


@app.post("/models/sync")
def sync_model_registry(token=Depends(require_doctor)):
    synced = sync_models_from_outputs()
    return {
        "success": True,
        "synced_models": synced,
        "models": db.get_all_models(),
    }

@app.get("/health")
def health():
    ready = bool(_ensemble_service and _ensemble_service.models)
    return {
        "status": "ok" if ready else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(_ensemble_service.models.keys()) if _ensemble_service else [],
        "models_excluded": [
            name for name, weight in ENSEMBLE_WEIGHTS.items() if weight <= 0 and name in MODEL_REGISTRY
        ],
        "registry_issues": REGISTRY_ISSUES,
    }
