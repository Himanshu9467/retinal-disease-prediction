"""
capture.py — Intelligent Retinal Image Capture with ML Quality Pipeline
========================================================================
100% Windows OpenCV 4.13 AVX2-safe.

Key design decisions:
- Eye detection is ADVISORY only — never blocks capture.
  Haar cascade is unreliable for close-up/fundus/webcam setups.
- Quality gate based on blur + brightness + contrast only.
- All image processing uses numpy / scipy — no GaussianBlur,
  Laplacian, addWeighted, bilateralFilter, LUT, or INTER_LANCZOS4.
"""

import cv2
import numpy as np
import uuid
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.ndimage import uniform_filter

# ─────────────────────────────────────────────
#  Paths & Constants
# ─────────────────────────────────────────────

SAVE_DIR = Path("./uploads/captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Only blur + brightness + contrast are used for the hard gate.
# Eye detection score is shown in the HUD but never blocks capture.
MIN_QUALITY_SCORE  = 0.35   # low enough for any decent webcam frame
FRAME_CANDIDATES   = 8
EYE_CONFIDENCE_MIN = 0.25   # advisory only

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"


# ─────────────────────────────────────────────
#  Data class
# ─────────────────────────────────────────────

@dataclass
class QualityReport:
    blur_score:       float
    brightness_score: float
    contrast_score:   float
    noise_score:      float
    eye_detected:     bool   # advisory — does NOT block capture
    eye_confidence:   float
    overall:          float  # computed WITHOUT eye component

    def passed(self) -> bool:
        # Eye detection intentionally excluded from gate
        return self.overall >= MIN_QUALITY_SCORE

    def summary(self) -> str:
        status = "PASS" if self.passed() else "FAIL"
        return (f"[{status}] Q={self.overall:.2f} Blur={self.blur_score:.2f} "
                f"Bright={self.brightness_score:.2f} "
                f"Eye={'detected' if self.eye_detected else 'not found (ok)'}")


# ─────────────────────────────────────────────
#  1. Eye Detection  (advisory)
# ─────────────────────────────────────────────

_eye_cascade: Optional[cv2.CascadeClassifier] = None

def _get_cascade() -> Optional[cv2.CascadeClassifier]:
    global _eye_cascade
    if _eye_cascade is None:
        c = cv2.CascadeClassifier(CASCADE_PATH)
        _eye_cascade = None if c.empty() else c
    return _eye_cascade


def detect_eye(frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, tuple]:
    """
    Try to detect an eye. Returns (roi, confidence, bbox).
    Never raises — returns (None, 0.0, ()) on any failure.
    """
    try:
        cascade = _get_cascade()
        if cascade is None:
            return None, 0.0, ()

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes  = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4,
            minSize=(25, 25), flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(eyes) == 0:
            return None, 0.0, ()

        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
        x, y, w, h = eyes[0]
        confidence = min(1.0, (w * h / (frame.shape[0] * frame.shape[1])) * 20)

        pad_x = int(w * 0.20);  pad_y = int(h * 0.20)
        x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        return frame[y1:y2, x1:x2], confidence, (x, y, w, h)
    except Exception:
        return None, 0.0, ()


# ─────────────────────────────────────────────
#  2. Quality Assessment  (numpy-only, AVX2-safe)
# ─────────────────────────────────────────────

def _blur_score(gray_f32: np.ndarray) -> float:
    """
    Laplacian variance via numpy finite differences.
    Avoids cv2.Laplacian (triggers AVX2 getLinearFilter crash).
    """
    c   = gray_f32[1:-1, 1:-1]
    lap = (4*c
           - gray_f32[0:-2, 1:-1]
           - gray_f32[2:,   1:-1]
           - gray_f32[1:-1, 0:-2]
           - gray_f32[1:-1, 2:  ])
    return float(np.clip(lap.var() / 600.0, 0.0, 1.0))


def assess_quality(frame: np.ndarray) -> QualityReport:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    blur_score       = _blur_score(gray)
    mean_bright      = gray.mean() / 255.0
    brightness_score = float(max(0.0, 1.0 - abs(mean_bright - 0.45) * 2.0))
    contrast_score   = float(np.clip(gray.std() / 70.0, 0.0, 1.0))

    blurred     = uniform_filter(gray, size=5)
    noise_score = float(np.clip(1.0 - np.abs(gray - blurred).mean() / 25.0, 0.0, 1.0))

    # Eye detection is advisory — run but don't include in overall gate score
    _, eye_conf, _ = detect_eye(frame)
    eye_detected   = eye_conf >= EYE_CONFIDENCE_MIN

    # Overall = blur(50%) + brightness(25%) + contrast(25%)
    # Eye deliberately excluded so a failed detection never blocks capture
    overall = (blur_score * 0.50 +
               brightness_score * 0.25 +
               contrast_score * 0.25)

    return QualityReport(
        blur_score=round(blur_score, 3),
        brightness_score=round(brightness_score, 3),
        contrast_score=round(contrast_score, 3),
        noise_score=round(noise_score, 3),
        eye_detected=eye_detected,
        eye_confidence=round(eye_conf, 3),
        overall=round(overall, 3),
    )


# ─────────────────────────────────────────────
#  3. Best-Frame Selector
# ─────────────────────────────────────────────

def select_best_frame(
    cap: cv2.VideoCapture, n: int = FRAME_CANDIDATES
) -> Tuple[Optional[np.ndarray], QualityReport]:
    candidates: List[Tuple[np.ndarray, QualityReport]] = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            continue
        candidates.append((frame, assess_quality(frame)))
        time.sleep(0.04)

    if not candidates:
        return None, QualityReport(0, 0, 0, 0, False, 0, 0)

    candidates.sort(key=lambda c: c[1].overall, reverse=True)
    best, report = candidates[0]
    print(f"[Quality] Best of {len(candidates)} → {report.summary()}")
    return best, report


# ─────────────────────────────────────────────
#  4. Image Cleaning  (AVX2-safe)
# ─────────────────────────────────────────────

def _box_blur(image: np.ndarray, size: int = 5) -> np.ndarray:
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        out[:, :, c] = uniform_filter(image[:, :, c].astype(np.float32), size=size)
    return out


def denoise(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    chs = cv2.split(image)
    return cv2.merge([cv2.medianBlur(ch, 3) for ch in chs])


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    chs   = cv2.split(image)
    return cv2.merge([clahe.apply(ch) for ch in chs])


def colour_normalise(image: np.ndarray) -> np.ndarray:
    out = np.empty_like(image, dtype=np.uint8)
    for c in range(3):
        ch = image[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = image[:, :, c]
    return out


def gamma_correct(image: np.ndarray, gamma: float = 1.15) -> np.ndarray:
    table = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0
    return table[image].astype(np.uint8)


def green_boost(image: np.ndarray) -> np.ndarray:
    g3 = np.stack([image[:, :, 1]] * 3, axis=2).astype(np.float32)
    return np.clip(image.astype(np.float32) * 0.65 + g3 * 0.35, 0, 255).astype(np.uint8)


def sharpen(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    blurred = _box_blur(image, size=5)
    out = image.astype(np.float32) * (1.0 + strength) - blurred * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_retinal_image(image: np.ndarray, report: Optional[QualityReport] = None) -> np.ndarray:
    out = denoise(image.copy())
    out = colour_normalise(out)
    if report is None or report.brightness_score < 0.45:
        out = gamma_correct(out)
    out = clahe_enhance(out)
    out = green_boost(out)
    strength = 0.8 if (report and report.blur_score < 0.4) else 0.5
    return sharpen(out, strength=strength)


# ─────────────────────────────────────────────
#  5. Validation Gate
# ─────────────────────────────────────────────

def validate_and_clean(
    frame: np.ndarray,
    report: Optional[QualityReport] = None,
) -> Tuple[Optional[np.ndarray], QualityReport, str]:

    if report is None:
        report = assess_quality(frame)

    # Only hard-fail on actual image quality — NOT eye detection
    if report.overall < MIN_QUALITY_SCORE:
        return None, report, (
            f"Image quality too low ({report.overall:.2f} < {MIN_QUALITY_SCORE}). "
            "Ensure good lighting and hold camera steady."
        )

    # Use eye ROI if detected, otherwise use full frame
    roi, _, _ = detect_eye(frame)
    if roi is None or roi.size == 0:
        roi = frame   # full frame fallback — always works

    roi_resized = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_AREA)
    cleaned     = clean_retinal_image(roi_resized, report)

    eye_note = " (eye region cropped)" if roi is not frame else " (full frame used)"
    return cleaned, report, f"Quality OK ({report.overall:.2f}){eye_note} — cleaned and ready."


# ─────────────────────────────────────────────
#  Overlay
# ─────────────────────────────────────────────

def apply_retinal_overlay(
    frame: np.ndarray,
    report: Optional[QualityReport] = None,
    eye_bbox: tuple = (),
) -> np.ndarray:
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 3

    mask    = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    darkened = (frame * 0.4).astype(np.uint8)
    out      = np.where(mask[:, :, np.newaxis] == 255, frame, darkened)

    # Ring colour based on quality (not eye detection)
    if report:
        if report.passed():
            ring_col = (0, 255, 100)    # green  — good quality
        elif report.overall >= MIN_QUALITY_SCORE * 0.7:
            ring_col = (0, 200, 255)    # amber  — borderline
        else:
            ring_col = (0, 60, 255)     # red    — poor quality
    else:
        ring_col = (0, 200, 200)

    inner_col = tuple(c // 2 for c in ring_col)
    cv2.circle(out, (cx, cy), radius,     ring_col, 2)
    cv2.circle(out, (cx, cy), radius + 4, inner_col, 1)

    if eye_bbox:
        x, y, ew, eh = eye_bbox
        cv2.rectangle(out, (x, y), (x + ew, y + eh), (0, 255, 200), 1)

    cv2.line(out, (cx - radius, cy), (cx + radius, cy), ring_col, 1)
    cv2.line(out, (cx, cy - radius), (cx, cy + radius), ring_col, 1)

    cv2.rectangle(out, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(out, "RetinaRisk AI  |  Retinal Capture",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 210, 150), 1)

    if report:
        q_col  = (0, 220, 80) if report.passed() else (0, 100, 255)
        eye_str = "Eye: detected" if report.eye_detected else "Eye: not found"
        q_text = f"Q={report.overall:.2f}  Blur={report.blur_score:.2f}  {eye_str}"
        cv2.rectangle(out, (0, h - 62), (w, h), (0, 0, 0), -1)
        cv2.putText(out, q_text, (12, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, q_col, 1)
        cv2.putText(out, "SPACE = Capture  |  Q = Quit",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1)

    return out


# ─────────────────────────────────────────────
#  Public API — Interactive Capture
# ─────────────────────────────────────────────

def capture_retinal_image(camera_index: int = 0, save_path: Optional[str] = None) -> Optional[str]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Capture] ERROR: Cannot open camera {camera_index}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    print("[Capture] Ready — SPACE to capture, Q to quit")

    live_report = None;  live_bbox = ();  flash = 0;  result_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        live_report     = assess_quality(frame)
        _, _, live_bbox = detect_eye(frame)

        if flash > 0:
            display = (np.ones_like(frame) * 255).astype(np.uint8)
            flash  -= 1
        else:
            display = apply_retinal_overlay(frame.copy(), live_report, live_bbox)

        cv2.imshow("RetinaRisk AI — Retinal Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            print(f"[Capture] Sampling {FRAME_CANDIDATES} frames…")
            best, best_report = select_best_frame(cap, FRAME_CANDIDATES)
            if best is None:
                continue

            cleaned, rpt, msg = validate_and_clean(best, best_report)
            print(f"[Capture] {msg}")

            if cleaned is None:
                rej = frame.copy()
                cv2.putText(rej, msg[:65], (20, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)
                cv2.imshow("RetinaRisk AI — Retinal Capture", rej)
                cv2.waitKey(1500)
                continue

            if save_path is None:
                fname     = f"retinal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
                save_path = str(SAVE_DIR / fname)

            cv2.imwrite(save_path, cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result_path = save_path
            flash = 3
            print(f"[Capture] Saved → {save_path}")
            cv2.waitKey(600)
            break

        elif key in (ord('q'), ord('Q')):
            print("[Capture] Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return result_path


# ─────────────────────────────────────────────
#  Public API — Silent Single Frame (API)
# ─────────────────────────────────────────────

def _open_camera_with_retry(camera_index: int, retries: int = 3) -> cv2.VideoCapture:
    """
    Try opening the camera multiple times with delays.
    Windows MSMF sometimes needs a retry after initial failure.
    """
    for attempt in range(retries):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if cap.isOpened():
            # Lower resolution avoids MSMF buffer errors on laptops
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            print(f"[Capture] Camera opened on attempt {attempt+1}")
            return cap
        cap.release()
        print(f"[Capture] Camera open failed (attempt {attempt+1}/{retries}), retrying...")
        time.sleep(1.0)

    # Fallback: try without MSMF backend
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[Capture] Camera opened with fallback backend")
        return cap

    raise RuntimeError(f"Cannot open camera {camera_index} after {retries} attempts")


def capture_single_frame(camera_index: int = 0) -> str:
    """
    Capture a single frame on explicit user request (Snap button).
    Uses retry logic and longer warm-up for Windows MSMF stability.
    """
    cap = _open_camera_with_retry(camera_index)

    # Extended warm-up — MSMF needs more frames to stabilise
    print("[Capture] Warming up camera...")
    for i in range(20):
        ret, _ = cap.read()
        if not ret and i > 5:
            print(f"[Capture] Warm-up frame {i} failed, continuing...")
        time.sleep(0.05)

    best, report = select_best_frame(cap, FRAME_CANDIDATES)
    cap.release()

    if best is None:
        raise RuntimeError("Failed to read any frames from camera")

    cleaned, rpt, msg = validate_and_clean(best, report)
    if cleaned is None:
        raise RuntimeError(f"Quality gate failed: {msg}")

    fname     = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    save_path = str(SAVE_DIR / fname)
    cv2.imwrite(save_path, cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[Capture] Saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────
#  Public API — MJPEG Stream (FastAPI)
# ─────────────────────────────────────────────

def generate_frames(camera_index: int = 0):
    try:
        cap = _open_camera_with_retry(camera_index)
    except RuntimeError:
        err = np.zeros((240, 480, 3), dtype=np.uint8)
        cv2.putText(err, "Camera not available", (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 60, 255), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        return

    failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            failures += 1
            if failures > 10:
                break
            time.sleep(0.05);  continue

        failures = 0
        try:
            report        = assess_quality(frame)
            _, _, eye_box = detect_eye(frame)
            display       = apply_retinal_overlay(frame.copy(), report, eye_box)
        except Exception:
            display = frame.copy()

        ok, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue

        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

    cap.release()


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  RetinaRisk AI — Intelligent Retinal Capture")
    print("=" * 55)
    result = capture_retinal_image()
    print(f"\n{'Saved: ' + result if result else 'No image captured.'}")