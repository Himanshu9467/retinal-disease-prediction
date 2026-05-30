"""
database.py — SQLite Database Layer
Tables: User | Patient | ClinicalData | Model | Prediction
Matches ER Diagram from Project Report (Fig 4.7)
"""

import sqlite3
import hashlib
import os
import hmac
import secrets
from pathlib import Path
from datetime import datetime

DB_PATH = os.environ.get("CVD_DB_PATH", "./database/cvd_system.db")


def get_connection():
    os.makedirs(Path(DB_PATH).parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ─────────────────────────────────────────────
#  Schema Creation
# ─────────────────────────────────────────────

def init_db():
    conn = get_connection()
    c    = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS User (
        UserID      INTEGER PRIMARY KEY AUTOINCREMENT,
        Name        TEXT    NOT NULL,
        Email       TEXT    UNIQUE NOT NULL,
        Password    TEXT    NOT NULL,
        Role        TEXT    NOT NULL CHECK(Role IN ('Doctor', 'Medical Staff')),
        CreatedAt   TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS Patient (
        PatientID   INTEGER PRIMARY KEY AUTOINCREMENT,
        Name        TEXT    NOT NULL,
        Age         INTEGER,
        Gender      TEXT    CHECK(Gender IN ('Male','Female','Other')),
        Email       TEXT    NOT NULL,
        Phone       TEXT    NOT NULL DEFAULT '',
        UserID      INTEGER REFERENCES User(UserID) ON DELETE SET NULL,
        CreatedAt   TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS ClinicalData (
        RecordID        INTEGER PRIMARY KEY AUTOINCREMENT,
        PatientID       INTEGER NOT NULL REFERENCES Patient(PatientID) ON DELETE CASCADE,
        Symptoms        TEXT,
        MedicalHistory  TEXT,
        BloodPressure   TEXT,
        Cholesterol     REAL,
        ImagePath       TEXT,
        CreatedAt       TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS Model (
        ModelID         INTEGER PRIMARY KEY AUTOINCREMENT,
        AlgorithmName   TEXT    NOT NULL,
        ModelVersion    TEXT,
        Accuracy        REAL,
        Precision       REAL,
        Recall          REAL,
        F1Score         REAL,
        CheckpointPath  TEXT,
        CreatedAt       TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS Prediction (
        PredictionID    INTEGER PRIMARY KEY AUTOINCREMENT,
        PatientID       INTEGER NOT NULL REFERENCES Patient(PatientID) ON DELETE CASCADE,
        ModelID         INTEGER REFERENCES Model(ModelID),
        PredictionResult TEXT   NOT NULL CHECK(PredictionResult IN ('Normal','At-Risk','Disease Detected')),
        ConfidenceScore REAL    NOT NULL,
        ExplanationPath TEXT,
        Timestamp       TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS PatientActionChecklist (
        PatientID           INTEGER PRIMARY KEY REFERENCES Patient(PatientID) ON DELETE CASCADE,
        FollowUpAdvised     INTEGER NOT NULL DEFAULT 0,
        DoctorReviewed      INTEGER NOT NULL DEFAULT 0,
        LifestyleCounseling INTEGER NOT NULL DEFAULT 0,
        RescreenScheduled   INTEGER NOT NULL DEFAULT 0,
        UpdatedAt           TEXT    DEFAULT (datetime('now'))
    );
    """)

    columns = {row["name"] for row in c.execute("PRAGMA table_info(Patient)").fetchall()}
    if "Phone" not in columns:
        c.execute("ALTER TABLE Patient ADD COLUMN Phone TEXT NOT NULL DEFAULT ''")

    conn.commit()
    conn.close()
    print(f"[DB] Initialised at {DB_PATH}")


# ─────────────────────────────────────────────
#  User Operations
# ─────────────────────────────────────────────

def _hash_password(pwd):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 210_000)
    return f"pbkdf2_sha256$210000${salt}${digest.hex()}"


def _legacy_hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()


def _verify_password(pwd, stored_hash):
    if not stored_hash:
        return False, False
    if str(stored_hash).startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt, digest = str(stored_hash).split("$", 3)
            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                pwd.encode(),
                salt.encode(),
                int(rounds),
            ).hex()
            return hmac.compare_digest(candidate, digest), False
        except (TypeError, ValueError):
            return False, False
    return hmac.compare_digest(_legacy_hash_password(pwd), str(stored_hash)), True


def create_user(name, email, password, role):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO User (Name, Email, Password, Role) VALUES (?,?,?,?)",
            (name, email, _hash_password(password), role)
        )
        conn.commit()
        return {"success": True, "message": f"User '{name}' created."}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Email already registered."}
    finally:
        conn.close()


def authenticate_user(email, password):
    conn = get_connection()
    row = conn.execute("SELECT * FROM User WHERE Email=?", (email,)).fetchone()
    if not row:
        conn.close()
        return {"success": False, "message": "Invalid credentials."}

    verified, needs_upgrade = _verify_password(password, row["Password"])
    if not verified:
        conn.close()
        return {"success": False, "message": "Invalid credentials."}

    if needs_upgrade:
        conn.execute(
            "UPDATE User SET Password=? WHERE UserID=?",
            (_hash_password(password), row["UserID"]),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM User WHERE UserID=?", (row["UserID"],)).fetchone()

    conn.close()
    return {"success": True, "user": dict(row)}


def get_all_users():
    conn = get_connection()
    rows = conn.execute("SELECT UserID, Name, Email, Role, CreatedAt FROM User").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
#  Patient Operations
# ─────────────────────────────────────────────

def create_patient(name, age, gender, email, phone, user_id=None):
    conn = get_connection()
    cur  = conn.execute(
        "INSERT INTO Patient (Name, Age, Gender, Email, Phone, UserID) VALUES (?,?,?,?,?,?)",
        (name, age, gender, email, phone, user_id)
    )
    conn.commit()
    patient_id = cur.lastrowid
    conn.close()
    return {"success": True, "PatientID": patient_id}


def get_patient(patient_id):
    conn = get_connection()
    row  = conn.execute("SELECT * FROM Patient WHERE PatientID=?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_patients():
    conn = get_connection()
    rows = conn.execute(
        "SELECT p.*, COUNT(pr.PredictionID) as PredictionCount "
        "FROM Patient p LEFT JOIN Prediction pr ON p.PatientID=pr.PatientID "
        "GROUP BY p.PatientID ORDER BY p.CreatedAt DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_patients(query):
    q = f"%{str(query or '').strip().lower()}%"
    conn = get_connection()
    rows = conn.execute(
        "SELECT p.*, COUNT(pr.PredictionID) as PredictionCount "
        "FROM Patient p LEFT JOIN Prediction pr ON p.PatientID=pr.PatientID "
        "WHERE lower(p.Name) LIKE ? OR CAST(p.PatientID AS TEXT) LIKE ? "
        "OR lower(COALESCE(p.Email, '')) LIKE ? OR lower(COALESCE(p.Phone, '')) LIKE ? "
        "GROUP BY p.PatientID ORDER BY p.CreatedAt DESC",
        (q, q, q, q),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_patient(patient_id, name, age, gender, email, phone):
    conn = get_connection()
    row = conn.execute(
        "SELECT PatientID FROM Patient WHERE PatientID=?",
        (patient_id,)
    ).fetchone()
    if not row:
        conn.close()
        return {"success": False, "updated": 0}

    conn.execute(
        "UPDATE Patient SET Name=?, Age=?, Gender=?, Email=?, Phone=? WHERE PatientID=?",
        (name, age, gender, email, phone, patient_id)
    )
    conn.commit()
    conn.close()
    return {"success": True, "updated": 1}


def delete_patient(patient_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT PatientID FROM Patient WHERE PatientID=?",
        (patient_id,)
    ).fetchone()
    if not row:
        conn.close()
        return {"success": False, "deleted": 0}

    conn.execute("DELETE FROM Patient WHERE PatientID=?", (patient_id,))
    conn.commit()
    conn.close()
    return {"success": True, "deleted": 1}


# ─────────────────────────────────────────────
#  Clinical Data Operations
# ─────────────────────────────────────────────

def add_clinical_data(patient_id, image_path, symptoms=None,
                      medical_history=None, blood_pressure=None, cholesterol=None):
    conn = get_connection()
    cur  = conn.execute(
        "INSERT INTO ClinicalData (PatientID, Symptoms, MedicalHistory, "
        "BloodPressure, Cholesterol, ImagePath) VALUES (?,?,?,?,?,?)",
        (patient_id, symptoms, medical_history, blood_pressure, cholesterol, image_path)
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return {"success": True, "RecordID": record_id}


# ─────────────────────────────────────────────
#  Model Registry Operations
# ─────────────────────────────────────────────

def register_model(algorithm_name, model_version, accuracy, precision,
                   recall, f1_score, checkpoint_path):
    conn = get_connection()
    cur  = conn.execute(
        "INSERT INTO Model (AlgorithmName, ModelVersion, Accuracy, Precision, "
        "Recall, F1Score, CheckpointPath) VALUES (?,?,?,?,?,?,?)",
        (algorithm_name, model_version, accuracy, precision,
         recall, f1_score, checkpoint_path)
    )
    conn.commit()
    model_id = cur.lastrowid
    conn.close()
    return {"success": True, "ModelID": model_id}


def upsert_model(algorithm_name, model_version, accuracy, precision,
                 recall, f1_score, checkpoint_path):
    conn = get_connection()
    existing = conn.execute(
        "SELECT ModelID FROM Model WHERE AlgorithmName=? AND ModelVersion=?",
        (algorithm_name, model_version)
    ).fetchone()

    if existing:
        model_id = existing["ModelID"]
        conn.execute(
            "UPDATE Model SET Accuracy=?, Precision=?, Recall=?, F1Score=?, CheckpointPath=?, CreatedAt=datetime('now') "
            "WHERE ModelID=?",
            (accuracy, precision, recall, f1_score, checkpoint_path, model_id)
        )
        action = "updated"
    else:
        cur = conn.execute(
            "INSERT INTO Model (AlgorithmName, ModelVersion, Accuracy, Precision, "
            "Recall, F1Score, CheckpointPath) VALUES (?,?,?,?,?,?,?)",
            (algorithm_name, model_version, accuracy, precision,
             recall, f1_score, checkpoint_path)
        )
        model_id = cur.lastrowid
        action = "created"

    conn.commit()
    conn.close()
    return {"success": True, "ModelID": model_id, "action": action}


def get_best_model():
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM Model ORDER BY F1Score DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_model_by_name(algorithm_name):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM Model WHERE lower(AlgorithmName)=lower(?) "
        "ORDER BY CreatedAt DESC LIMIT 1",
        (algorithm_name,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_models():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM Model ORDER BY CreatedAt DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def merge_model_records(source_model_id, target_model_id):
    """
    Move any predictions pointing to source_model_id into target_model_id,
    then delete the source model row.
    """
    if source_model_id == target_model_id:
        return {"success": True, "merged": False}

    conn = get_connection()
    conn.execute(
        "UPDATE Prediction SET ModelID=? WHERE ModelID=?",
        (target_model_id, source_model_id)
    )
    conn.execute("DELETE FROM Model WHERE ModelID=?", (source_model_id,))
    conn.commit()
    conn.close()
    return {"success": True, "merged": True}


# ─────────────────────────────────────────────
#  Prediction Operations
# ─────────────────────────────────────────────

def save_prediction(patient_id, model_id, prediction_result,
                    confidence_score, explanation_path=None):
    conn = get_connection()
    cur  = conn.execute(
        "INSERT INTO Prediction (PatientID, ModelID, PredictionResult, "
        "ConfidenceScore, ExplanationPath) VALUES (?,?,?,?,?)",
        (patient_id, model_id, prediction_result, confidence_score, explanation_path)
    )
    conn.commit()
    pred_id = cur.lastrowid
    conn.close()
    return {"success": True, "PredictionID": pred_id}


def get_patient_predictions(patient_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT pr.*, m.AlgorithmName, m.ModelVersion, cd.ImagePath "
        "FROM Prediction pr LEFT JOIN Model m ON pr.ModelID=m.ModelID "
        "LEFT JOIN ClinicalData cd ON cd.RecordID = ("
        "  SELECT cd2.RecordID FROM ClinicalData cd2 "
        "  WHERE cd2.PatientID=pr.PatientID AND cd2.CreatedAt <= pr.Timestamp "
        "  ORDER BY cd2.CreatedAt DESC, cd2.RecordID DESC LIMIT 1"
        ") "
        "WHERE pr.PatientID=? ORDER BY pr.Timestamp DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_predictions(limit=100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT pr.*, p.Name as PatientName, p.Age as PatientAge, p.Email as PatientEmail, "
        "p.Phone as PatientPhone, m.AlgorithmName, cd.ImagePath "
        "FROM Prediction pr "
        "LEFT JOIN Patient p ON pr.PatientID=p.PatientID "
        "LEFT JOIN Model m   ON pr.ModelID=m.ModelID "
        "LEFT JOIN ClinicalData cd ON cd.RecordID = ("
        "  SELECT cd2.RecordID FROM ClinicalData cd2 "
        "  WHERE cd2.PatientID=pr.PatientID AND cd2.CreatedAt <= pr.Timestamp "
        "  ORDER BY cd2.CreatedAt DESC, cd2.RecordID DESC LIMIT 1"
        ") "
        "ORDER BY pr.Timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction(prediction_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT pr.*, p.Name as PatientName, m.AlgorithmName "
        "FROM Prediction pr "
        "LEFT JOIN Patient p ON pr.PatientID=p.PatientID "
        "LEFT JOIN Model m ON pr.ModelID=m.ModelID "
        "WHERE pr.PredictionID=?",
        (prediction_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_prediction(prediction_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT PredictionID FROM Prediction WHERE PredictionID=?",
        (prediction_id,)
    ).fetchone()
    if not row:
        conn.close()
        return {"success": False, "deleted": 0}

    conn.execute("DELETE FROM Prediction WHERE PredictionID=?", (prediction_id,))
    conn.commit()
    conn.close()
    return {"success": True, "deleted": 1}


def delete_all_predictions():
    conn = get_connection()
    cur = conn.execute("DELETE FROM Prediction")
    deleted = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    conn.close()
    return {"success": True, "deleted": deleted}


def get_prediction_stats():
    conn = get_connection()
    stats = {}
    for result in ["Normal", "At-Risk", "Disease Detected"]:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM Prediction WHERE PredictionResult=?",
            (result,)
        ).fetchone()
        stats[result] = row["cnt"]
    stats["total"] = sum(stats.values())
    conn.close()
    return stats


def get_patient_checklist(patient_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM PatientActionChecklist WHERE PatientID=?",
        (patient_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {
            "PatientID": patient_id,
            "FollowUpAdvised": 0,
            "DoctorReviewed": 0,
            "LifestyleCounseling": 0,
            "RescreenScheduled": 0,
        }
    return dict(row)


def save_patient_checklist(patient_id, checklist):
    conn = get_connection()
    patient = conn.execute(
        "SELECT PatientID FROM Patient WHERE PatientID=?",
        (patient_id,)
    ).fetchone()
    if not patient:
        conn.close()
        return {"success": False}

    values = {
        "FollowUpAdvised": int(bool(checklist.get("FollowUpAdvised"))),
        "DoctorReviewed": int(bool(checklist.get("DoctorReviewed"))),
        "LifestyleCounseling": int(bool(checklist.get("LifestyleCounseling"))),
        "RescreenScheduled": int(bool(checklist.get("RescreenScheduled"))),
    }
    conn.execute(
        "INSERT INTO PatientActionChecklist "
        "(PatientID, FollowUpAdvised, DoctorReviewed, LifestyleCounseling, RescreenScheduled, UpdatedAt) "
        "VALUES (?, ?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT(PatientID) DO UPDATE SET "
        "FollowUpAdvised=excluded.FollowUpAdvised, "
        "DoctorReviewed=excluded.DoctorReviewed, "
        "LifestyleCounseling=excluded.LifestyleCounseling, "
        "RescreenScheduled=excluded.RescreenScheduled, "
        "UpdatedAt=datetime('now')",
        (
            patient_id,
            values["FollowUpAdvised"],
            values["DoctorReviewed"],
            values["LifestyleCounseling"],
            values["RescreenScheduled"],
        ),
    )
    conn.commit()
    conn.close()
    return {"success": True, **values}


if __name__ == "__main__":
    init_db()
    # Seed demo users
    create_user("Dr. Sharma", "doctor@rit.edu", "doctor123", "Doctor")
    create_user("Nurse Priya", "staff@rit.edu", "staff123",  "Medical Staff")
    print("[DB] Demo users seeded.")
    print("[DB] Users:", get_all_users())
