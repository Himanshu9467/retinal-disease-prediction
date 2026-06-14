import {
  Suspense,
  createElement,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  ClipboardList,
  Eye,
  EyeOff,
  FileClock,
  History,
  LineChart,
  Lock,
  LogOut,
  Mail,
  Moon,
  Pill,
  Phone,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Stethoscope,
  Sun,
  Trash2,
  Users,
} from "lucide-react";
import FilterPanel from "./components/dashboard/FilterPanel";
import NotificationToast from "./components/dashboard/NotificationToast";
import PatientDrawer from "./components/dashboard/PatientDrawer";
import "./App.css";

const ConfidenceChart = lazy(() => import("./components/dashboard/ConfidenceChart"));
const RiskTimelineChart = lazy(() => import("./components/dashboard/RiskTimelineChart"));

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";
const SESSION_KEY = "retina-dashboard-session";
const THEME_KEY = "retina-dashboard-theme";

const RISK_META = {
  Normal: { tone: "normal", priority: 1 },
  Disease: { tone: "critical", priority: 2 },
};

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "screening", label: "Screening", icon: Eye },
  { id: "patients", label: "Patients", icon: Users },
  { id: "history", label: "History", icon: History },
  { id: "analytics", label: "Analytics", icon: LineChart },
  { id: "medications", label: "Guidance", icon: Pill },
];

const CARE_GUIDANCE = {
  Disease: {
    title: "First Phase Response",
    urgency: "Arrange same-day clinical review and confirm the finding with a doctor.",
    steps: [
      "Check blood pressure, pulse, oxygen saturation, blood sugar, and current symptoms.",
      "Review patient history for diabetes, hypertension, cholesterol, smoking, chest pain, stroke symptoms, or kidney disease.",
      "Request physician-led follow-up such as ECG, lipid profile, fasting glucose or HbA1c, and retinal/ophthalmology review when indicated.",
      "Escalate immediately if the patient reports chest pain, severe breathlessness, fainting, weakness on one side, facial droop, speech trouble, or sudden vision loss.",
    ],
    precautions: [
      "Do not ignore the result even if the patient feels normal; retinal disease signs can be silent.",
      "Avoid heavy exertion until a clinician reviews symptoms and vital signs.",
      "Continue prescribed medicines unless the treating doctor changes them.",
      "Avoid smoking, alcohol excess, high-salt meals, and fried or high saturated-fat foods.",
    ],
    foodsTitle: "Food Pattern Until Review",
    foods: [
      "Vegetables, fruits, beans/lentils, oats, whole grains, nuts, and seeds.",
      "Fish or lean protein where appropriate, and low-fat dairy if tolerated.",
      "Low-salt meals; avoid processed snacks, sugary drinks, bakery items, and deep-fried foods.",
    ],
  },
  Normal: {
    title: "Maintenance Plan",
    urgency: "Continue healthy habits and repeat screening as advised.",
    steps: [
      "Keep routine checks for blood pressure, cholesterol, blood sugar, and eye health.",
      "Maintain regular activity, balanced meals, sleep, and follow-up if symptoms appear.",
    ],
    precautions: [
      "A normal screening result is not a guarantee of zero risk.",
      "Seek care if chest pain, breathlessness, stroke-like symptoms, or sudden vision changes occur.",
    ],
    foodsTitle: "Keep Prioritizing",
    foods: [
      "Vegetables, fruits, whole grains, legumes, lean proteins, and low-salt meals.",
      "Limit processed foods, sugary drinks, smoking/tobacco, and alcohol excess.",
    ],
  },
};

function readStoredSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed?.token && parsed?.user) return parsed;
    return null;
  } catch {
    localStorage.removeItem(SESSION_KEY);
    return null;
  }
}

function readStoredTheme() {
  try {
    return localStorage.getItem(THEME_KEY) === "night" ? "night" : "day";
  } catch {
    return "day";
  }
}

async function apiRequest(path, options = {}, token = null) {
  const headers = { ...(options.headers || {}) };
  const isFormData = options.body instanceof FormData;
  let requestBody = options.body;

  if (token) headers.Authorization = `Bearer ${token}`;
  if (!isFormData && requestBody !== undefined && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (!isFormData && requestBody && typeof requestBody !== "string") {
    requestBody = JSON.stringify(requestBody);
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
      body: requestBody,
    });
  } catch {
    throw new Error(
      `Backend API is not reachable at ${API_BASE}. Start the FastAPI server, then try again.`
    );
  }

  const raw = await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const message =
      payload?.detail ||
      payload?.message ||
      payload?.error ||
      `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

function parseApiDate(value) {
  if (!value) return null;
  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function dayKey(date) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function riskTone(value) {
  return RISK_META[value]?.tone || "neutral";
}

function formatKeyLabel(rawKey) {
  return String(rawKey || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function confidenceBand(value) {
  const score = Number(value || 0);
  if (score >= 0.8) return "high";
  if (score >= 0.6) return "medium";
  return "low";
}

function compactModelSource(path) {
  if (!path) return "--";
  const parts = String(path).split(/[\\/]+/);
  const outputIndex = parts.findIndex((part) => part === "research_training_outputs2");
  return outputIndex >= 0 ? parts.slice(outputIndex).join("/") : parts.slice(-2).join("/");
}

function withinTimeFilter(timestamp, filter) {
  if (filter === "all") return true;
  const parsed = parseApiDate(timestamp);
  if (!parsed) return false;
  const now = new Date();
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  if (filter === "week") {
    start.setDate(now.getDate() - now.getDay());
  }
  if (filter === "month") {
    start.setDate(1);
  }
  return parsed >= start;
}

function riskTrend(predictions) {
  const now = new Date();
  const thisWeekStart = new Date(now);
  thisWeekStart.setHours(0, 0, 0, 0);
  thisWeekStart.setDate(now.getDate() - now.getDay());
  const lastWeekStart = new Date(thisWeekStart);
  lastWeekStart.setDate(thisWeekStart.getDate() - 7);

  const score = (row) => RISK_META[row.PredictionResult]?.priority || 0;
  const average = (rows) =>
    rows.length ? rows.reduce((sum, row) => sum + score(row), 0) / rows.length : 0;
  const thisWeek = predictions.filter((row) => {
    const date = parseApiDate(row.Timestamp);
    return date && date >= thisWeekStart;
  });
  const lastWeek = predictions.filter((row) => {
    const date = parseApiDate(row.Timestamp);
    return date && date >= lastWeekStart && date < thisWeekStart;
  });
  const current = average(thisWeek);
  const previous = average(lastWeek);
  const delta = previous ? ((current - previous) / previous) * 100 : current ? 100 : 0;
  return {
    delta,
    direction: delta > 0 ? "up" : delta < 0 ? "down" : "flat",
    label: `${Math.abs(delta).toFixed(1)}%`,
  };
}

function RiskPill({ value }) {
  if (!value) return null;
  return <span className={`risk-pill ${riskTone(value)}`}>{value}</span>;
}

function PanelHeader({ title, subtitle, actions = null }) {
  return (
    <div className="panel-header">
      <div>
        <h3>{title}</h3>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {actions}
    </div>
  );
}

function StatTile({ label, value, helper, icon: Icon = Activity, tone = "neutral" }) {
  return (
    <article className={`stat-tile ${tone}`}>
      <div className="stat-icon" aria-hidden="true">
        {createElement(Icon, { size: 18 })}
      </div>
      <div>
        <p>{label}</p>
        <h4>{value}</h4>
        {helper ? <span>{helper}</span> : null}
      </div>
    </article>
  );
}

function CareGuidancePanel({ risk }) {
  const guidance = CARE_GUIDANCE[risk] || CARE_GUIDANCE.Normal;

  return (
    <article className={`care-guidance ${riskTone(risk)}`}>
      <div className="care-guidance-head">
        <div>
          <span>Patient Guidance</span>
          <h4>{guidance.title}</h4>
        </div>
        <RiskPill value={risk} />
      </div>
      <p className="care-urgency">{guidance.urgency}</p>

      <div className="care-grid">
        <div>
          <h5>First Steps</h5>
          <ul>
            {guidance.steps.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h5>{guidance.foodsTitle}</h5>
          <ul>
            {guidance.foods.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="full">
          <h5>Precautions</h5>
          <ul>
            {guidance.precautions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>

      <p className="care-note">
        Educational support only. Final advice, medicines, tests, and diet changes should
        be confirmed by a qualified clinician.
      </p>
    </article>
  );
}

function AuthView({ onLogin, notice = "" }) {
  const [mode, setMode] = useState("login");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);

  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [registerForm, setRegisterForm] = useState({
    name: "",
    email: "",
    password: "",
    role: "Doctor",
  });

  const submitLogin = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");
    setBusy(true);
    try {
      const response = await apiRequest("/auth/login", {
        method: "POST",
        body: loginForm,
      });
      onLogin(response.access_token, response.user, rememberMe);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  const submitRegister = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");
    setBusy(true);
    try {
      await apiRequest("/auth/register", {
        method: "POST",
        body: registerForm,
      });
      setMode("login");
      setLoginForm((prev) => ({ ...prev, email: registerForm.email }));
      setSuccess("Account created. Sign in with your new clinical profile.");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-shell">
      <section className="auth-experience" aria-label="CardioVision AI authentication">
        <div className="auth-hero">
          <div className="auth-brand">
            <div className="auth-brand-mark" aria-hidden="true">
              <Eye size={24} />
            </div>
            <div>
              <span>CardioVision AI</span>
              <strong>Early Detection Through Retinal Intelligence</strong>
            </div>
          </div>

          <div className="auth-hero-copy">
            <span className="auth-eyebrow">Clinical AI Platform</span>
            <h1>CardioVision AI</h1>
            <p>
              AI-Powered Cardiovascular Disease Risk Assessment Through Retinal Image Analysis
            </p>
          </div>

          <div className="auth-platform-preview" aria-label="AI platform preview">
            <div className="fundus-preview">
              <div className="fundus-image" aria-hidden="true">
                <span className="optic-disc" />
                <span className="fundus-vessel vessel-a" />
                <span className="fundus-vessel vessel-b" />
                <span className="fundus-vessel vessel-c" />
                <span className="gradcam-heat heat-a" />
                <span className="gradcam-heat heat-b" />
                <span className="scan-beam" />
              </div>
              <div className="fundus-caption">
                <span>Retina Fundus + GradCAM</span>
                <strong>Explainable screening preview</strong>
              </div>
            </div>

            <div className="auth-ai-grid">
              <div className="ai-widget risk-score">
                <span>Disease Risk Score</span>
                <strong>0.31</strong>
                <small>Normal band</small>
              </div>
              <div className="ai-widget confidence-score">
                <span>Model Confidence</span>
                <strong>95.14%</strong>
                <small>EfficientNet-B3</small>
              </div>
              <div className="ai-widget screening-status">
                <span>AI Screening</span>
                <strong>Active</strong>
                <small>Real-time inference ready</small>
              </div>
              <div className="ai-widget analytics-widget">
                <span>Clinical AI Analytics</span>
                <div className="mini-bars" aria-hidden="true">
                  <i style={{ height: "42%" }} />
                  <i style={{ height: "68%" }} />
                  <i style={{ height: "54%" }} />
                  <i style={{ height: "82%" }} />
                </div>
              </div>
            </div>

            <div className="auth-model-strip">
              <div>
                <span>Production Model</span>
                <strong>EfficientNet-B3</strong>
              </div>
              <div>
                <span>Dataset</span>
                <strong>7000 Patient-Safe Images</strong>
              </div>
            </div>

            <div className="auth-model-list" aria-label="Deep learning model information">
              {["EfficientNet-B3", "ResNet50", "MobileNetV3-Large", "CNN", "Ensemble"].map(
                (model) => (
                  <span key={model}>{model}</span>
                )
              )}
            </div>
          </div>
        </div>

        <section className="auth-card" aria-label="Sign in panel">
          <header className="auth-title">
            <span className="secure-access-badge">
              <ShieldCheck size={14} />
              Secure Clinical Access
            </span>
            <h2>{mode === "login" ? "Welcome back" : "Create clinical profile"}</h2>
            <p>
              Advanced Deep Learning Platform for Cardiovascular Disease Risk Assessment
              Using Retinal Fundus Imaging
            </p>
          </header>
          {notice ? <div className="form-error auth-notice">{notice}</div> : null}
          {success ? <div className="form-success auth-notice">{success}</div> : null}

          <div className="auth-mode">
            <button
              className={mode === "login" ? "active" : ""}
              onClick={() => {
                setMode("login");
                setError("");
              }}
              type="button"
            >
              Sign In
            </button>
            <button
              className={mode === "register" ? "active" : ""}
              onClick={() => {
                setMode("register");
                setError("");
              }}
              type="button"
            >
              Register
            </button>
          </div>

          {mode === "login" ? (
            <form onSubmit={submitLogin} className="form-grid auth-form">
              <label>
                <span>Email</span>
                <div className="auth-input">
                  <Mail size={17} />
                  <input
                    type="email"
                    autoComplete="email"
                    placeholder="clinician@hospital.org"
                    value={loginForm.email}
                    onChange={(event) =>
                      setLoginForm((prev) => ({ ...prev, email: event.target.value }))
                    }
                    required
                  />
                </div>
              </label>
              <label>
                <span>Password</span>
                <div className="auth-input">
                  <Lock size={17} />
                  <input
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    placeholder="Enter secure password"
                    value={loginForm.password}
                    onChange={(event) =>
                      setLoginForm((prev) => ({
                        ...prev,
                        password: event.target.value,
                      }))
                    }
                    required
                  />
                  <button
                    className="password-toggle"
                    onClick={() => setShowPassword((prev) => !prev)}
                    type="button"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                    title={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </div>
              </label>
              <div className="auth-options">
                <label className="remember-control">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(event) => setRememberMe(event.target.checked)}
                  />
                  <span>Remember me</span>
                </label>
                <button
                  className="forgot-link"
                  onClick={() =>
                    setError("Password recovery is handled by your system administrator.")
                  }
                  type="button"
                >
                  Forgot Password
                </button>
              </div>
              {error ? <div className="form-error">{error}</div> : null}
              <button className="btn btn-primary auth-submit" disabled={busy} type="submit">
                {busy ? (
                  <>
                    <RefreshCw size={17} className="spin" />
                    Signing in...
                  </>
                ) : (
                  <>
                    <ShieldCheck size={17} />
                    Sign In
                  </>
                )}
              </button>
            </form>
          ) : (
            <form onSubmit={submitRegister} className="form-grid auth-form">
              <label>
                <span>Full Name</span>
                <div className="auth-input">
                  <Stethoscope size={17} />
                  <input
                    autoComplete="name"
                    placeholder="Dr. Anika Sharma"
                    value={registerForm.name}
                    onChange={(event) =>
                      setRegisterForm((prev) => ({ ...prev, name: event.target.value }))
                    }
                    required
                  />
                </div>
              </label>
              <label>
                <span>Email</span>
                <div className="auth-input">
                  <Mail size={17} />
                  <input
                    type="email"
                    autoComplete="email"
                    placeholder="clinician@hospital.org"
                    value={registerForm.email}
                    onChange={(event) =>
                      setRegisterForm((prev) => ({ ...prev, email: event.target.value }))
                    }
                    required
                  />
                </div>
              </label>
              <label>
                <span>Password</span>
                <div className="auth-input">
                  <Lock size={17} />
                  <input
                    type="password"
                    autoComplete="new-password"
                    placeholder="Create secure password"
                    value={registerForm.password}
                    onChange={(event) =>
                      setRegisterForm((prev) => ({
                        ...prev,
                        password: event.target.value,
                      }))
                    }
                    required
                  />
                </div>
              </label>
              <label>
                <span>Role</span>
                <div className="auth-input select-input">
                  <ShieldCheck size={17} />
                  <select
                    value={registerForm.role}
                    onChange={(event) =>
                      setRegisterForm((prev) => ({ ...prev, role: event.target.value }))
                    }
                  >
                    <option value="Doctor">Doctor</option>
                    <option value="Medical Staff">Medical Staff</option>
                  </select>
                </div>
              </label>
              {error ? <div className="form-error">{error}</div> : null}
              <button className="btn btn-primary auth-submit" disabled={busy} type="submit">
                {busy ? (
                  <>
                    <RefreshCw size={17} className="spin" />
                    Creating account...
                  </>
                ) : (
                  <>
                    <ShieldCheck size={17} />
                    Create Account
                  </>
                )}
              </button>
            </form>
          )}
        </section>
      </section>
    </main>
  );
}

function LegacyDashboardView({ patients, predictions, stats, onNavigate, onRefresh, refreshing }) {
  const highRiskCount = stats?.Disease ?? 0;

  const weeklyTrend = useMemo(() => {
    const buckets = Array.from({ length: 7 }, (_, idx) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - idx));
      return { key: dayKey(date), count: 0 };
    });
    const index = new Map(buckets.map((entry, idx) => [entry.key, idx]));
    predictions.forEach((row) => {
      const timestamp = parseApiDate(row.Timestamp);
      if (!timestamp) return;
      const idx = index.get(dayKey(timestamp));
      if (idx !== undefined) buckets[idx].count += 1;
    });
    return buckets;
  }, [predictions]);

  const highRiskQueue = useMemo(
    () =>
      [...predictions]
        .filter((row) => row.PredictionResult !== "Normal")
        .sort((a, b) => {
          const diff =
            (RISK_META[b.PredictionResult]?.priority || 0) -
            (RISK_META[a.PredictionResult]?.priority || 0);
          if (diff !== 0) return diff;
          return (
            (parseApiDate(b.Timestamp)?.getTime() || 0) -
            (parseApiDate(a.Timestamp)?.getTime() || 0)
          );
        })
        .slice(0, 6),
    [predictions]
  );

  const patientCards = useMemo(() => {
    const latestByPatient = new Map();
    predictions.forEach((row) => {
      const key = String(row.PatientID || "");
      if (!key) return;
      const current = latestByPatient.get(key);
      const nextTime = parseApiDate(row.Timestamp)?.getTime() || 0;
      const currentTime = parseApiDate(current?.Timestamp)?.getTime() || 0;
      if (!current || nextTime > currentTime) latestByPatient.set(key, row);
    });

    const sourcePatients =
      patients.length > 0
        ? patients
        : predictions.slice(0, 6).map((row) => ({
            PatientID: row.PatientID,
            Name: row.PatientName || `Patient ${row.PatientID}`,
          }));

    return sourcePatients.slice(0, 7).map((patient, index) => {
      const latest = latestByPatient.get(String(patient.PatientID));
      const name = patient.Name || latest?.PatientName || `Patient ${patient.PatientID}`;
      const initials = name
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0]?.toUpperCase())
        .join("");
      const risk = latest?.PredictionResult || "Not screened";
      return {
        id: patient.PatientID || index,
        name,
        initials: initials || "P",
        age: patient.Age ? `${patient.Age}y` : `${72 + index}y`,
        room: patient.Room || `Room ${101 + index}`,
        risk,
      };
    });
  }, [patients, predictions]);

  const chartPoints = weeklyTrend.map((entry, index) => ({
    x: 10 + index * 15,
    y: 68 - Math.min(42, entry.count * 7 + (index % 3) * 4),
  }));
  const heartPath =
    chartPoints.length > 0
      ? chartPoints
          .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
          .join(" ")
      : "M 10 52 L 25 58 L 40 56 L 55 60 L 70 48 L 85 45 L 100 53";

  const activities = [
    predictions[0]
      ? {
          title: "Prediction",
          body: `#${predictions[0].PredictionID} saved`,
          time:
            parseApiDate(predictions[0].Timestamp)?.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }) || "Just now",
        }
      : { title: "Order", body: "No activity yet", time: "Now" },
    highRiskQueue[0]
      ? {
          title: "Alert",
          body: `${highRiskQueue[0].PatientName || "Patient"} added to review`,
          time: "Recent",
        }
      : null,
  ].filter(Boolean);

  return (
    <section className="care-dashboard">
      <aside className="patient-rail">
        <div className="rail-heading">
          <h3>Patients</h3>
          <span>{patients.length || patientCards.length} total</span>
        </div>
        <button className="care-add-btn" type="button" onClick={() => onNavigate("patients")}>
          <Plus size={16} />
          Add Patient
        </button>
        <div className="patient-list">
          {patientCards.length === 0 ? (
            <p className="empty">No patients yet.</p>
          ) : (
            patientCards.map((patient, index) => (
              <button
                className={`patient-card ${index === 0 ? "selected" : ""}`}
                key={patient.id}
                type="button"
              >
                <span className={`avatar ${riskTone(patient.risk)}`}>{patient.initials}</span>
                <span>
                  <strong>{patient.name}</strong>
                  <small>
                    {patient.room} · {patient.age}
                  </small>
                </span>
                <i className={`patient-status ${riskTone(patient.risk)}`} />
              </button>
            ))
          )}
        </div>
        <button className="sos-btn" type="button">
          <AlertTriangle size={18} />
          Trigger SOS
        </button>
      </aside>

      <div className="care-main">
        <article className="care-panel chart-panel">
          <div className="care-panel-head">
            <h3>Heart Rate History</h3>
            <button className="panel-icon-btn" onClick={onRefresh} disabled={refreshing} type="button">
              <RefreshCw size={16} />
            </button>
          </div>
          <svg className="heart-chart" viewBox="0 0 110 76" role="img" aria-label="Heart rate history">
            <path d={heartPath} fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />
          </svg>
        </article>

        <div className="care-work-grid">
          <article className="care-panel order-panel">
            <div className="care-panel-title">
              <ClipboardList size={18} />
              <h3>New Care Order</h3>
            </div>
            <div className="order-form-grid">
              <label>
                <span>Patient</span>
                <select defaultValue="">
                  <option value="" disabled>
                    Select patient...
                  </option>
                  {patientCards.map((patient) => (
                    <option key={patient.id} value={patient.id}>
                      {patient.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Type</span>
                <select defaultValue="medication">
                  <option value="medication">Medication</option>
                  <option value="screening">Screening</option>
                  <option value="followup">Follow-up</option>
                </select>
              </label>
              <label>
                <span>Priority</span>
                <select defaultValue="medium">
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="critical">Critical</option>
                </select>
              </label>
              <label>
                <span>Status</span>
                <select defaultValue="pending">
                  <option value="pending">Pending</option>
                  <option value="active">Active</option>
                </select>
              </label>
              <label className="full">
                <span>Description</span>
                <textarea placeholder="Describe..." />
              </label>
            </div>
            <button className="submit-order-btn" type="button" onClick={() => onNavigate("screening")}>
              Submit Order
            </button>
          </article>

          <article className="care-panel tasks-panel">
            <div className="care-panel-title">
              <CheckCircle2 size={18} />
              <h3>Daily Tasks</h3>
              <span>{highRiskCount}</span>
            </div>
            <div className="task-add-row">
              <input placeholder="Add task..." />
              <button type="button" aria-label="Add task">
                <Plus size={20} />
              </button>
            </div>
            <div className="task-empty">No tasks</div>
          </article>
        </div>
      </div>

      <aside className="alert-rail">
        <div className="rail-heading">
          <h3>Active Alerts</h3>
          <span className="alert-count">{highRiskQueue.length}</span>
        </div>
        <select className="alert-filter" defaultValue="all">
          <option value="all">All Patients</option>
          <option value="critical">Critical</option>
        </select>
        <div className="alert-list">
          {highRiskQueue.length === 0 ? (
            <p className="empty">No active alerts.</p>
          ) : (
            highRiskQueue.slice(0, 4).map((row) => (
              <button
                className={`alert-card ${riskTone(row.PredictionResult)}`}
                key={row.PredictionID}
                type="button"
                onClick={() => onNavigate("history")}
              >
                <strong>
                  {row.PatientName || `Patient ${row.PatientID}`} -{" "}
                  Critical
                </strong>
                <span>
                  {row.PredictionResult} · {formatPercent(Number(row.ConfidenceScore || 0))} confidence
                </span>
              </button>
            ))
          )}
        </div>

        <div className="activity-section">
          <div className="rail-heading">
            <h3>Activity Feed</h3>
            <button className="clear-feed-btn" type="button">
              Clear
            </button>
          </div>
          <div className="activity-list">
            {activities.map((item) => (
              <div className="activity-item" key={`${item.title}-${item.body}`}>
                <span />
                <div>
                  <strong>{item.title}</strong>
                  <small>{item.body}</small>
                </div>
                <time>{item.time}</time>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </section>
  );
}

function DashboardView({
  token,
  patients,
  predictions,
  onNavigate,
  onDeletePatient,
  notify,
}) {
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [alertFilter, setAlertFilter] = useState("all");
  const [filters, setFilters] = useState({
    query: "",
    risk: "all",
    time: "all",
    highConfidenceOnly: false,
  });
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [drawerPatient, setDrawerPatient] = useState(null);
  const [drawerHistory, setDrawerHistory] = useState([]);
  const [drawerChecklist, setDrawerChecklist] = useState({});
  const [drawerPrediction, setDrawerPrediction] = useState(null);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const drawerRequestRef = useRef(0);

  useEffect(() => {
    const id = setTimeout(() => setDebouncedQuery(filters.query.trim().toLowerCase()), 250);
    return () => clearTimeout(id);
  }, [filters.query]);

  const filteredPredictions = useMemo(
    () =>
      predictions.filter((row) => {
        if (filters.risk !== "all" && row.PredictionResult !== filters.risk) return false;
        if (!withinTimeFilter(row.Timestamp, filters.time)) return false;
        if (filters.highConfidenceOnly && confidenceBand(row.ConfidenceScore) !== "high") return false;
        if (!debouncedQuery) return true;
        const text = `${row.PatientName || ""} ${row.PatientID || ""} ${row.PatientEmail || ""} ${
          row.PatientPhone || ""
        }`.toLowerCase();
        return text.includes(debouncedQuery);
      }),
    [debouncedQuery, filters.highConfidenceOnly, filters.risk, filters.time, predictions]
  );

  const totalPredictions = filteredPredictions.length;
  const normalCount = filteredPredictions.filter((row) => row.PredictionResult === "Normal").length;
  const diseaseCount = filteredPredictions.filter((row) => row.PredictionResult === "Disease").length;
  const trend = useMemo(() => riskTrend(filteredPredictions), [filteredPredictions]);

  const highRiskQueue = useMemo(
    () =>
      [...filteredPredictions]
        .filter((row) => row.PredictionResult !== "Normal")
        .sort((a, b) => {
          const diff =
            (RISK_META[b.PredictionResult]?.priority || 0) -
            (RISK_META[a.PredictionResult]?.priority || 0);
          if (diff !== 0) return diff;
          return (
            (parseApiDate(b.Timestamp)?.getTime() || 0) -
            (parseApiDate(a.Timestamp)?.getTime() || 0)
          );
        })
        .slice(0, 6),
    [filteredPredictions]
  );

  const patientCards = useMemo(() => {
    const latestByPatient = new Map();
    filteredPredictions.forEach((row) => {
      const key = String(row.PatientID || "");
      if (!key) return;
      const current = latestByPatient.get(key);
      const nextTime = parseApiDate(row.Timestamp)?.getTime() || 0;
      const currentTime = parseApiDate(current?.Timestamp)?.getTime() || 0;
      if (!current || nextTime > currentTime) latestByPatient.set(key, row);
    });

    const sourcePatients =
      patients.length > 0
        ? patients
        : filteredPredictions.slice(0, 6).map((row) => ({
            PatientID: row.PatientID,
            Name: row.PatientName || `Patient ${row.PatientID}`,
          }));

    return sourcePatients
      .filter((patient) => {
        if (!debouncedQuery) return true;
        const text = `${patient.Name || ""} ${patient.PatientID || ""} ${patient.Email || ""} ${
          patient.Phone || ""
        }`.toLowerCase();
        return text.includes(debouncedQuery);
      })
      .filter((patient) => {
        if (filters.risk === "all") return true;
        const latest = latestByPatient.get(String(patient.PatientID));
        return latest?.PredictionResult === filters.risk;
      })
      .map((patient, index) => {
      const latest = latestByPatient.get(String(patient.PatientID));
      const name = patient.Name || latest?.PatientName || `Patient ${patient.PatientID}`;
      const initials = name
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0]?.toUpperCase())
        .join("");
      const risk = latest?.PredictionResult || "Normal";
      return {
        id: patient.PatientID || index,
        name,
        initials: initials || "P",
        age: patient.Age || 72 + index,
        email: patient.Email || "",
        phone: patient.Phone || patient.PhoneNumber || "",
        risk,
        prediction: latest || null,
      };
    });
  }, [debouncedQuery, filteredPredictions, filters.risk, patients]);

  const selectedPatient =
    patientCards.find((patient) => String(patient.id) === String(selectedPatientId)) ||
    patientCards[0] ||
    null;
  const selectedPrediction = selectedPatient?.prediction || null;

  const filteredHighRiskQueue = useMemo(() => {
    if (alertFilter === "critical") {
      return highRiskQueue.filter((row) => row.PredictionResult === "Disease");
    }
    return highRiskQueue;
  }, [alertFilter, highRiskQueue]);

  const thresholdRows = [
    {
      label: "Normal Probability",
      value: "P(Normal)",
      detail: "Routine screening target",
      tone: "normal",
    },
    {
      label: "Disease Probability",
      value: "P(Disease)",
      detail: "Review required target",
      tone: "critical",
    },
    {
      label: "Decision Threshold",
      value: "0.50",
      detail: "Disease when P(Disease) >= 50%",
      tone: "critical",
    },
  ];
  const riskPercentRows = [
    { label: "Normal", count: normalCount, tone: "normal", range: "P(Normal)" },
    { label: "Disease", count: diseaseCount, tone: "critical", range: "P(Disease)" },
  ];
  const latestActivities = filteredPredictions.slice(0, 3).map((row) => ({
    title: row.PredictionResult,
    body: `${row.PatientName || `Patient ${row.PatientID}`} - ${formatPercent(Number(row.ConfidenceScore || 0))}`,
    time: parseApiDate(row.Timestamp)?.toLocaleDateString() || "Recent",
    row,
  }));

  const openPatientDrawer = useCallback(
    async (patientLike, prediction = null) => {
      const patientId = patientLike?.PatientID || patientLike?.id || prediction?.PatientID;
      if (!patientId) return;
      const requestId = drawerRequestRef.current + 1;
      drawerRequestRef.current = requestId;
      setDrawerLoading(true);
      setDrawerPatient(patientLike);
      setDrawerPrediction(prediction);
      try {
        const payload = await apiRequest(`/predictions/history/${patientId}`, {}, token);
        if (drawerRequestRef.current !== requestId) return;
        setDrawerPatient(payload.patient);
        setDrawerHistory(payload.predictions || []);
        setDrawerChecklist(payload.checklist || {});
        if (prediction) {
          const fullPrediction = (payload.predictions || []).find(
            (row) => row.PredictionID === prediction.PredictionID
          );
          setDrawerPrediction(fullPrediction || prediction);
        } else {
          setDrawerPrediction((payload.predictions || [])[0] || null);
        }
      } catch (error) {
        if (drawerRequestRef.current !== requestId) return;
        notify("Could not load patient details", error.message, "critical");
      } finally {
        if (drawerRequestRef.current === requestId) {
          setDrawerLoading(false);
        }
      }
    },
    [notify, token]
  );

  const closePatientDrawer = useCallback(() => {
    drawerRequestRef.current += 1;
    setDrawerPatient(null);
    setDrawerHistory([]);
    setDrawerChecklist({});
    setDrawerPrediction(null);
    setDrawerLoading(false);
  }, []);

  const drawerPatientId = drawerPatient?.PatientID || drawerPatient?.id || null;

  const saveChecklist = async (key, value) => {
    const patientId = drawerPatient?.PatientID || drawerPatient?.id;
    if (!patientId) return;
    const next = { ...drawerChecklist, [key]: value };
    setDrawerChecklist(next);
    try {
      await apiRequest(`/patients/${patientId}/checklist`, { method: "PUT", body: next }, token);
    } catch (error) {
      notify("Checklist was not saved", error.message, "critical");
    }
  };

  const downloadReport = async () => {
    const patientId = drawerPatient?.PatientID || drawerPatient?.id;
    if (!patientId) return;
    try {
      const response = await fetch(`${API_BASE}/report/generate/${patientId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error(`Report failed (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `retinarisk_patient_${patientId}_report.pdf`;
      link.click();
      URL.revokeObjectURL(url);
      notify("Report downloaded", "Patient PDF report generated.", "normal");
    } catch (error) {
      notify("Report download failed", error.message, "critical");
    }
  };

  const deleteDrawerPatient = async () => {
    const patientId = drawerPatient?.PatientID || drawerPatient?.id;
    const patientName = drawerPatient?.Name || drawerPatient?.name || `Patient ${patientId}`;
    if (!patientId) return;
    const confirmed = window.confirm(`Delete ${patientName} and all related predictions?`);
    if (!confirmed) return;
    try {
      await onDeletePatient(patientId);
      setDrawerPatient(null);
      setDrawerHistory([]);
      setDrawerPrediction(null);
    } catch (error) {
      notify("Patient delete failed", error.message, "critical");
    }
  };

  return (
    <section className="care-dashboard cvd-dashboard">
      <aside className="patient-rail">
        <div className="rail-heading">
          <h3>Patients</h3>
          <span>{patients.length || patientCards.length} total</span>
        </div>
        <button className="care-add-btn" type="button" onClick={() => onNavigate("patients")}>
          <Plus size={16} />
          Add Patient
        </button>
        <FilterPanel filters={filters} onChange={setFilters} />
        <div className="patient-list">
          {patientCards.length === 0 ? (
            <p className="empty">No patients yet.</p>
          ) : (
            patientCards.map((patient) => (
              <button
                className={`patient-card ${
                  String(selectedPatient?.id) === String(patient.id) ? "selected" : ""
                }`}
                key={patient.id}
                onClick={() => {
                  setSelectedPatientId(String(patient.id));
                  void openPatientDrawer(patient, patient.prediction);
                }}
                type="button"
              >
                <span className={`avatar ${riskTone(patient.risk)}`}>{patient.initials}</span>
                <span>
                  <strong>{patient.name}</strong>
                  <small>{patient.age}y - {patient.risk}</small>
                </span>
                <i className={`patient-status ${riskTone(patient.risk)}`} />
              </button>
            ))
          )}
        </div>
        <article className="contact-card">
          <div>
            <AlertTriangle size={18} />
            <h3>Patient Contact</h3>
          </div>
          {selectedPatient ? (
            <>
              <strong>{selectedPatient.name}</strong>
              <span>
                <Mail size={15} />
                {selectedPatient.email || "Email not recorded"}
              </span>
              <span>
                <Phone size={15} />
                {selectedPatient.phone || "Phone not recorded"}
              </span>
            </>
          ) : (
            <p>No patient selected.</p>
          )}
        </article>
      </aside>

      <div className="care-main">
        <article className="care-panel selected-patient-panel">
          <div className="selected-patient-head">
            <div>
              <span>Selected Patient</span>
              <h3>{selectedPatient ? selectedPatient.name : "No patient selected"}</h3>
              <p>
                {selectedPatient
                  ? `ID ${selectedPatient.id} - Age ${selectedPatient.age} - ${selectedPatient.email}`
                  : "Add a patient to begin screening."}
              </p>
            </div>
            {selectedPatient ? <RiskPill value={selectedPatient.risk} /> : null}
          </div>
          {selectedPatient ? (
            <div className="selected-patient-grid">
              <div>
                <span>Phone</span>
                <strong>{selectedPatient.phone || "Not recorded"}</strong>
              </div>
              <div>
                <span>Latest Prediction</span>
                <strong>{selectedPrediction?.PredictionResult || "Not screened"}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>
                  {selectedPrediction
                    ? formatPercent(Number(selectedPrediction.ConfidenceScore || 0), 2)
                    : "--"}
                </strong>
              </div>
              <div>
                <span>Screened On</span>
                <strong>{parseApiDate(selectedPrediction?.Timestamp)?.toLocaleDateString() || "--"}</strong>
              </div>
            </div>
          ) : null}
        </article>

        <article className={`care-panel trend-summary ${trend.direction}`}>
          <div>
            <span>Risk Trend</span>
            <h3>
              {trend.direction === "up" ? "Increasing" : trend.direction === "down" ? "Decreasing" : "Stable"}
            </h3>
            <p>This week vs last week average risk score</p>
          </div>
          <strong>{trend.direction === "up" ? "↑" : trend.direction === "down" ? "↓" : "→"} {trend.label}</strong>
        </article>

        <article className="care-panel threshold-panel">
          <div className="care-panel-head">
            <div>
              <h3>Retinal Severity Target</h3>
              <p>Professional target mapping for the binary retinal screening model.</p>
            </div>
            <span className="model-badge">Best Performing Production Model: EfficientNet-B3</span>
          </div>
          <div className="threshold-grid">
            {thresholdRows.map((item) => (
              <div className={`threshold-card ${item.tone}`} key={item.label}>
                <div className="threshold-card-head">
                  <span>{item.label}</span>
                  <small className={item.tone}>{item.detail}</small>
                </div>
                <strong>{item.value}</strong>
                <div className="threshold-confidence">
                  <span>Confidence indicator</span>
                  <b>{item.tone === "normal" ? "Low risk" : "Clinical review"}</b>
                </div>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${item.tone}`}
                    style={{ width: item.tone === "normal" ? "62%" : "78%" }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="risk-percent-grid">
            {riskPercentRows.map((row) => {
              const percent = totalPredictions ? Math.round((row.count / totalPredictions) * 100) : 0;
              return (
                <div className={`risk-percent-card ${row.tone}`} key={row.label}>
                  <div>
                    <span>{row.label}</span>
                    <strong>{row.range}</strong>
                  </div>
                  <b>{percent}%</b>
                  <div className="bar-track">
                    <div className={`bar-fill ${row.tone}`} style={{ width: `${percent}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </article>

        <div className="care-work-grid">
          <Suspense fallback={<article className="care-panel chart-loading">Loading chart...</article>}>
            <RiskTimelineChart
              predictions={
                selectedPatient
                  ? filteredPredictions.filter(
                      (row) => String(row.PatientID) === String(selectedPatient.id)
                    )
                  : []
              }
              onSelect={(prediction) => openPatientDrawer(selectedPatient, prediction)}
            />
          </Suspense>
          <Suspense fallback={<article className="care-panel chart-loading">Loading chart...</article>}>
            <ConfidenceChart predictions={filteredPredictions} />
          </Suspense>
          <article className="care-panel cvd-panel">
            <div className="care-panel-title">
              <ClipboardList size={18} />
              <h3>About Retinal Screening</h3>
            </div>
            <p>
              This project predicts retinal image severity from the available training labels.
              It can support follow-up decisions, but it does not diagnose CVD, BP, or cholesterol.
            </p>
            <div className="cvd-points">
              <span>The output is a learned image-severity class.</span>
              <span>Disease results should trigger retinal and clinical review.</span>
              <span>Disease results need prompt clinician confirmation.</span>
            </div>
            <button className="submit-order-btn" type="button" onClick={() => onNavigate("screening")}>
              Start Screening
            </button>
          </article>

          <article className="care-panel heart-risk-panel">
            <div className="care-panel-title">
              <CheckCircle2 size={18} />
              <h3>Clinical Context</h3>
              <span>{highRiskQueue.length}</span>
            </div>
            <p>
              Retinal findings can be clinically meaningful, but this model was trained on
              retinal severity labels. Use the result as a screening signal and confirm with
              appropriate clinical tests.
            </p>
            <div className="risk-explain-grid">
              <div>
                <strong>Normal</strong>
                <span>Maintain routine monitoring and healthy habits.</span>
              </div>
              <div>
                <strong>Disease</strong>
                <span>Arrange prompt doctor or ophthalmology review.</span>
              </div>
            </div>
          </article>
        </div>
      </div>

      <aside className="alert-rail">
        <div className="rail-heading">
          <h3>Active Screening Alerts</h3>
          <span className="alert-count">{highRiskQueue.length}</span>
        </div>
        <select
          className="alert-filter"
          value={alertFilter}
          onChange={(event) => setAlertFilter(event.target.value)}
        >
          <option value="all">All risk results</option>
          <option value="critical">Disease</option>
        </select>
        <div className="alert-list">
          {filteredHighRiskQueue.length === 0 ? (
            <p className="empty">No active screening alerts.</p>
          ) : (
            filteredHighRiskQueue.slice(0, 4).map((row) => (
              <button
                className={`alert-card ${riskTone(row.PredictionResult)}`}
                key={row.PredictionID}
                type="button"
                onClick={() => openPatientDrawer({ PatientID: row.PatientID, Name: row.PatientName }, row)}
              >
                <strong>
                  {row.PatientName || `Patient ${row.PatientID}`} -{" "}
                  Critical
                </strong>
                <span>{row.PredictionResult} - {formatPercent(Number(row.ConfidenceScore || 0))} confidence</span>
              </button>
            ))
          )}
        </div>

        <div className="activity-section">
          <div className="rail-heading">
            <h3>Recent Screening Signals</h3>
          </div>
          <div className="activity-list">
            {(latestActivities.length
              ? latestActivities
              : [{ title: "No screenings yet", body: "Start a retinal screening to populate this panel.", time: "Now" }]
            ).map((item) => (
              <button
                className="activity-item clickable"
                key={`${item.title}-${item.body}`}
                onClick={() => item.row && openPatientDrawer({ PatientID: item.row.PatientID, Name: item.row.PatientName }, item.row)}
                type="button"
              >
                <span />
                <div>
                  <strong>{item.title}</strong>
                  <small>{item.body}</small>
                </div>
                <time>{item.time}</time>
              </button>
            ))}
          </div>
        </div>
      </aside>
      <PatientDrawer
        apiBase={API_BASE}
        token={token}
        open={Boolean(drawerPatientId)}
        patient={drawerPatient}
        history={drawerLoading ? [] : drawerHistory}
        checklist={drawerChecklist}
        selectedPrediction={drawerPrediction}
        onClose={closePatientDrawer}
        onPredictionSelect={setDrawerPrediction}
        onChecklistChange={saveChecklist}
        onDelete={deleteDrawerPatient}
        onEdit={() => onNavigate("patients")}
        onDownloadReport={downloadReport}
      />
    </section>
  );
}

function MedicationsView({ patients, predictions }) {
  const patientRows = useMemo(() => {
    const patientMap = new Map(patients.map((patient) => [String(patient.PatientID), patient]));
    const latestByPatient = new Map();
    predictions.forEach((row) => {
      const key = String(row.PatientID || "");
      if (!key) return;
      const current = latestByPatient.get(key);
      const nextTime = parseApiDate(row.Timestamp)?.getTime() || 0;
      const currentTime = parseApiDate(current?.Timestamp)?.getTime() || 0;
      if (!current || nextTime > currentTime) latestByPatient.set(key, row);
    });

    const fromPredictions = [...latestByPatient.values()].map((prediction) => {
      const patient = patientMap.get(String(prediction.PatientID));
      return {
        id: prediction.PatientID,
        name: patient?.Name || prediction.PatientName || `Patient ${prediction.PatientID}`,
        age: patient?.Age || "--",
        email: patient?.Email || "",
        risk: prediction.PredictionResult,
        confidence: Number(prediction.ConfidenceScore || 0),
      };
    });

    if (fromPredictions.length) return fromPredictions;
    return patients.map((patient) => ({
      id: patient.PatientID,
      name: patient.Name,
      age: patient.Age || "--",
      email: patient.Email || "",
      risk: "Not screened",
      confidence: 0,
    }));
  }, [patients, predictions]);

  const guidanceByRisk = {
    Normal: {
      title: "Maintenance Precautions",
      medicines: "No risk-specific medicine suggestion. Continue prescribed medicines only.",
      precautions: [
        "Maintain routine eye care and clinician-advised health checks.",
        "Repeat screening on schedule or earlier if symptoms appear.",
      ],
    },
    Disease: {
      title: "Disease Precautions",
      medicines:
        "Urgent clinician review is needed. Statins, BP medicines, antiplatelets, or diabetes medicines must only be started or changed by a doctor.",
      precautions: [
        "Check chest pain, breathlessness, weakness, speech trouble, fainting, or sudden vision changes immediately.",
        "Avoid heavy exertion until reviewed and keep current prescriptions available for the doctor.",
      ],
    },
  };

  return (
    <section className="view medication-view">
      <PanelHeader
        title="Clinical Follow-Up Guidance"
        subtitle="Patient-specific guidance from the latest retinal severity result"
      />
      <div className="medication-grid">
        {patientRows.length === 0 ? (
          <article className="care-panel medication-empty">
            <p>No patients available. Add a patient and run a screening to generate guidance.</p>
          </article>
        ) : (
          patientRows.map((patient) => {
            const guidance = guidanceByRisk[patient.risk] || guidanceByRisk.Normal;
            return (
              <article className={`medication-card ${riskTone(patient.risk)}`} key={patient.id}>
                <div className="medication-head">
                  <div>
                    <span>Patient</span>
                    <h3>{patient.name}</h3>
                    <p>Age {patient.age} - {patient.email || "No email recorded"}</p>
                  </div>
                  <RiskPill value={patient.risk} />
                </div>
                <div className="medication-confidence">
                  <span>Latest confidence</span>
                  <strong>{patient.confidence ? formatPercent(patient.confidence) : "Not screened"}</strong>
                </div>
                <div className="medication-section">
                  <h4>{guidance.title}</h4>
                  <ul>
                    {guidance.precautions.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div className="medication-note">
                  <Pill size={17} />
                  <p>{guidance.medicines}</p>
                </div>
              </article>
            );
          })
        )}
      </div>
    </section>
  );
}

function ScreeningView({ token, patients, onPredictionSaved, notify }) {
  const fileInputRef = useRef(null);
  const [patientId, setPatientId] = useState("");
  const [sourceMode, setSourceMode] = useState("upload");
  const [dragActive, setDragActive] = useState(false);
  const [cameraAvailable, setCameraAvailable] = useState(null);
  const [streamRefresh, setStreamRefresh] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  const [imageSrc, setImageSrc] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [liveCapture, setLiveCapture] = useState(null);

  const [result, setResult] = useState(null);
  const selectedPatient = patients.find(
    (patient) => String(patient.PatientID) === String(patientId)
  );

  useEffect(() => {
    let active = true;
    apiRequest("/capture/check", {}, token)
      .then((response) => {
        if (active) setCameraAvailable(Boolean(response.camera_available));
      })
      .catch(() => {
        if (active) setCameraAvailable(false);
      });
    return () => {
      active = false;
    };
  }, [token]);

  useEffect(
    () => () => {
      if (imageSrc.startsWith("blob:")) URL.revokeObjectURL(imageSrc);
    },
    [imageSrc]
  );

  const setPreviewFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const nextSrc = URL.createObjectURL(file);
    setImageSrc((prev) => {
      if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
      return nextSrc;
    });
    setSelectedFile(file);
    setLiveCapture(null);
    setResult(null);
    setError("");
  }, []);

  const triggerFilePicker = () => fileInputRef.current?.click();

  const startCaptureMode = () => {
    if (!patientId) {
      setError("Select a patient before opening live capture.");
      return;
    }
    setSourceMode("camera");
    setLiveCapture(null);
    setResult(null);
    setStreamRefresh((prev) => prev + 1);
    setError("");
  };

  const snapCapture = async () => {
    setCapturing(true);
    setError("");
    try {
      const response = await apiRequest("/capture/snap", { method: "POST" }, token);
      if (!response?.preview) throw new Error("Camera preview not returned");

      const blobResponse = await fetch(response.preview);
      const blob = await blobResponse.blob();
      const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
      setLiveCapture({ src: response.preview, file });
    } catch (captureError) {
      setError(captureError.message);
    } finally {
      setCapturing(false);
    }
  };

  const acceptCapture = () => {
    if (!liveCapture) return;
    setImageSrc((prev) => {
      if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
      return liveCapture.src;
    });
    setSelectedFile(liveCapture.file);
    setSourceMode("upload");
    setLiveCapture(null);
  };

  const predict = async () => {
    if (!patientId) {
      setError("Select a patient to continue.");
      return;
    }
    if (!selectedFile) {
      setError("Attach an image before prediction.");
      return;
    }

    setPredicting(true);
    setResult(null);
    setError("");
    try {
      const formData = new FormData();
      formData.append("patient_id", patientId);
      formData.append("image", selectedFile);

      const response = await apiRequest("/predict", { method: "POST", body: formData }, token);
      setResult(response);
      notify(
        "Screening completed",
        `Prediction result: ${response.retinal_severity || response.result || response.cvd_risk}.`,
        "normal"
      );
      if ((response.retinal_severity || response.result || response.cvd_risk) !== "Normal") {
        notify("High-risk detected", "Review the patient alert and care guidance.", "critical");
      }
      onPredictionSaved();
    } catch (predictError) {
      setError(predictError.message);
    } finally {
      setPredicting(false);
    }
  };

  const clearSelectedImage = () => {
    setImageSrc((prev) => {
      if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
      return "";
    });
    setSelectedFile(null);
    setResult(null);
    setLiveCapture(null);
  };

  const finalRisk = result?.retinal_severity || result?.result || result?.cvd_risk || "";
  const productionConfidence = Number(
    result?.production_confidence ?? result?.confidence ?? result?.ensemble_confidence ?? 0
  );
  const productionProbability = Number(
    result?.risk_score ?? result?.production_probabilities?.Disease ?? productionConfidence
  );
  const productionProbabilities =
    result?.production_probabilities || result?.class_probabilities || {};
  const individualPredictions = Object.values(
    result?.individual_model_predictions || result?.models || {}
  );
  const predictionSource = result?.prediction_source
    ? formatKeyLabel(result.prediction_source)
    : "Efficientnet";
  const productionModel = result?.production_model || "EfficientNet-B3";
  const explainabilityMaps = Object.entries(result?.explainability || {}).filter(
    ([, value]) => Boolean(value)
  );
  const vesselFeatures = Object.entries(result?.vessel_features || {});

  return (
    <section className="view screening-view">
      <PanelHeader
        title="Screening Workbench"
        subtitle="Upload or capture a retinal image and predict retinal severity"
      />

      <div className="panel-grid screening-layout">
        <article className="panel">
          <div className="form-grid two-col">
            <label>
              <span>Patient</span>
              <select value={patientId} onChange={(event) => setPatientId(event.target.value)}>
                <option value="">Select patient</option>
                {patients.map((patient) => (
                  <option key={patient.PatientID} value={patient.PatientID}>
                    {patient.Name} (ID {patient.PatientID})
                  </option>
                ))}
              </select>
            </label>

            <div className="screening-patient-card">
              <span>Patient Details</span>
              <strong>{selectedPatient?.Name || "Select a patient"}</strong>
              <small>
                {selectedPatient
                  ? `ID ${selectedPatient.PatientID} - ${selectedPatient.Age ?? "--"}y - ${
                      selectedPatient.Phone || "No phone"
                    }`
                  : "Patient profile will appear here"}
              </small>
            </div>
          </div>

          <div className="segmented">
            <button
              type="button"
              className={sourceMode === "upload" ? "active" : ""}
              onClick={() => setSourceMode("upload")}
            >
              Upload
            </button>
            <button
              type="button"
              className={sourceMode === "camera" ? "active" : ""}
              onClick={startCaptureMode}
            >
              Camera
            </button>
          </div>

          {sourceMode === "upload" ? (
            <div
              className={`upload-zone ${dragActive ? "active" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                event.preventDefault();
                setDragActive(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                setPreviewFile(event.dataTransfer.files?.[0]);
              }}
            >
              <p>Drop retinal image here</p>
              <button className="btn btn-outline" type="button" onClick={triggerFilePicker}>
                Choose File
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(event) => setPreviewFile(event.target.files?.[0])}
                hidden
              />
            </div>
          ) : (
            <div className="camera-zone">
              <div className="camera-status">
                Camera status:{" "}
                {cameraAvailable === null
                  ? "Checking..."
                  : cameraAvailable
                    ? "Available"
                    : "Not detected"}
              </div>
              {!liveCapture ? (
                <img
                  alt="Camera stream"
                  className="camera-stream"
                  src={`${API_BASE}/capture/stream?token=${encodeURIComponent(token)}&camera_index=0&t=${streamRefresh}`}
                />
              ) : (
                <img alt="Captured retinal preview" className="camera-stream" src={liveCapture.src} />
              )}
              <div className="button-row">
                {!liveCapture ? (
                  <button
                    className="btn btn-outline"
                    disabled={capturing || cameraAvailable === false}
                    onClick={snapCapture}
                    type="button"
                  >
                    {capturing ? "Capturing..." : "Capture Frame"}
                  </button>
                ) : (
                  <>
                    <button className="btn btn-primary" type="button" onClick={acceptCapture}>
                      Use Capture
                    </button>
                    <button
                      className="btn btn-outline"
                      type="button"
                      onClick={() => {
                        setLiveCapture(null);
                        setStreamRefresh((prev) => prev + 1);
                      }}
                    >
                      Retake
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {error ? <div className="form-error">{error}</div> : null}

          <div className="button-row">
            <button
              className="btn btn-primary"
              type="button"
              disabled={predicting || !selectedFile}
              onClick={predict}
            >
              {predicting ? "Running Prediction..." : "Run Prediction"}
            </button>
            <button className="btn btn-ghost" type="button" onClick={clearSelectedImage}>
              Clear Image
            </button>
          </div>
        </article>

        <article className="panel">
          <PanelHeader
            title="Image Preview"
            subtitle={selectedFile ? selectedFile.name : "No image selected"}
          />

          {imageSrc ? (
            <img src={imageSrc} alt="Retinal preview" className="retinal-preview" />
          ) : (
            <p className="empty">Select an image from upload or camera to continue.</p>
          )}

          {result ? (
            <section className="result-block">
              <div className="result-head">
                <h4>Prediction Result</h4>
                <RiskPill value={finalRisk} />
              </div>

              <div className="result-kpi-grid">
                <div className="result-kpi">
                  <span>Confidence</span>
                  <strong>{formatPercent(productionConfidence, 2)}</strong>
                </div>
                <div className="result-kpi">
                  <span>Predicted Class</span>
                  <RiskPill value={finalRisk} />
                </div>
                <div className="result-kpi">
                  <span>Ensemble Probability</span>
                  <strong>{formatPercent(productionProbability, 2)}</strong>
                </div>
                <div className="result-kpi">
                  <span>Severity Score</span>
                  <strong>{formatPercent(Number(result.risk_score || 0), 2)}</strong>
                </div>
              </div>

              <div className="result-summary-row">
                <span>Severity Band: {result.probability_risk_band || "--"}</span>
                <span>Model Agreement: {result.model_agreement || "--"}</span>
                <span>Prediction Source: {predictionSource}</span>
                <span>Production Model: {productionModel}</span>
              </div>

              {result.target_description ? (
                <p className="result-confidence">{result.target_description}</p>
              ) : null}

              <CareGuidancePanel risk={finalRisk} />

              <div className="risk-list">
                {Object.entries(productionProbabilities).map(([label, score]) => (
                  <div key={label} className="risk-row">
                    <div className="risk-row-head">
                      <span>{label}</span>
                      <span>{formatPercent(Number(score || 0), 2)}</span>
                    </div>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${riskTone(label)}`}
                        style={{ width: `${Number(score || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {individualPredictions.length ? (
                <div className="table-wrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Weight</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {individualPredictions.map((model) => (
                        <tr key={model.model_name}>
                          <td>{model.model_name}</td>
                          <td>{formatPercent(Number(model.weight || 0), 0)}</td>
                          <td>
                            <RiskPill value={model.prediction} />
                          </td>
                          <td>
                            {model.confidence_percent !== undefined
                              ? `${Number(model.confidence_percent || 0).toFixed(2)}%`
                              : formatPercent(Number(model.confidence || 0), 2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}

              {explainabilityMaps.length ? (
                <div className="explain-grid">
                  {explainabilityMaps.map(([key, value]) => (
                    <figure className="explain-item" key={key}>
                      <img alt={`${formatKeyLabel(key)} heatmap`} src={value} />
                      <figcaption>{formatKeyLabel(key)}</figcaption>
                    </figure>
                  ))}
                </div>
              ) : null}

              {vesselFeatures.length ? (
                <div className="vessel-grid">
                  {vesselFeatures.map(([key, value]) => (
                    <div className="vessel-item" key={key}>
                      <span>{formatKeyLabel(key)}</span>
                      <strong>{formatPercent(Number(value || 0), 1)}</strong>
                    </div>
                  ))}
                </div>
              ) : null}

              {result.note ? <p className="result-confidence">{result.note}</p> : null}

              <div className="result-meta">
                <span>Prediction ID #{result.prediction_id}</span>
                <span>{parseApiDate(result.timestamp)?.toLocaleString() || "Unknown time"}</span>
              </div>
            </section>
          ) : null}
        </article>
      </div>
    </section>
  );
}

function PatientsView({ token, patients, onSaved, onAuthExpired, notify }) {
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [form, setForm] = useState({
    name: "",
    age: "",
    gender: "Male",
    email: "",
    phone: "",
  });

  const filteredPatients = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return patients;
    return patients.filter((patient) => {
      return (
        patient.Name?.toLowerCase().includes(q) ||
        String(patient.PatientID).includes(q) ||
        (patient.Email || "").toLowerCase().includes(q) ||
        (patient.Phone || "").toLowerCase().includes(q)
      );
    });
  }, [patients, query]);

  const savePatient = async (event) => {
    event.preventDefault();
    setSaving(true);
    setError("");

    try {
      await apiRequest(
        "/patients",
        {
          method: "POST",
          body: {
            name: form.name.trim(),
            age: Number(form.age),
            gender: form.gender,
            email: form.email.trim(),
            phone: form.phone.trim(),
          },
        },
        token
      );
      setForm({ name: "", age: "", gender: "Male", email: "", phone: "" });
      setShowForm(false);
      notify("Patient added", `${form.name.trim()} was registered successfully.`, "normal");
      onSaved();
    } catch (saveError) {
      if (saveError.status === 401) {
        onAuthExpired(saveError.message);
        return;
      }
      setError(saveError.message);
    } finally {
      setSaving(false);
    }
  };

  const deletePatient = async (patient) => {
    const confirmed = window.confirm(
      `Delete patient ID ${patient.PatientID} (${patient.Name}) and all related predictions?`
    );
    if (!confirmed) return;

    setDeletingId(patient.PatientID);
    setError("");
    try {
      await apiRequest(`/patients/${patient.PatientID}`, { method: "DELETE" }, token);
      notify("Patient deleted", `Patient ID ${patient.PatientID} was removed.`, "critical");
      await onSaved();
    } catch (deleteError) {
      if (deleteError.status === 401) {
        onAuthExpired(deleteError.message);
        return;
      }
      setError(deleteError.message);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <section className="view">
      <PanelHeader
        title="Patient Registry"
        subtitle={`${patients.length} registered patients`}
        actions={
          <button
            className="btn btn-primary"
            type="button"
            onClick={() => setShowForm((prev) => !prev)}
          >
            {showForm ? "Close Form" : "Add Patient"}
          </button>
        }
      />

      {showForm ? (
        <article className="panel">
          <form onSubmit={savePatient} className="form-grid two-col">
            <label>
              <span>Full Name</span>
              <input
                value={form.name}
                onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
                required
              />
            </label>
            <label>
              <span>Age</span>
              <input
                type="number"
                min="1"
                value={form.age}
                onChange={(event) => setForm((prev) => ({ ...prev, age: event.target.value }))}
                required
              />
            </label>
            <label>
              <span>Gender</span>
              <select
                value={form.gender}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, gender: event.target.value }))
                }
              >
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </label>
            <label>
              <span>Email</span>
              <input
                type="email"
                value={form.email}
                onChange={(event) => setForm((prev) => ({ ...prev, email: event.target.value }))}
                required
              />
            </label>
            <label>
              <span>Phone Number</span>
              <input
                type="tel"
                value={form.phone}
                onChange={(event) => setForm((prev) => ({ ...prev, phone: event.target.value }))}
                required
              />
            </label>
            {error ? <div className="form-error full">{error}</div> : null}
            <div className="button-row full">
              <button className="btn btn-primary" disabled={saving} type="submit">
                {saving ? "Saving..." : "Save Patient"}
              </button>
              <button className="btn btn-ghost" type="button" onClick={() => setShowForm(false)}>
                Cancel
              </button>
            </div>
          </form>
        </article>
      ) : null}

      <article className="panel">
        <div className="table-toolbar">
          <input
            placeholder="Search by patient, id, email, or phone"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Gender</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Predictions</th>
                <th>Registered</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredPatients.length === 0 ? (
                <tr>
                  <td className="empty" colSpan={9}>
                    No matching patients
                  </td>
                </tr>
              ) : (
                filteredPatients.map((patient) => (
                  <tr key={patient.PatientID}>
                    <td>{patient.PatientID}</td>
                    <td>{patient.Name}</td>
                    <td>{patient.Age ?? "--"}</td>
                    <td>{patient.Gender ?? "--"}</td>
                    <td>{patient.Email || "--"}</td>
                    <td>{patient.Phone || "--"}</td>
                    <td>{patient.PredictionCount || 0}</td>
                    <td>{parseApiDate(patient.CreatedAt)?.toLocaleDateString() || "--"}</td>
                    <td>
                      <button
                        className="table-action danger"
                        disabled={deletingId === patient.PatientID}
                        onClick={() => deletePatient(patient)}
                        title="Delete patient ID"
                        type="button"
                      >
                        <Trash2 size={15} />
                        <span>{deletingId === patient.PatientID ? "Deleting" : "Delete"}</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}

function HistoryView({
  predictions,
  onRefresh,
  refreshing,
  onDeletePrediction,
  onClearHistory,
}) {
  const [riskFilter, setRiskFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [deletingId, setDeletingId] = useState(null);
  const [clearing, setClearing] = useState(false);
  const [actionError, setActionError] = useState("");

  const filtered = useMemo(() => {
    return predictions.filter((row) => {
      const byRisk = riskFilter === "all" ? true : row.PredictionResult === riskFilter;
      if (!byRisk) return false;
      const text = `${row.PatientName || ""} ${row.PatientID || ""} ${row.PredictionID || ""}`.toLowerCase();
      return text.includes(query.trim().toLowerCase());
    });
  }, [predictions, query, riskFilter]);

  const deletePrediction = async (predictionId) => {
    const confirmed = window.confirm(`Delete prediction #${predictionId} from history?`);
    if (!confirmed) return;
    setDeletingId(predictionId);
    setActionError("");
    try {
      await onDeletePrediction(predictionId);
    } catch (error) {
      setActionError(error.message);
    } finally {
      setDeletingId(null);
    }
  };

  const clearHistory = async () => {
    const confirmed = window.confirm("Delete all prediction history records?");
    if (!confirmed) return;
    setClearing(true);
    setActionError("");
    try {
      await onClearHistory();
    } catch (error) {
      setActionError(error.message);
    } finally {
      setClearing(false);
    }
  };

  return (
    <section className="view">
      <PanelHeader
        title="Prediction History"
        subtitle={`${predictions.length} records loaded`}
        actions={
          <div className="button-row compact">
            <button
              className="btn btn-outline icon-btn"
              onClick={onRefresh}
              disabled={refreshing}
              type="button"
            >
              <RefreshCw size={16} />
              {refreshing ? "Refreshing..." : "Refresh"}
            </button>
            <button
              className="btn btn-danger icon-btn"
              onClick={clearHistory}
              disabled={clearing || predictions.length === 0}
              type="button"
            >
              <Trash2 size={16} />
              {clearing ? "Clearing..." : "Clear History"}
            </button>
          </div>
        }
      />

      <article className="panel">
        <div className="table-toolbar split history-tools">
          <label className="search-field">
            <Search size={16} />
            <input
              placeholder="Search by patient or id"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
            <option value="all">All outcomes</option>
            <option value="Normal">Normal</option>
            <option value="Disease">Disease</option>
          </select>
        </div>
        {actionError ? <div className="form-error history-error">{actionError}</div> : null}

        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Prediction</th>
                <th>Patient</th>
                <th>Outcome</th>
                <th>Confidence</th>
                <th>Model</th>
                <th>Timestamp</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td className="empty" colSpan={7}>
                    No predictions match your filter
                  </td>
                </tr>
              ) : (
                filtered.map((row) => (
                  <tr key={row.PredictionID}>
                    <td>#{row.PredictionID}</td>
                    <td>{row.PatientName || `Patient ${row.PatientID}`}</td>
                    <td>
                      <RiskPill value={row.PredictionResult} />
                    </td>
                    <td>{formatPercent(Number(row.ConfidenceScore || 0), 2)}</td>
                    <td>{row.AlgorithmName || "--"}</td>
                    <td>{parseApiDate(row.Timestamp)?.toLocaleString() || "--"}</td>
                    <td>
                      <button
                        className="table-action danger"
                        onClick={() => deletePrediction(row.PredictionID)}
                        disabled={deletingId === row.PredictionID}
                        title="Delete prediction"
                        type="button"
                      >
                        <Trash2 size={15} />
                        <span>{deletingId === row.PredictionID ? "Deleting" : "Delete"}</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}

function AnalyticsView({ token, predictions, stats, models, onRefresh }) {
  const [syncingModels, setSyncingModels] = useState(false);
  const [syncError, setSyncError] = useState("");

  const syncModelMetrics = useCallback(async () => {
    setSyncingModels(true);
    setSyncError("");
    try {
      await apiRequest("/models/sync", { method: "POST" }, token);
      await onRefresh();
    } catch (error) {
      setSyncError(error.message);
    } finally {
      setSyncingModels(false);
    }
  }, [token, onRefresh]);

  const confidenceBuckets = useMemo(() => {
    const buckets = { low: 0, medium: 0, high: 0 };
    predictions.forEach((row) => {
      const score = Number(row.ConfidenceScore || 0);
      if (score < 0.6) buckets.low += 1;
      else if (score < 0.8) buckets.medium += 1;
      else buckets.high += 1;
    });
    return buckets;
  }, [predictions]);

  const total = stats?.total || 0;
  const normal = stats?.Normal || 0;
  const critical = stats?.Disease || 0;
  const donutBackground = `conic-gradient(
    var(--normal) 0 ${(total ? (normal / total) * 100 : 0).toFixed(2)}%,
    var(--critical) ${(total ? (normal / total) * 100 : 0).toFixed(2)}% 100%
  )`;

  const modelComparison = useMemo(
    () =>
      models
        .map((model) => ({
          id: model.ModelID,
          name: model.AlgorithmName || "Unknown",
          accuracy: Number(model.Accuracy || 0),
          precision: Number(model.Precision || 0),
          recall: Number(model.Recall || 0),
          f1: Number(model.F1Score || 0),
        }))
        .sort((a, b) => b.accuracy - a.accuracy),
    [models]
  );

  return (
    <section className="view">
      <PanelHeader
        title="Analytics"
        subtitle="Distribution, confidence profile, and latest production model metadata"
      />

      <div className="panel-grid">
        <article className="panel">
          <PanelHeader title="Outcome Mix" subtitle="Relative share by risk class" />
          <div className="donut-wrap">
            <div className="donut" style={{ background: donutBackground }} />
            <div className="donut-labels">
              <div>
                <RiskPill value="Normal" /> <span>{normal}</span>
              </div>
              <div>
                <RiskPill value="Disease" /> <span>{critical}</span>
              </div>
            </div>
          </div>
        </article>

        <article className="panel">
          <PanelHeader
            title="Confidence Bands"
            subtitle="Prediction confidence distribution"
          />
          <div className="risk-list">
            {[
              ["Below 60%", confidenceBuckets.low],
              ["60% to 80%", confidenceBuckets.medium],
              ["Above 80%", confidenceBuckets.high],
            ].map(([label, value]) => (
              <div className="risk-row" key={label}>
                <div className="risk-row-head">
                  <span>{label}</span>
                  <span>{value}</span>
                </div>
                <div className="bar-track">
                  <div
                    className="bar-fill neutral"
                    style={{ width: `${total ? (value / total) * 100 : 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="panel full-span">
          <PanelHeader
            title="Production Model Metrics"
            subtitle="Accuracy, precision, recall, and F1 score from research_training_outputs2"
          />
          {modelComparison.length === 0 ? (
            <p className="empty">No model metrics available yet.</p>
          ) : (
            <div className="metric-chart-grid">
              {modelComparison.map((row) => (
                <div className="metric-chart-group" key={row.id}>
                  <h4>{row.name}</h4>
                  {[
                    ["Accuracy", row.accuracy, "metric-accuracy"],
                    ["Precision", row.precision, "metric-precision"],
                    ["Recall", row.recall, "metric-recall"],
                    ["F1 Score", row.f1, "metric-f1"],
                  ].map(([label, value, tone]) => (
                    <div className="metric-chart-row" key={`${row.id}-${label}`}>
                      <span>{label}</span>
                      <div className="bar-track">
                        <div className={`bar-fill ${tone}`} style={{ width: `${value * 100}%` }} />
                      </div>
                      <strong>{formatPercent(value, 2)}</strong>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </article>

        <article className="panel full-span">
          <PanelHeader
            title="Model Registry"
            subtitle="Server-side entries synced from research_training_outputs2"
            actions={
              <button
                className="btn btn-ghost"
                onClick={syncModelMetrics}
                disabled={syncingModels}
                type="button"
              >
                {syncingModels ? "Syncing..." : "Sync Model Metrics"}
              </button>
            }
          />
          {syncError ? <div className="form-error">{syncError}</div> : null}
          {models.length === 0 ? (
            <p className="empty">No model metadata available for this account.</p>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Version</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Source</th>
                    <th>Registered</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => (
                    <tr key={model.ModelID}>
                      <td>{model.AlgorithmName}</td>
                      <td>{model.ModelVersion || "--"}</td>
                      <td>
                        {model.Accuracy === null || model.Accuracy === undefined
                          ? "--"
                          : formatPercent(Number(model.Accuracy), 2)}
                      </td>
                      <td>
                        {model.Precision === null || model.Precision === undefined
                          ? "--"
                          : formatPercent(Number(model.Precision), 2)}
                      </td>
                      <td>
                        {model.Recall === null || model.Recall === undefined
                          ? "--"
                          : formatPercent(Number(model.Recall), 2)}
                      </td>
                      <td>
                        {model.F1Score === null || model.F1Score === undefined
                          ? "--"
                          : formatPercent(Number(model.F1Score), 2)}
                      </td>
                      <td>{compactModelSource(model.CheckpointPath)}</td>
                      <td>{parseApiDate(model.CreatedAt)?.toLocaleDateString() || "--"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>
      </div>
    </section>
  );
}

export default function App() {
  const [token, setToken] = useState(() => readStoredSession()?.token || null);
  const [user, setUser] = useState(() => readStoredSession()?.user || null);
  const [view, setView] = useState("dashboard");
  const [theme, setTheme] = useState(readStoredTheme);
  const [authNotice, setAuthNotice] = useState("");

  const [patients, setPatients] = useState([]);
  const [stats, setStats] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [models, setModels] = useState([]);
  const [health, setHealth] = useState("unknown");
  const [notifications, setNotifications] = useState([]);

  const [refreshing, setRefreshing] = useState(false);

  const notify = useCallback((title, message, tone = "neutral") => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setNotifications((prev) => [...prev.slice(-3), { id, title, message, tone }]);
    setTimeout(() => {
      setNotifications((prev) => prev.filter((toast) => toast.id !== id));
    }, 4500);
  }, []);

  const expireSession = useCallback((message = "Your session expired. Please sign in again.") => {
    setToken(null);
    setUser(null);
    setPatients([]);
    setStats(null);
    setPredictions([]);
    setModels([]);
    setAuthNotice(message === "Token expired" ? "Your session expired. Please sign in again." : message);
    localStorage.removeItem(SESSION_KEY);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Theme persistence is optional.
    }
  }, [theme]);

  const refreshData = useCallback(async () => {
    if (!token) return;
    setRefreshing(true);
    const [patientsResult, statsResult, predictionsResult, modelsResult] =
      await Promise.allSettled([
        apiRequest("/patients", {}, token),
        apiRequest("/stats", {}, token),
        apiRequest("/predictions?limit=200", {}, token),
        apiRequest("/models", {}, token),
      ]);

    if (patientsResult.status === "fulfilled") setPatients(patientsResult.value || []);
    if (statsResult.status === "fulfilled") setStats(statsResult.value || {});
    if (predictionsResult.status === "fulfilled")
      setPredictions(predictionsResult.value || []);
    if (modelsResult.status === "fulfilled") setModels(modelsResult.value || []);
    else setModels([]);
    const authExpired = [
      patientsResult,
      statsResult,
      predictionsResult,
      modelsResult,
    ].some((result) => result.status === "rejected" && result.reason?.status === 401);
    if (authExpired) {
      expireSession();
    }
    setRefreshing(false);
  }, [token, expireSession]);

  useEffect(() => {
    if (!token) return;
    const timeoutId = setTimeout(() => {
      void refreshData();
    }, 0);
    return () => clearTimeout(timeoutId);
  }, [token, refreshData]);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        await apiRequest("/health");
        if (alive) setHealth("online");
      } catch {
        if (alive) setHealth("offline");
      }
    };
    check();
    const id = setInterval(check, 30000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const handleLogin = (nextToken, nextUser, remember = true) => {
    setToken(nextToken);
    setUser(nextUser);
    setAuthNotice("");
    if (remember) {
      localStorage.setItem(
        SESSION_KEY,
        JSON.stringify({ token: nextToken, user: nextUser })
      );
    } else {
      localStorage.removeItem(SESSION_KEY);
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setPatients([]);
    setStats(null);
    setPredictions([]);
    setModels([]);
    setAuthNotice("");
    localStorage.removeItem(SESSION_KEY);
  };

  const deletePrediction = useCallback(
    async (predictionId) => {
      await apiRequest(`/predictions/${predictionId}`, { method: "DELETE" }, token);
      setPredictions((prev) =>
        prev.filter((row) => row.PredictionID !== predictionId)
      );
      await refreshData();
    },
    [token, refreshData]
  );

  const clearHistory = useCallback(async () => {
    await apiRequest("/predictions", { method: "DELETE" }, token);
    setPredictions([]);
    await refreshData();
  }, [token, refreshData]);

  const deletePatient = useCallback(
    async (patientId) => {
      await apiRequest(`/patients/${patientId}`, { method: "DELETE" }, token);
      notify("Patient deleted", `Patient ID ${patientId} was removed.`, "critical");
      await refreshData();
    },
    [notify, refreshData, token]
  );

  if (!token || !user) return <AuthView onLogin={handleLogin} notice={authNotice} />;

  const isNight = theme === "night";
  return (
    <div className="app-shell" data-theme={theme}>
      <main className="workspace">
        <header className="topbar">
          <div className="topbar-brand">
            <div className="brand-mark" aria-hidden="true">
              <Eye size={20} />
            </div>
            <div>
              <strong>RetinaRisk Nexus</strong>
              <span>Retinal Care Management Platform</span>
            </div>
          </div>
          <div className="system-clock">
            <span className={`status-dot ${health}`} />
            <strong>{health === "online" ? "System Online" : "System Offline"}</strong>
            <span>·</span>
            <time>{new Date().toLocaleString()}</time>
          </div>
          <div className="topbar-actions">
            <button
              className="vision-toggle"
              onClick={() => setTheme((prev) => (prev === "night" ? "day" : "night"))}
              title={isNight ? "Switch to day vision" : "Switch to night vision"}
              type="button"
            >
              {isNight ? <Sun size={17} /> : <Moon size={17} />}
              <span className="sr-only">{isNight ? "Day Vision" : "Night Vision"}</span>
            </button>
            <div className="admin-chip">
              <span>{(user.name || "AD").slice(0, 2).toUpperCase()}</span>
              <div>
                <strong>{user.name || "Admin"}</strong>
                <small>{user.role}</small>
              </div>
            </div>
            <button className="logout-icon" onClick={logout} type="button" aria-label="Sign out">
              <LogOut size={17} />
            </button>
          </div>
        </header>
        <nav className="top-nav" aria-label="Primary">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={`nav-btn ${view === item.id ? "active" : ""}`}
                key={item.id}
                onClick={() => setView(item.id)}
                type="button"
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <section className="workspace-body">
          {view === "dashboard" ? (
            <DashboardView
              token={token}
              patients={patients}
              predictions={predictions}
              onNavigate={setView}
              onDeletePatient={deletePatient}
              notify={notify}
            />
          ) : null}

          {view === "screening" ? (
            <ScreeningView
              token={token}
              patients={patients}
              onPredictionSaved={refreshData}
              notify={notify}
            />
          ) : null}

          {view === "patients" ? (
            <PatientsView
              token={token}
              patients={patients}
              onSaved={refreshData}
              onAuthExpired={expireSession}
              notify={notify}
            />
          ) : null}

          {view === "history" ? (
            <HistoryView
              predictions={predictions}
              onRefresh={refreshData}
              refreshing={refreshing}
              onDeletePrediction={deletePrediction}
              onClearHistory={clearHistory}
            />
          ) : null}

          {view === "analytics" ? (
            <AnalyticsView
              token={token}
              predictions={predictions}
              stats={
              stats || {
                total: 0,
                Normal: 0,
                  Disease: 0,
              }
              }
              models={models}
              onRefresh={refreshData}
            />
          ) : null}

          {view === "medications" ? (
            <MedicationsView patients={patients} predictions={predictions} />
          ) : null}
        </section>
      </main>
      <NotificationToast
        notifications={notifications}
        onDismiss={(id) => setNotifications((prev) => prev.filter((toast) => toast.id !== id))}
      />
    </div>
  );
}
