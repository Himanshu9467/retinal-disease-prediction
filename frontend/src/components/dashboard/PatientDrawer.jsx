import { Download, Edit3, Trash2, X, ZoomIn } from "lucide-react";
import { useEffect } from "react";

const CHECKLIST = [
  ["FollowUpAdvised", "Follow-up advised"],
  ["DoctorReviewed", "Doctor reviewed"],
  ["LifestyleCounseling", "Lifestyle counseling"],
  ["RescreenScheduled", "Re-screen scheduled"],
];

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function mediaUrl(apiBase, token, path) {
  if (!path) return "";
  return `${apiBase}/media?path=${encodeURIComponent(path)}&token=${encodeURIComponent(token)}`;
}

export default function PatientDrawer({
  apiBase,
  token,
  open,
  patient,
  history,
  checklist,
  selectedPrediction,
  onClose,
  onPredictionSelect,
  onChecklistChange,
  onDelete,
  onEdit,
  onDownloadReport,
}) {
  const latest = selectedPrediction || history[0] || null;
  const recommendation =
    latest?.PredictionResult === "Disease Detected"
      ? "Arrange prompt clinician review and confirm the screening result."
      : latest?.PredictionResult === "At-Risk"
        ? "Schedule follow-up and reduce modifiable cardiovascular risk factors."
        : latest?.PredictionResult === "Normal"
          ? "Continue routine screening and healthy prevention habits."
          : "Run screening to generate patient guidance.";
  const imageSrc = mediaUrl(apiBase, token, latest?.ImagePath);
  const heatmapSrc = mediaUrl(apiBase, token, latest?.ExplanationPath);

  const closeDrawer = (event) => {
    event?.preventDefault();
    event?.stopPropagation();
    onClose();
  };

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) return null;

  return (
    <aside className="patient-drawer open" aria-hidden={false}>
      <div className="drawer-backdrop" onClick={closeDrawer} />
      <section className="drawer-panel" onClick={(event) => event.stopPropagation()}>
        <header className="drawer-head">
          <div>
            <span>Patient Detail</span>
            <h2>{patient?.Name || patient?.name || "No patient selected"}</h2>
            <p>ID {patient?.PatientID || patient?.id || "--"} - Age {patient?.Age || patient?.age || "--"}</p>
          </div>
          <button
            type="button"
            className="drawer-close"
            onPointerDown={closeDrawer}
            onClick={closeDrawer}
            aria-label="Close drawer"
          >
            <X size={18} />
          </button>
        </header>

        {patient ? (
          <div className="drawer-content">
            <div className="drawer-actions">
              <button type="button" onClick={onEdit}>
                <Edit3 size={15} />
                Edit patient
              </button>
              <button type="button" onClick={onDownloadReport}>
                <Download size={15} />
                Download report
              </button>
              <button className="danger" type="button" onClick={onDelete}>
                <Trash2 size={15} />
                Delete patient
              </button>
            </div>

            <div className="drawer-metrics">
              <div>
                <span>Latest prediction</span>
                <strong>{latest?.PredictionResult || "Not screened"}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>{latest ? percent(latest.ConfidenceScore) : "--"}</strong>
              </div>
            </div>

            <article className="drawer-section">
              <h3>Retinal Image</h3>
              {imageSrc ? (
                <button
                  className="retinal-media"
                  type="button"
                  onClick={() => window.open(heatmapSrc || imageSrc, "_blank", "noopener,noreferrer")}
                >
                  <img alt="Retinal screening" src={heatmapSrc || imageSrc} />
                  <span>
                    <ZoomIn size={15} />
                    Open zoom
                  </span>
                </button>
              ) : (
                <p className="empty">No retinal image available for this prediction.</p>
              )}
            </article>

            <article className="drawer-section">
              <h3>Care Recommendation</h3>
              <p>{recommendation}</p>
            </article>

            {latest?.PredictionResult && latest.PredictionResult !== "Normal" ? (
              <article className="drawer-section">
                <h3>Doctor Action Checklist</h3>
                <div className="checklist-grid">
                  {CHECKLIST.map(([key, label]) => (
                    <label key={key}>
                      <input
                        type="checkbox"
                        checked={Boolean(checklist?.[key])}
                        onChange={(event) => onChecklistChange(key, event.target.checked)}
                      />
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
              </article>
            ) : null}

            <article className="drawer-section">
              <h3>Previous Screening History</h3>
              <div className="drawer-history">
                {history.length === 0 ? (
                  <p className="empty">No previous screenings.</p>
                ) : (
                  history.map((row) => (
                    <button
                      type="button"
                      key={row.PredictionID}
                      onClick={() => onPredictionSelect(row)}
                      className={latest?.PredictionID === row.PredictionID ? "active" : ""}
                    >
                      <strong>{row.PredictionResult}</strong>
                      <span>{percent(row.ConfidenceScore)} - {row.Timestamp || "Recent"}</span>
                    </button>
                  ))
                )}
              </div>
            </article>
          </div>
        ) : null}
      </section>
    </aside>
  );
}
