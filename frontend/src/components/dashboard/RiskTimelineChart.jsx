const RISK_VALUE = {
  Normal: 1,
  "At-Risk": 2,
  "Disease Detected": 3,
};

const VALUE_LABEL = {
  1: "Normal",
  2: "At-Risk",
  3: "Disease",
};

function formatDate(value) {
  const parsed = value ? new Date(String(value).replace(" ", "T")) : null;
  if (!parsed || Number.isNaN(parsed.getTime())) return "Recent";
  return parsed.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function RiskTimelineChart({ predictions, onSelect }) {
  const ordered = [...predictions]
    .filter((row) => row.Timestamp)
    .sort((a, b) => new Date(a.Timestamp) - new Date(b.Timestamp))
    .slice(-8);

  const points = ordered.map((row, index) => {
    const x = ordered.length <= 1 ? 50 : 8 + (index / (ordered.length - 1)) * 84;
    const y = 86 - ((RISK_VALUE[row.PredictionResult] || 1) - 1) * 36;
    return { row, x, y };
  });
  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  return (
    <article className="care-panel timeline-panel">
      <div className="care-panel-title">
        <h3>Patient Risk Timeline</h3>
      </div>
      {points.length === 0 ? (
        <p className="empty">No screening history for the selected patient.</p>
      ) : (
        <>
          <svg className="risk-timeline-chart" viewBox="0 0 100 100" role="img">
            {[1, 2, 3].map((value) => (
              <g key={value}>
                <line x1="8" x2="94" y1={86 - (value - 1) * 36} y2={86 - (value - 1) * 36} />
                <text x="1" y={89 - (value - 1) * 36}>{VALUE_LABEL[value]}</text>
              </g>
            ))}
            <path d={path} />
            {points.map((point) => (
              <circle
                className="timeline-point"
                key={point.row.PredictionID}
                cx={point.x}
                cy={point.y}
                r="3.3"
                onClick={() => onSelect(point.row)}
              />
            ))}
          </svg>
          <div className="timeline-labels">
            {points.map((point) => (
              <button key={point.row.PredictionID} type="button" onClick={() => onSelect(point.row)}>
                {formatDate(point.row.Timestamp)}
              </button>
            ))}
          </div>
        </>
      )}
    </article>
  );
}
