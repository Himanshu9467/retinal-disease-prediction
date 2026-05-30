const BANDS = [
  { key: "low", label: "Low", tone: "neutral", min: 0, max: 0.6 },
  { key: "medium", label: "Medium", tone: "atrisk", min: 0.6, max: 0.8 },
  { key: "high", label: "High", tone: "normal", min: 0.8, max: Infinity },
];

export default function ConfidenceChart({ predictions }) {
  const counts = BANDS.map((band) => ({
    ...band,
    count: predictions.filter((row) => {
      const score = Number(row.ConfidenceScore || 0);
      return score >= band.min && score < band.max;
    }).length,
  }));
  const total = Math.max(1, predictions.length);

  return (
    <article className="care-panel confidence-chart-panel">
      <div className="care-panel-title">
        <h3>Prediction Confidence</h3>
      </div>
      <div className="confidence-bars">
        {counts.map((band) => {
          const width = Math.round((band.count / total) * 100);
          return (
            <div className="confidence-row" key={band.key}>
              <div>
                <span>{band.label}</span>
                <strong>{band.count}</strong>
              </div>
              <div className="bar-track">
                <div className={`bar-fill ${band.tone}`} style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}
