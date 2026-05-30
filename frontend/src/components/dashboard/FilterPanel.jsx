import { Filter, Search } from "lucide-react";

export default function FilterPanel({ filters, onChange }) {
  const update = (key, value) => onChange({ ...filters, [key]: value });

  return (
    <section className="dashboard-filter-panel" aria-label="Dashboard filters">
      <div className="filter-title">
        <Filter size={16} />
        <span>Filters</span>
      </div>
      <label className="filter-search">
        <Search size={15} />
        <input
          value={filters.query}
          onChange={(event) => update("query", event.target.value)}
          placeholder="Search patient"
        />
      </label>
      <select value={filters.risk} onChange={(event) => update("risk", event.target.value)}>
        <option value="all">All risk levels</option>
        <option value="Normal">Normal</option>
        <option value="At-Risk">At-Risk</option>
        <option value="Disease Detected">Disease Detected</option>
      </select>
      <select value={filters.time} onChange={(event) => update("time", event.target.value)}>
        <option value="all">All time</option>
        <option value="today">Today</option>
        <option value="week">This week</option>
        <option value="month">This month</option>
      </select>
      <label className="confidence-toggle">
        <input
          type="checkbox"
          checked={filters.highConfidenceOnly}
          onChange={(event) => update("highConfidenceOnly", event.target.checked)}
        />
        <span>High confidence only</span>
      </label>
    </section>
  );
}
