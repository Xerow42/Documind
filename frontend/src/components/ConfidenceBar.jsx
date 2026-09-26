import React from "react";

export default function ConfidenceBar({ confidence, note }) {
  const pct = Math.round((confidence ?? 0) * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span className="muted">Confidence</span>
        <strong>{pct}%</strong>
      </div>
      <div className="confidence-bar-track">
        <div className="confidence-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      {note && <p className="muted" style={{ marginTop: 6 }}>{note}</p>}
    </div>
  );
}
