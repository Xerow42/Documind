import React from "react";

export default function StatGrid({ charCount, wordCount, sentenceCount }) {
  const stats = [
    { label: "Characters", value: charCount },
    { label: "Words", value: wordCount },
    { label: "Sentences", value: sentenceCount },
  ];
  return (
    <div style={{ display: "flex", gap: 24 }}>
      {stats.map((s) => (
        <div key={s.label}>
          <div style={{ fontSize: 20, fontWeight: 700 }}>{s.value ?? "—"}</div>
          <div className="muted">{s.label}</div>
        </div>
      ))}
    </div>
  );
}
