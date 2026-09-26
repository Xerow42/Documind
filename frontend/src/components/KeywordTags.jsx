import React from "react";

export default function KeywordTags({ keywords }) {
  if (!keywords || keywords.length === 0) {
    return <p className="muted">No keywords extracted.</p>;
  }
  return (
    <div>
      {keywords.map((k) => (
        <span key={k.term} className="keyword-tag">
          {k.term}
        </span>
      ))}
    </div>
  );
}
