import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getDocumentDetail } from "../api/client";
import CategoryBadge from "../components/CategoryBadge";
import ConfidenceBar from "../components/ConfidenceBar";
import KeywordTags from "../components/KeywordTags";
import StatGrid from "../components/StatGrid";
import ErrorBanner from "../components/ErrorBanner";
import LoadingSpinner from "../components/LoadingSpinner";

export default function AnalysisResult() {
  const { id } = useParams();
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getDocumentDetail(id)
      .then((data) => !cancelled && setDoc(data))
      .catch((err) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) return <ErrorBanner message={error} />;
  if (!doc) return <LoadingSpinner label="Loading analysis..." />;

  const analysis = doc.analysis;

  return (
    <div>
      <h2>Analysis result</h2>
      <p className="muted">{doc.filename}</p>

      {!analysis && <div className="card">No analysis yet for this document.</div>}

      {analysis && analysis.status === "failed" && (
        <ErrorBanner message={`Analysis failed: ${analysis.error_message}`} />
      )}

      {analysis && analysis.status === "completed" && analysis.classification && (
        <>
          <div className="card">
            <div style={{ marginBottom: 16 }}>
              <span className="muted">Predicted category</span>
              <div style={{ marginTop: 4 }}>
                <CategoryBadge category={analysis.classification.predicted_category} />
              </div>
            </div>
            <ConfidenceBar
              confidence={analysis.classification.confidence}
              note={analysis.classification.confidence_note}
            />
            <p className="muted" style={{ marginTop: 8 }}>
              Model: {analysis.classification.model_version}
            </p>
          </div>

          <div className="card">
            <h3 style={{ marginTop: 0 }}>Keywords</h3>
            <KeywordTags keywords={analysis.keywords} />
          </div>
        </>
      )}

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Document statistics</h3>
        <StatGrid
          charCount={doc.char_count}
          wordCount={doc.word_count}
          sentenceCount={doc.sentence_count}
        />
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Extracted text</h3>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 13, maxHeight: 300, overflowY: "auto" }}>
          {doc.raw_text}
        </pre>
      </div>

      <Link to="/history" className="btn-secondary btn">
        Back to history
      </Link>
    </div>
  );
}
