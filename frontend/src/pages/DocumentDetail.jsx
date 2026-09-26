import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getDocumentDetail, analyzeDocument } from "../api/client";
import CategoryBadge from "../components/CategoryBadge";
import ConfidenceBar from "../components/ConfidenceBar";
import KeywordTags from "../components/KeywordTags";
import StatGrid from "../components/StatGrid";
import ErrorBanner from "../components/ErrorBanner";
import LoadingSpinner from "../components/LoadingSpinner";

export default function DocumentDetail() {
  const { id } = useParams();
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);
  const [reanalyzing, setReanalyzing] = useState(false);
  const navigate = useNavigate();

  const load = () => {
    getDocumentDetail(id)
      .then(setDoc)
      .catch((err) => setError(err.message));
  };

  useEffect(load, [id]);

  const handleReanalyze = async () => {
    setReanalyzing(true);
    setError(null);
    try {
      await analyzeDocument(id);
      navigate(`/results/${id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setReanalyzing(false);
    }
  };

  if (error) return <ErrorBanner message={error} />;
  if (!doc) return <LoadingSpinner label="Loading document..." />;

  const analysis = doc.analysis;

  return (
    <div>
      <h2>{doc.filename}</h2>
      <p className="muted">Uploaded {doc.uploaded_at}</p>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Document statistics</h3>
        <StatGrid
          charCount={doc.char_count}
          wordCount={doc.word_count}
          sentenceCount={doc.sentence_count}
        />
      </div>

      {analysis && analysis.status === "completed" && analysis.classification ? (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Latest analysis</h3>
          <CategoryBadge category={analysis.classification.predicted_category} />
          <div style={{ marginTop: 12 }}>
            <ConfidenceBar
              confidence={analysis.classification.confidence}
              note={analysis.classification.confidence_note}
            />
          </div>
          <div style={{ marginTop: 12 }}>
            <KeywordTags keywords={analysis.keywords} />
          </div>
        </div>
      ) : (
        <div className="card">
          <p className="muted">No completed analysis for this document yet.</p>
        </div>
      )}

      <button className="btn" onClick={handleReanalyze} disabled={reanalyzing}>
        {reanalyzing ? "Analyzing..." : "Re-run analysis"}
      </button>

      <div className="card" style={{ marginTop: 20 }}>
        <h3 style={{ marginTop: 0 }}>Extracted text</h3>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 13, maxHeight: 300, overflowY: "auto" }}>
          {doc.raw_text}
        </pre>
      </div>
    </div>
  );
}
