import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listDocuments } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";
import LoadingSpinner from "../components/LoadingSpinner";

export default function Dashboard() {
  const [documents, setDocuments] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    listDocuments(5)
      .then((data) => setDocuments(data.documents))
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div>
      <h2>Dashboard</h2>
      <p className="muted">
        AI-powered resume classification — upload a resume to categorize it into
        an IT/business job field with a confidence score and keywords.
      </p>

      <div className="card">
        <Link to="/upload" className="btn">
          Upload a resume
        </Link>
      </div>

      <h3>Recent documents</h3>
      <ErrorBanner message={error} />
      {!documents && !error && <LoadingSpinner />}
      {documents && documents.length === 0 && (
        <p className="muted">No documents yet — upload your first resume to get started.</p>
      )}
      {documents && documents.length > 0 && (
        <div className="card">
          {documents.map((doc) => (
            <div key={doc.id} style={{ padding: "8px 0", borderBottom: "1px solid var(--color-border)" }}>
              <Link to={`/documents/${doc.id}`}>{doc.filename}</Link>
              <span className="muted" style={{ marginLeft: 8 }}>{doc.uploaded_at}</span>
            </div>
          ))}
          <div style={{ marginTop: 12 }}>
            <Link to="/history">View all →</Link>
          </div>
        </div>
      )}
    </div>
  );
}
