import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadDocument, analyzeDocument } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | analyzing | error
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setError(null);
    setStatus("uploading");
    try {
      const doc = await uploadDocument(file);
      setStatus("analyzing");
      await analyzeDocument(doc.id);
      navigate(`/results/${doc.id}`);
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
      setStatus("error");
    }
  };

  const isBusy = status === "uploading" || status === "analyzing";

  return (
    <div>
      <h2>Upload a resume</h2>
      <p className="muted">PDF or TXT, up to 5 MB.</p>

      <div className="card">
        <ErrorBanner message={error} />
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={handleFileChange}
            disabled={isBusy}
          />
          <div style={{ marginTop: 16 }}>
            <button type="submit" className="btn" disabled={!file || isBusy}>
              {status === "uploading" && "Uploading..."}
              {status === "analyzing" && "Analyzing..."}
              {(status === "idle" || status === "error") && "Upload & Analyze"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
