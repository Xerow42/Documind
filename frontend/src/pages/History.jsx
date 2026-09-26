import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listDocuments, deleteDocument } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";
import LoadingSpinner from "../components/LoadingSpinner";

export default function History() {
  const [documents, setDocuments] = useState(null);
  const [error, setError] = useState(null);

  const load = () => {
    listDocuments()
      .then((data) => setDocuments(data.documents))
      .catch((err) => setError(err.message));
  };

  useEffect(load, []);

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this document and its analysis?")) return;
    try {
      await deleteDocument(id);
      load();
    } catch (err) {
      setError(err.message);
    }
  };

  if (error) return <ErrorBanner message={error} />;
  if (!documents) return <LoadingSpinner label="Loading history..." />;

  return (
    <div>
      <h2>History</h2>
      {documents.length === 0 && <p className="muted">No documents uploaded yet.</p>}
      {documents.length > 0 && (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Filename</th>
                <th>Type</th>
                <th>Uploaded</th>
                <th>Words</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.id}>
                  <td>
                    <Link to={`/documents/${doc.id}`}>{doc.filename}</Link>
                  </td>
                  <td>{doc.file_type.toUpperCase()}</td>
                  <td>{doc.uploaded_at}</td>
                  <td>{doc.word_count}</td>
                  <td>
                    <button className="btn-secondary btn" onClick={() => handleDelete(doc.id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
