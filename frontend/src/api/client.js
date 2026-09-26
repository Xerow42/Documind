const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_) {
      // response wasn't JSON — keep the generic message
    }
    throw new Error(detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

export function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/documents", { method: "POST", body: formData });
}

export function analyzeDocument(documentId) {
  return request(`/documents/${documentId}/analyze`, { method: "POST" });
}

export function listDocuments(limit = 50, offset = 0) {
  return request(`/documents?limit=${limit}&offset=${offset}`);
}

export function getDocumentDetail(documentId) {
  return request(`/documents/${documentId}`);
}

export function deleteDocument(documentId) {
  return request(`/documents/${documentId}`, { method: "DELETE" });
}
