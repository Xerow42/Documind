"""
Full API integration tests via FastAPI's TestClient.

NOTE: requires `pip install -r backend/requirements.txt` (fastapi,
httpx) and a trained model (`python ml/prepare_dataset.py && python
ml/train.py`, or point MODEL_DIR at the placeholder synthetic model
for a quick local smoke test). Not executed in the build sandbox — see
README "Known Limitations". Run with: pytest backend/tests/test_api_documents.py
"""
import io
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    tmp_db = Path(tempfile.mkdtemp()) / "test.db"
    monkeypatch.setenv("DATABASE_PATH", str(tmp_db))

    from app.main import app
    with TestClient(app) as c:
        yield c


def make_pdf_bytes(text: str) -> bytes:
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


class TestUpload:
    def test_upload_txt_succeeds(self, client):
        resp = client.post(
            "/documents",
            files={"file": ("resume.txt", b"Python developer with Flask and SQL", "text/plain")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "resume.txt"
        assert body["file_type"] == "txt"
        assert body["word_count"] > 0

    def test_upload_pdf_succeeds(self, client):
        pdf_bytes = make_pdf_bytes("DevOps Engineer with Kubernetes experience")
        resp = client.post(
            "/documents",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 201
        assert resp.json()["file_type"] == "pdf"

    def test_upload_rejects_invalid_extension(self, client):
        resp = client.post(
            "/documents",
            files={"file": ("resume.docx", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 422

    def test_upload_rejects_empty_file(self, client):
        resp = client.post(
            "/documents",
            files={"file": ("resume.txt", b"", "text/plain")},
        )
        assert resp.status_code == 422


class TestAnalyze:
    def test_analyze_uploaded_document(self, client):
        upload = client.post(
            "/documents",
            files={"file": ("resume.txt", b"Administered MySQL databases and wrote backup scripts", "text/plain")},
        )
        doc_id = upload.json()["id"]

        resp = client.post(f"/documents/{doc_id}/analyze")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["classification"]["predicted_category"]
        assert 0.0 <= body["classification"]["confidence"] <= 1.0
        assert isinstance(body["keywords"], list)

    def test_analyze_nonexistent_document_returns_404(self, client):
        resp = client.post("/documents/999999/analyze")
        assert resp.status_code == 404


class TestListAndDetail:
    def test_list_documents(self, client):
        client.post("/documents", files={"file": ("a.txt", b"Java developer with Spring Boot", "text/plain")})
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert len(resp.json()["documents"]) >= 1

    def test_get_document_detail(self, client):
        upload = client.post("/documents", files={"file": ("a.txt", b"HR recruitment and onboarding", "text/plain")})
        doc_id = upload.json()["id"]
        resp = client.get(f"/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["raw_text"]

    def test_get_nonexistent_document_returns_404(self, client):
        resp = client.get("/documents/999999")
        assert resp.status_code == 404


class TestDelete:
    def test_delete_document(self, client):
        upload = client.post("/documents", files={"file": ("a.txt", b"Web designer with HTML and CSS", "text/plain")})
        doc_id = upload.json()["id"]
        resp = client.delete(f"/documents/{doc_id}")
        assert resp.status_code == 204
        assert client.get(f"/documents/{doc_id}").status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/documents/999999")
        assert resp.status_code == 404
