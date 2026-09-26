"""
FastAPI integration test.

NOTE: requires `pip install -r backend/requirements.txt` (fastapi,
httpx). These packages could not be installed in the offline sandbox
this project was built in, so this file is written and reviewed
carefully but NOT executed there — see README "Known Limitations".
Run it yourself with: pytest backend/tests/test_health.py
"""
from fastapi.testclient import TestClient

from app.main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
