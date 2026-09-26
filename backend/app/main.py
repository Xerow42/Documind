"""
DocuMind API entrypoint.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db
from app.routers import documents
from app.services.classification import ModelNotLoadedError, classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("documind")

app = FastAPI(
    title="DocuMind API",
    description="AI-powered resume classification and analysis platform.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db(settings.database_path)
    try:
        classifier.load()
        logger.info("Classifier loaded (%s)", classifier._model_version)
    except ModelNotLoadedError as exc:
        # Don't crash the whole app: /health, /documents (upload/list) still
        # work without a trained model — only /analyze needs it. Run
        # `python ml/prepare_dataset.py && python ml/train.py` to fix.
        logger.warning("Classifier not loaded at startup: %s", exc)


@app.get("/health")
def health_check():
    """Basic liveness check used by tests and local dev."""
    return {"status": "ok", "model_loaded": classifier.is_loaded()}


app.include_router(documents.router)
