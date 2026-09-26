"""
Document endpoints: upload, analyze, list, detail, delete.

Routes are intentionally thin — all real logic lives in
app/services/*.py and app/db/repository.py, which are unit-tested
independently of FastAPI (see backend/tests/). That split is also why
this file could be written and reasoned about correctly even in an
environment where FastAPI itself couldn't be installed to run it.
"""
import logging

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.config import settings
from app.db import database, repository
from app.models.schemas import (
    AnalysisOut,
    ClassificationOut,
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    KeywordOut,
)
from app.services.classification import ModelNotLoadedError, classifier, confidence_note_for_model
from app.services.keywords import extract_keywords
from app.services.preprocessing import compute_stats
from app.services.text_extraction import ValidationError, process_upload

logger = logging.getLogger("documind.documents")
router = APIRouter(prefix="/documents", tags=["documents"])


def get_conn():
    return database.get_connection(settings.database_path)


@router.post("", response_model=DocumentSummary, status_code=201)
async def upload_document(file: UploadFile):
    content = await file.read()

    try:
        extracted = process_upload(file.filename, content)
    except ValidationError as exc:
        logger.info("Upload rejected for %s: %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc))

    stats = compute_stats(extracted.raw_text)

    conn = get_conn()
    try:
        doc_id = repository.insert_document(
            conn,
            filename=extracted.filename,
            file_type=extracted.file_type,
            raw_text=extracted.raw_text,
            **stats,
        )
        doc = repository.get_document(conn, doc_id)
    finally:
        conn.close()

    logger.info("Document %s uploaded (%s, %d chars)", doc_id, extracted.file_type, stats["char_count"])
    return DocumentSummary(**dict(doc))


@router.post("/{document_id}/analyze", response_model=AnalysisOut)
def analyze_document(document_id: int):
    conn = get_conn()
    try:
        doc = repository.get_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        analysis_id = repository.create_analysis(conn, document_id)

        try:
            outcome = classifier.predict(doc["raw_text"])
            keywords = extract_keywords(doc["raw_text"], classifier.vectorizer, top_n=8)

            repository.save_classification_result(
                conn, analysis_id, outcome.category, outcome.confidence, outcome.model_version
            )
            repository.save_keywords(conn, analysis_id, keywords)
            repository.complete_analysis(conn, analysis_id)

        except ModelNotLoadedError as exc:
            repository.fail_analysis(conn, analysis_id, str(exc))
            logger.error("Analysis %s failed: model not loaded (%s)", analysis_id, exc)
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 — convert any unexpected failure into a safe response
            repository.fail_analysis(conn, analysis_id, "Internal error during analysis")
            logger.exception("Analysis %s failed unexpectedly", analysis_id)
            raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

        detail = repository.get_document_detail(conn, document_id)
    finally:
        conn.close()

    return _build_analysis_out(detail["analysis"])


@router.get("", response_model=DocumentListResponse)
def list_documents(limit: int = 50, offset: int = 0):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")

    conn = get_conn()
    try:
        rows = repository.list_documents(conn, limit=limit, offset=offset)
    finally:
        conn.close()

    return DocumentListResponse(
        documents=[DocumentSummary(**dict(r)) for r in rows],
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document_detail(document_id: int):
    conn = get_conn()
    try:
        detail = repository.get_document_detail(conn, document_id)
    finally:
        conn.close()

    if detail is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    analysis_out = _build_analysis_out(detail["analysis"]) if detail["analysis"] else None
    return DocumentDetail(
        id=detail["id"],
        filename=detail["filename"],
        file_type=detail["file_type"],
        uploaded_at=detail["uploaded_at"],
        char_count=detail["char_count"],
        word_count=detail["word_count"],
        sentence_count=detail["sentence_count"],
        raw_text=detail["raw_text"],
        analysis=analysis_out,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: int):
    conn = get_conn()
    try:
        deleted = repository.delete_document(conn, document_id)
    finally:
        conn.close()

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")


def _build_analysis_out(analysis: dict) -> AnalysisOut:
    classification_out = None
    if analysis.get("classification"):
        c = analysis["classification"]
        classification_out = ClassificationOut(
            predicted_category=c["predicted_category"],
            confidence=c["confidence"],
            confidence_note=confidence_note_for_model(c["model_version"]),
            model_version=c["model_version"],
        )
    return AnalysisOut(
        id=analysis["id"],
        status=analysis["status"],
        created_at=analysis["created_at"],
        completed_at=analysis.get("completed_at"),
        error_message=analysis.get("error_message"),
        classification=classification_out,
        keywords=[KeywordOut(**k) for k in analysis.get("keywords", [])],
    )
