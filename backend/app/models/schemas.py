"""
Pydantic request/response models for the API.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DocumentStats(BaseModel):
    char_count: int
    word_count: int
    sentence_count: int


class DocumentSummary(BaseModel):
    id: int
    filename: str
    file_type: str
    uploaded_at: str
    char_count: int
    word_count: int
    sentence_count: int


class KeywordOut(BaseModel):
    term: str
    score: float


class ClassificationOut(BaseModel):
    predicted_category: str
    confidence: float
    confidence_note: str
    model_version: str


class AnalysisOut(BaseModel):
    id: int
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    classification: Optional[ClassificationOut] = None
    keywords: list[KeywordOut] = []


class DocumentDetail(DocumentSummary):
    raw_text: str
    analysis: Optional[AnalysisOut] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    detail: str
