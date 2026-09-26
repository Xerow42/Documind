"""
Extracts text from uploaded resume files and validates them before
anything touches the database or the model.
"""
import io
from dataclasses import dataclass

import pdfplumber

ALLOWED_EXTENSIONS = {"pdf", "txt"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class ValidationError(Exception):
    """Raised for any user-facing upload validation failure. The API
    layer catches this and returns a 422 with the message — never a
    raw stack trace."""


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    raw_text: str


def validate_filename(filename: str) -> str:
    if not filename or "." not in filename:
        raise ValidationError("File must have an extension (.pdf or .txt).")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '.{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    return ext


def validate_size(content: bytes) -> None:
    if len(content) == 0:
        raise ValidationError("The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
            f"Max size is {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )


def extract_text_from_pdf(content: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            # x_tolerance=1 (pdfplumber's default is 3): tag/chip-style skill
            # lists and tightly-kerned layouts (common in LaTeX-typeset CVs)
            # can have only ~2pt of gap between adjacent words, which the
            # default tolerance merges into one run (e.g. "keras tensorflow"
            # -> "kerastensorflow"). A tighter tolerance fixes that without
            # breaking normal prose (verified: does not fragment ordinary
            # sentences into individual letters).
            pages = [page.extract_text(x_tolerance=1) or "" for page in pdf.pages]
    except Exception as exc:  # pdfplumber/pdfminer raise assorted exceptions on corrupt files
        raise ValidationError(f"Could not read this PDF — it may be corrupted or scanned/image-only. ({exc})")
    text = "\n".join(pages).strip()
    if not text:
        raise ValidationError(
            "No extractable text found in this PDF. If it's a scanned "
            "image, OCR isn't supported yet — try a text-based PDF or a .txt file."
        )
    return text


def extract_text_from_txt(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError:
            raise ValidationError("Could not decode this .txt file as UTF-8 or Latin-1 text.")
    text = text.strip()
    if not text:
        raise ValidationError("The uploaded .txt file is empty.")
    return text


def process_upload(filename: str, content: bytes) -> ExtractedDocument:
    """Single entry point the API route calls: validates, then extracts.
    Raises ValidationError with a user-safe message on any problem."""
    file_type = validate_filename(filename)
    validate_size(content)

    if file_type == "pdf":
        text = extract_text_from_pdf(content)
    else:
        text = extract_text_from_txt(content)

    return ExtractedDocument(filename=filename, file_type=file_type, raw_text=text)
