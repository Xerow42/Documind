import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.text_extraction import (
    ValidationError,
    extract_text_from_pdf,
    extract_text_from_txt,
    process_upload,
    validate_filename,
    validate_size,
)


def make_test_pdf(text: str) -> bytes:
    """Generates a real, minimal single-page PDF containing `text`,
    using reportlab — so extract_text_from_pdf is tested against an
    actual PDF binary, not a mock."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


class TestValidateFilename(unittest.TestCase):
    def test_accepts_pdf_and_txt(self):
        self.assertEqual(validate_filename("resume.pdf"), "pdf")
        self.assertEqual(validate_filename("resume.txt"), "txt")

    def test_rejects_other_extensions(self):
        with self.assertRaises(ValidationError):
            validate_filename("resume.docx")

    def test_rejects_no_extension(self):
        with self.assertRaises(ValidationError):
            validate_filename("resume")

    def test_case_insensitive(self):
        self.assertEqual(validate_filename("Resume.PDF"), "pdf")


class TestValidateSize(unittest.TestCase):
    def test_rejects_empty_file(self):
        with self.assertRaises(ValidationError):
            validate_size(b"")

    def test_rejects_oversized_file(self):
        with self.assertRaises(ValidationError):
            validate_size(b"x" * (6 * 1024 * 1024))

    def test_accepts_normal_file(self):
        validate_size(b"some resume content")  # should not raise


class TestExtractTxt(unittest.TestCase):
    def test_extracts_utf8_text(self):
        text = extract_text_from_txt("Python developer with SQL skills".encode("utf-8"))
        self.assertIn("Python", text)

    def test_rejects_empty_txt(self):
        with self.assertRaises(ValidationError):
            extract_text_from_txt(b"   ")

    def test_falls_back_to_latin1(self):
        # a byte sequence invalid as UTF-8 but valid as latin-1
        raw = "café resumé".encode("latin-1")
        text = extract_text_from_txt(raw)
        self.assertIn("caf", text)


class TestExtractPdf(unittest.TestCase):
    def test_extracts_text_from_real_pdf(self):
        pdf_bytes = make_test_pdf("Data Scientist with Python and SQL experience")
        text = extract_text_from_pdf(pdf_bytes)
        self.assertIn("Data Scientist", text)
        self.assertIn("Python", text)

    def test_rejects_corrupted_pdf(self):
        with self.assertRaises(ValidationError):
            extract_text_from_pdf(b"this is not a real pdf file")

    def test_rejects_pdf_with_no_extractable_text(self):
        # a syntactically valid but blank PDF page -> no text to extract
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.showPage()
        c.save()
        with self.assertRaises(ValidationError):
            extract_text_from_pdf(buf.getvalue())


class TestProcessUpload(unittest.TestCase):
    def test_full_txt_upload_flow(self):
        doc = process_upload("resume.txt", b"Java developer with Spring Boot experience")
        self.assertEqual(doc.file_type, "txt")
        self.assertIn("Java", doc.raw_text)

    def test_rejects_bad_extension_before_reading_content(self):
        with self.assertRaises(ValidationError):
            process_upload("resume.exe", b"anything")

    def test_rejects_empty_upload(self):
        with self.assertRaises(ValidationError):
            process_upload("resume.txt", b"")

    def test_full_pdf_upload_flow(self):
        pdf_bytes = make_test_pdf("DevOps Engineer skilled in CI/CD and Kubernetes")
        doc = process_upload("resume.pdf", pdf_bytes)
        self.assertEqual(doc.file_type, "pdf")
        self.assertIn("DevOps Engineer", doc.raw_text)


if __name__ == "__main__":
    unittest.main()
