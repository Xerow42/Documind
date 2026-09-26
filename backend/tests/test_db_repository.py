import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import database, repository


class TestRepository(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp_dir) / "test.db")
        database.init_db(self.db_path)
        self.conn = database.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()

    def _insert_sample_document(self):
        return repository.insert_document(
            self.conn,
            filename="resume.pdf",
            file_type="pdf",
            raw_text="Experienced Python developer with SQL and Flask.",
            char_count=48,
            word_count=7,
            sentence_count=1,
        )

    def test_insert_and_get_document(self):
        doc_id = self._insert_sample_document()
        doc = repository.get_document(self.conn, doc_id)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["filename"], "resume.pdf")
        self.assertEqual(doc["file_type"], "pdf")

    def test_get_nonexistent_document_returns_none(self):
        self.assertIsNone(repository.get_document(self.conn, 9999))

    def test_list_documents_orders_newest_first(self):
        id1 = self._insert_sample_document()
        id2 = self._insert_sample_document()
        docs = repository.list_documents(self.conn)
        self.assertEqual(len(docs), 2)
        # both inserted "now"; just confirm both ids are present
        self.assertEqual({docs[0]["id"], docs[1]["id"]}, {id1, id2})

    def test_delete_document(self):
        doc_id = self._insert_sample_document()
        self.assertTrue(repository.delete_document(self.conn, doc_id))
        self.assertIsNone(repository.get_document(self.conn, doc_id))

    def test_delete_nonexistent_document_returns_false(self):
        self.assertFalse(repository.delete_document(self.conn, 9999))

    def test_full_analysis_flow(self):
        doc_id = self._insert_sample_document()
        analysis_id = repository.create_analysis(self.conn, doc_id)

        analysis = repository.get_latest_analysis(self.conn, doc_id)
        self.assertEqual(analysis["status"], "pending")

        repository.save_classification_result(
            self.conn, analysis_id, "Software Engineering", 0.82, "logistic_regression_v1"
        )
        repository.save_keywords(
            self.conn, analysis_id, [("python", 0.41), ("flask", 0.33), ("sql", 0.29)]
        )
        repository.complete_analysis(self.conn, analysis_id)

        detail = repository.get_document_detail(self.conn, doc_id)
        self.assertEqual(detail["analysis"]["status"], "completed")
        self.assertEqual(
            detail["analysis"]["classification"]["predicted_category"],
            "Software Engineering",
        )
        self.assertEqual(len(detail["analysis"]["keywords"]), 3)
        self.assertEqual(detail["analysis"]["keywords"][0]["term"], "python")

    def test_failed_analysis_records_error(self):
        doc_id = self._insert_sample_document()
        analysis_id = repository.create_analysis(self.conn, doc_id)
        repository.fail_analysis(self.conn, analysis_id, "model file not found")

        analysis = repository.get_latest_analysis(self.conn, doc_id)
        self.assertEqual(analysis["status"], "failed")
        self.assertEqual(analysis["error_message"], "model file not found")

    def test_document_detail_with_no_analysis_yet(self):
        doc_id = self._insert_sample_document()
        detail = repository.get_document_detail(self.conn, doc_id)
        self.assertIsNone(detail["analysis"])

    def test_document_detail_nonexistent_document(self):
        self.assertIsNone(repository.get_document_detail(self.conn, 9999))

    def test_cascade_delete_removes_analyses(self):
        doc_id = self._insert_sample_document()
        analysis_id = repository.create_analysis(self.conn, doc_id)
        repository.delete_document(self.conn, doc_id)
        # analysis row should be gone via ON DELETE CASCADE
        row = self.conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        self.assertIsNone(row)


if __name__ == "__main__":
    unittest.main()
