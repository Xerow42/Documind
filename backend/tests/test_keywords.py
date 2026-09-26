import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.keywords import extract_keywords

PLACEHOLDER_MODEL_DIR = Path(__file__).resolve().parents[2] / "ml" / "model_placeholder_synthetic"


@unittest.skipUnless(PLACEHOLDER_MODEL_DIR.exists(), "Placeholder synthetic model not found")
class TestExtractKeywords(unittest.TestCase):
    def setUp(self):
        import joblib
        self.vectorizer = joblib.load(PLACEHOLDER_MODEL_DIR / "vectorizer.pkl")

    def test_returns_top_n_keywords(self):
        keywords = extract_keywords(
            "Administered MySQL and PostgreSQL databases including backup strategies",
            self.vectorizer,
            top_n=5,
        )
        self.assertLessEqual(len(keywords), 5)
        self.assertTrue(all(isinstance(term, str) and isinstance(score, float) for term, score in keywords))

    def test_sorted_by_score_descending(self):
        keywords = extract_keywords(
            "Configured firewalls and performed vulnerability assessments on web applications",
            self.vectorizer,
            top_n=8,
        )
        scores = [score for _, score in keywords]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(extract_keywords("", self.vectorizer), [])

    def test_text_with_only_stopwords_returns_empty_list(self):
        self.assertEqual(extract_keywords("the a is of", self.vectorizer), [])


if __name__ == "__main__":
    unittest.main()
