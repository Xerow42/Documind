"""
Tests the classifier-loading and prediction path using the SYNTHETIC
placeholder model trained on hand-written fixture sentences (see
ml/tests/test_train.py) — proves the loading/inference code path works
end-to-end. It does NOT prove real-world classification accuracy; that
requires the real dataset (see README "Known Limitations").
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.classification import Classifier, ModelNotLoadedError

PLACEHOLDER_MODEL_DIR = Path(__file__).resolve().parents[2] / "ml" / "model_placeholder_synthetic"


class TestClassifierMissingModel(unittest.TestCase):
    def test_raises_clear_error_when_model_absent(self):
        clf = Classifier(model_dir=Path("/tmp/does_not_exist_documind"))
        with self.assertRaises(ModelNotLoadedError):
            clf.load()


@unittest.skipUnless(
    PLACEHOLDER_MODEL_DIR.exists(),
    "Placeholder synthetic model not found — run ml/tests/test_train.py's "
    "save_artifacts step first (see ml/README workflow).",
)
class TestClassifierWithPlaceholderModel(unittest.TestCase):
    def setUp(self):
        self.clf = Classifier(model_dir=PLACEHOLDER_MODEL_DIR)
        self.clf.load()

    def test_loads_successfully(self):
        self.assertTrue(self.clf.is_loaded())

    def test_predict_returns_known_category(self):
        outcome = self.clf.predict(
            "Managed CI CD pipelines using Jenkins and containerized applications with Kubernetes"
        )
        self.assertIn(
            outcome.category,
            {
                "Data Science", "Data Engineering", "Software Engineering",
                "Web Development", "Cybersecurity", "Database Administration",
                "DevOps / Cloud Computing", "Business Analysis", "Human Resources",
            },
        )

    def test_confidence_is_bounded(self):
        outcome = self.clf.predict("Administered MySQL and PostgreSQL databases")
        self.assertGreaterEqual(outcome.confidence, 0.0)
        self.assertLessEqual(outcome.confidence, 1.0)

    def test_confidence_note_present(self):
        outcome = self.clf.predict("Gathered business requirements and wrote user stories")
        self.assertTrue(len(outcome.confidence_note) > 0)

    def test_lazy_loads_if_not_loaded_yet(self):
        fresh_clf = Classifier(model_dir=PLACEHOLDER_MODEL_DIR)
        self.assertFalse(fresh_clf.is_loaded())
        outcome = fresh_clf.predict("Built responsive websites with HTML and CSS")
        self.assertTrue(fresh_clf.is_loaded())
        self.assertIsNotNone(outcome.category)


if __name__ == "__main__":
    unittest.main()
