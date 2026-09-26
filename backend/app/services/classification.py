"""
Loads the trained TF-IDF vectorizer + classifier and predicts a
category + confidence for a given resume text.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib

from app.services.preprocessing import preprocess

MODEL_DIR = Path(__file__).resolve().parents[3] / "ml" / "model"


class ModelNotLoadedError(Exception):
    """Raised when model.pkl/vectorizer.pkl aren't present yet — i.e.
    `python ml/train.py` hasn't been run. The API layer turns this into
    a clear 503, not a raw stack trace."""


@dataclass
class ClassificationOutcome:
    category: str
    confidence: float
    model_version: str
    confidence_note: str


def confidence_note_for_model(model_version: str) -> str:
    if "svm" in model_version:
        return (
            "Linear SVM does not produce a true probability — this score is "
            "derived from the model's decision margin and should be read as "
            "relative confidence, not a calibrated percentage."
        )
    return (
        "Confidence is the model's predicted probability for the top "
        "category, on a document possibly very different from the "
        "training data's domain — treat it as a rough signal, not certainty."
    )


class Classifier:
    """Loads the model once and reuses it. A thin wrapper rather than a
    global — makes it trivial to inject a fake/mocked model in tests."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._model = None
        self._vectorizer = None
        self._model_version = None

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self.load()
        return self._vectorizer

    def load(self) -> None:
        model_path = self.model_dir / "model.pkl"
        vectorizer_path = self.model_dir / "vectorizer.pkl"
        metrics_path = self.model_dir / "metrics.json"

        if not model_path.exists() or not vectorizer_path.exists():
            raise ModelNotLoadedError(
                f"No trained model found at {self.model_dir}. Run "
                "`python ml/prepare_dataset.py && python ml/train.py` first."
            )

        self._model = joblib.load(model_path)
        self._vectorizer = joblib.load(vectorizer_path)

        if metrics_path.exists():
            with open(metrics_path) as f:
                self._model_version = json.load(f).get("best_model_name", "unknown")
        else:
            self._model_version = "unknown"

    def predict(self, raw_text: str) -> ClassificationOutcome:
        if not self.is_loaded():
            self.load()

        cleaned = preprocess(raw_text)
        vec = self._vectorizer.transform([cleaned])

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(vec)[0]
            classes = self._model.classes_
            best_idx = probs.argmax()
            category = classes[best_idx]
            confidence = float(probs[best_idx])
        elif hasattr(self._model, "decision_function"):
            # LinearSVC: no predict_proba: use softmax over decision
            # scores as a bounded, comparable stand-in for "confidence"
            # (explicitly caveated to the user — see confidence_note_for_model).
            import numpy as np

            scores = self._model.decision_function(vec)[0]
            classes = self._model.classes_
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            best_idx = probs.argmax()
            category = classes[best_idx]
            confidence = float(probs[best_idx])
        else:
            category = self._model.predict(vec)[0]
            confidence = 1.0  # no scoring mechanism available

        return ClassificationOutcome(
            category=str(category),
            confidence=round(confidence, 4),
            model_version=self._model_version,
            confidence_note=confidence_note_for_model(self._model_version),
        )


# Module-level singleton used by the API layer.
classifier = Classifier()
