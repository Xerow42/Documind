"""
Regenerates ml/model_placeholder_synthetic/{model,vectorizer}.pkl from a
small hand-written synthetic dataset (see ml/tests/test_train.py).

This is NOT the real DocuMind model — it exists only so the backend's
classification/keyword tests (backend/tests/test_classification.py,
test_keywords.py) and a local demo of the app have *something* to load
before you've trained on the real dataset. Never present metrics from
this model as DocuMind's real evaluation results.

Usage:
    python ml/generate_placeholder_model.py

For the real model:
    1. Download UpdatedResumeDataSet.csv (see ml/dataset/README.md)
    2. python ml/prepare_dataset.py
    3. python ml/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from test_train import build_synthetic_dataframe  # noqa: E402
from train import save_artifacts, train_and_evaluate  # noqa: E402

if __name__ == "__main__":
    df = build_synthetic_dataframe()
    outcome = train_and_evaluate(df, test_size=0.33, random_state=42)
    out_dir = Path(__file__).resolve().parent / "model_placeholder_synthetic"
    save_artifacts(outcome, model_dir=out_dir)
    print(f"Synthetic placeholder model written to {out_dir}")
    print("Reminder: this is NOT trained on real resumes. See docstring.")
