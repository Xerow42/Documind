"""
Trains and compares three TF-IDF-based classifiers (Logistic Regression,
Linear SVM, Multinomial Naive Bayes), evaluates them on a held-out test
split, and saves the best one (by macro-F1) for the backend to load.

Usage:
    python ml/train.py
    python ml/train.py --dataset ml/dataset/documind_dataset.csv

Outputs:
    ml/model/model.pkl        - the selected best model
    ml/model/vectorizer.pkl   - the fitted TfidfVectorizer
    ml/model/metrics.json     - metrics for all 3 models + which was selected
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset" / "documind_dataset.csv"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError(f"{csv_path} must have 'Category' and 'Resume' columns")
    return df


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    # unigrams + bigrams: bigrams help catch multi-word skills/tools
    # ("machine learning", "network security") that unigrams alone lose.
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Splits, vectorizes, trains all 3 models, evaluates on the same
    test split, and picks the best by macro-F1 (not raw accuracy —
    macro-F1 doesn't let the largest class hide poor performance on
    smaller ones, which matters given our class imbalance)."""

    X_raw = df["Resume"]
    y = df["Category"]

    class_counts = y.value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            f"Every category needs >=2 samples for a stratified split. "
            f"Got: {class_counts.to_dict()}"
        )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = X_train_raw.apply(preprocess)
    X_test = X_test_raw.apply(preprocess)

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "multinomial_nb": MultinomialNB(),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        }
        fitted[name] = model

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    best_model = fitted[best_name]
    best_preds = best_model.predict(X_test_vec)
    labels = sorted(y.unique().tolist())

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "vectorizer": vectorizer,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, best_preds, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test, best_preds, labels=labels, zero_division=0
        ),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_artifacts(outcome: dict, model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(outcome["best_model"], model_dir / "model.pkl")
    joblib.dump(outcome["vectorizer"], model_dir / "vectorizer.pkl")

    meta = {
        "best_model_name": outcome["best_model_name"],
        "selection_criterion": "highest macro-F1 on the held-out test split",
        "results": outcome["results"],
        "labels": outcome["labels"],
        "n_train": outcome["n_train"],
        "n_test": outcome["n_test"],
        "confusion_matrix": outcome["confusion_matrix"],
        "classification_report": outcome["classification_report"],
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(
            f"ERROR: {dataset_path} not found. Run `python ml/prepare_dataset.py` "
            "first (after downloading the real dataset — see ml/dataset/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_dataset(dataset_path)
    outcome = train_and_evaluate(df, test_size=args.test_size)
    save_artifacts(outcome)

    print(f"Train size: {outcome['n_train']}  Test size: {outcome['n_test']}")
    print(json.dumps(outcome["results"], indent=2))
    print(f"\nSelected model: {outcome['best_model_name']} (highest macro-F1)")
    print(f"\nSaved to {MODEL_DIR}/model.pkl and {MODEL_DIR}/vectorizer.pkl")


if __name__ == "__main__":
    main()
