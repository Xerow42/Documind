import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# ADAPTE le nom de la fonction à celle de ton ml/preprocessing.py
from preprocessing import preprocess

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "dataset" / "documind_dataset.csv")
df["clean"] = df["text"].apply(preprocess)

# 1. Split : le test n'est touché qu'une seule fois, à la fin
X_tr, X_te, y_tr, y_te = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

def tfidf():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "linear_svm": LinearSVC(class_weight="balanced"),
    "multinomial_nb": MultinomialNB(),
}

# 2. Choix du modèle par validation croisée sur le train uniquement
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, clf in models.items():
    s = cross_val_score(make_pipeline(tfidf(), clf), X_tr, y_tr, cv=cv, scoring="f1_macro")
    cv_scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
    print(f"{name}: macro-F1 CV = {s.mean():.3f} ± {s.std():.3f}")

best = max(cv_scores, key=lambda n: cv_scores[n]["mean"])
print(f"\nMeilleur modèle (choisi par CV) : {best}")

# 3. Entraînement final sur tout le train + évaluation test (une seule fois)
vec = tfidf()
Xtr_vec = vec.fit_transform(X_tr)
clf = models[best].fit(Xtr_vec, y_tr)
y_pred = clf.predict(vec.transform(X_te))

labels = sorted(df["label"].unique())
report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_te, y_pred, labels=labels)
print("\n=== Évaluation sur le jeu de test ===")
print(classification_report(y_te, y_pred, zero_division=0))

# 4. Sauvegarde
joblib.dump(clf, MODEL_DIR / "model.pkl")
joblib.dump(vec, MODEL_DIR / "vectorizer.pkl")
(MODEL_DIR / "metrics.json").write_text(json.dumps({
    "n_samples": len(df), "n_train": len(X_tr), "n_test": len(X_te),
    "labels": labels, "best_model": best, "cv_macro_f1": cv_scores,
    "test_report": report, "confusion_matrix": cm.tolist(),
}, indent=2))
print(f"Modèle et métriques écrits dans {MODEL_DIR}")