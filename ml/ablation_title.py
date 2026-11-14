import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from preprocessing import preprocess

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "dataset" / "documind_dataset.csv")

def strip_title(text: str) -> str:
    # Retire tout ce qui précède le premier "Summary" (le titre).
    # Si "Summary" est absent, on garde le texte tel quel.
    m = re.search(r"\bSummary\b", text)
    return text[m.end():] if m else text

print("CV sans 'Summary' trouvé :", (~df["text"].str.contains(r"\bSummary\b")).sum(), "sur", len(df))

X_tr, X_te, y_tr, y_te = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tfidf():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)

for name, fn in [("avec titre", lambda t: t), ("sans titre", strip_title)]:
    X = X_tr.apply(fn).apply(preprocess)
    s = cross_val_score(
        make_pipeline(tfidf(), LinearSVC(class_weight="balanced")),
        X, y_tr, cv=cv, scoring="f1_macro",
    )
    print(f"{name}: macro-F1 CV = {s.mean():.3f} ± {s.std():.3f}")