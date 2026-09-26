"""
Extracts the top-N keywords for a single document by reusing the
already-fitted TF-IDF vectorizer's weights on that one document —
no separate keyword-extraction model to maintain.
"""
from app.services.preprocessing import preprocess


def extract_keywords(raw_text: str, vectorizer, top_n: int = 8) -> list[tuple[str, float]]:
    """Returns [(term, tfidf_score), ...] sorted by score descending.
    `vectorizer` must already be fitted (the same one used for
    classification, loaded once by the Classifier)."""
    cleaned = preprocess(raw_text)
    if not cleaned:
        return []

    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    row = vec.tocoo()

    scored = [(feature_names[col], float(val)) for col, val in zip(row.col, row.data)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]
