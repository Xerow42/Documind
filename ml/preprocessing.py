"""
Shared text preprocessing for training and inference.

Kept intentionally dependency-free (stdlib `re` only) so it can be
imported by both the ml/ training pipeline and the FastAPI backend
without dragging in nltk/spacy for a task this small and explainable.
"""
import re

# Small, explicit English stopword list (kept local/stdlib rather than
# pulling in nltk's corpus, which needs a one-time network download of
# its own — not worth the extra dependency for a TF-IDF baseline).
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "once", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "should", "could", "ought", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "this", "that", "these", "those", "as", "not",
    "no", "nor", "too", "very", "can", "will", "just", "don", "now",
    "also", "will", "shall",
}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails, drop non-alphanumeric characters,
    collapse whitespace. Deterministic and easy to explain in an
    interview — no hidden magic."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)


def preprocess(text: str) -> str:
    """Full pipeline used before vectorization: clean -> stopword removal."""
    return remove_stopwords(clean_text(text))


def compute_stats(raw_text: str) -> dict:
    """Document statistics shown in the UI. Computed on the RAW text
    (not the cleaned/stopword-stripped version) since that's what a
    human means by "word count" of their document."""
    if not raw_text or not raw_text.strip():
        return {"char_count": 0, "word_count": 0, "sentence_count": 0}

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)
    # Simple sentence heuristic: split on ./!/? followed by whitespace or
    # end of string. Good enough for resumes (short, list-heavy text);
    # a full sentence tokenizer (e.g. nltk.punkt) would be overkill here.
    sentences = re.split(r"[.!?]+(?:\s|$)", raw_text.strip())
    sentence_count = len([s for s in sentences if s.strip()])

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
    }
