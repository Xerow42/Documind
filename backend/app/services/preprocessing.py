"""
Re-exports the shared preprocessing functions from ml/preprocessing.py
so training and inference use the exact same cleaning logic — a single
source of truth instead of two copies that could silently drift apart.

Loaded via importlib (rather than a sys.path insert + plain import) to
avoid a same-module-name collision with this file itself.
"""
import importlib.util
from pathlib import Path

_ML_PREPROCESSING_PATH = Path(__file__).resolve().parents[3] / "ml" / "preprocessing.py"

_spec = importlib.util.spec_from_file_location("ml_preprocessing", _ML_PREPROCESSING_PATH)
_ml_preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml_preprocessing)

clean_text = _ml_preprocessing.clean_text
remove_stopwords = _ml_preprocessing.remove_stopwords
preprocess = _ml_preprocessing.preprocess
compute_stats = _ml_preprocessing.compute_stats
