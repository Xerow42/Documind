# DocuMind — Resume Classification Platform

Full-stack application that classifies resumes into **24 professional sectors** using a TF-IDF + classical machine learning pipeline, extracts keywords, computes document statistics, and keeps a searchable history of every analysis.

**Stack:** Python, scikit-learn (TF-IDF, Logistic Regression, Linear SVM, Multinomial Naive Bayes), FastAPI, SQLite, React.

## Results

Evaluated on the Kaggle *Resume Dataset* (2,482 unique resumes, 24 categories), stratified 80/20 split (1,985 train / 497 test, `random_state=42`).

| Model | Macro-F1, 5-fold CV on train (mean ± std) |
|---|---|
| **Linear SVM** (selected) | **0.655 ± 0.016** |
| Logistic Regression | 0.581 ± 0.017 |
| Multinomial Naive Bayes | 0.456 ± 0.016 |

Model selection used cross-validation on the training split only. The test split was evaluated **once**, after selection:

| Metric (test, 497 resumes) | Value |
|---|---|
| Accuracy | 0.726 |
| Macro precision | 0.703 |
| Macro recall | 0.691 |
| Macro F1 | 0.685 |

Best categories: Information Technology (F1 0.91), Designer (0.91), HR (0.88), Aviation (0.87), Banking (0.82).
Weakest: BPO (0.00, 4 test samples), Apparel (0.36), Arts (0.51), Consultant (0.51).

Full per-category table: [`docs/evaluation_report.md`](docs/evaluation_report.md). Confusion matrix: `docs/confusion_matrix.png`.

### Robustness check: job title in the resume header

Many resumes in this dataset start with a job title close to the category name. A cross-validation ablation on the training split gives macro-F1 **0.655 ± 0.016 with** the title and **0.614 ± 0.011 without** it. The title removal was partial (it relied on the word "Summary", missing for about 24% of resumes), so 0.614 is an upper bound for the title-free score. The model therefore relies on resume content as well as the title.

## Features

- **Resume upload** — PDF and TXT, with extension, size and empty-file validation and safe error messages.
- **Text extraction and preprocessing** — `pdfplumber` for PDF, UTF-8/Latin-1 for TXT; one shared cleaning pipeline (lowercase, strip URLs/emails/punctuation, stopword removal) used identically at training and inference.
- **Classification into 24 sectors** with a confidence score.
- **Keyword extraction** — reuses the fitted TF-IDF vectorizer's per-document term weights.
- **Classification history** — SQLite-backed (`documents`, `analyses`, `classification_results`, `keywords`), with History and Document Detail pages.
- **React frontend** — Dashboard, Upload, Analysis Result, History, Document Detail.
- **Validation, logging, basic security** — checks before parsing, structured JSON errors, no secrets in code (`.env.example`), CORS restricted to the configured frontend origin.

## Architecture

Training happens offline (`ml/`); the API loads `model.pkl` and `vectorizer.pkl` at startup and never retrains on the fly.

```
documind/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, CORS, startup (loads model + DB)
│   │   ├── routers/documents.py     Upload / analyze / list / detail / delete
│   │   ├── services/                text extraction, preprocessing, classification, keywords
│   │   ├── db/                      sqlite3 connection, schema, repository
│   │   ├── models/schemas.py        Pydantic models
│   │   └── core/config.py           Env-based settings
│   ├── tests/
│   └── requirements.txt
├── ml/
│   ├── preprocessing.py             Shared cleaning pipeline
│   ├── prepare_dataset.py           Loads Resume.csv, removes duplicates
│   ├── train_cv.py                  CV model selection, single test evaluation
│   ├── evaluate.py                  Generates report + confusion matrix
│   ├── ablation_title.py            Job-title ablation
│   └── model/                       model.pkl, vectorizer.pkl, metrics.json
├── frontend/src/                    pages, components, api/client.js
├── docs/                            evaluation_report.md, confusion_matrix.png
└── README.md
```

### Technology choices

| Decision | Reasoning |
|---|---|
| FastAPI | Automatic request validation (Pydantic), OpenAPI docs, async support. |
| Raw `sqlite3` | Four small tables; plain SQL is simpler to explain than an ORM. |
| TF-IDF + classical ML | Fast, no GPU, every prediction traceable to weighted terms. |
| Macro-F1 for selection | Classes are imbalanced (22 to 120 resumes); macro-F1 weights every sector equally. |
| Cross-validation for selection | Keeps the test set untouched until the final evaluation. |

## Dataset

Kaggle [*Resume Dataset*](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (snehaanbhawal): resumes scraped from livecareer.com. 2,484 rows, 2,482 unique texts, 24 categories (22 to 120 resumes each). The CSV is not committed; download `Resume.csv` and place it in `ml/dataset/`.

## Reproduce the results

```bash
pip install -r ml/requirements.txt
python ml/prepare_dataset.py     # dedupe -> ml/dataset/documind_dataset.csv
python ml/train_cv.py            # CV selection + single test evaluation
python ml/evaluate.py            # docs/evaluation_report.md + confusion matrix
python ml/ablation_title.py      # optional: job-title ablation
```

## Run locally

```bash
# Backend
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env       # set MODEL_DIR=../ml/model
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm start                        # http://localhost:3000
```

Without a trained model, `/analyze` returns a 503 with a clear message.

## Known limitations

- **Labels are noisy.** Categories come from the search category on the source site, not from the resume content (for example, an accountant resume labeled AUTOMOBILE). This caps achievable accuracy.
- **Sectors, not IT specialties.** The 24 categories are broad professional sectors; Information Technology is a single class.
- **Small and imbalanced classes.** BPO (22 resumes) and Automobile (36) are poorly learned; BPO has F1 0.00 on the test split.
- **Confusable neighbours.** Accounting, finance and banking resumes overlap in vocabulary.
- **Confidence scores.** Logistic Regression and Naive Bayes expose probabilities; Linear SVM does not. For the selected SVM the UI shows an uncalibrated score derived from the decision margins, not a calibrated probability.
- **Single split, single seed.** Test metrics come from one stratified split (`random_state=42`).
- **No Docker, CI, authentication or HTTPS yet.** SQLite is suited to a demo, not concurrent production traffic.

## Possible improvements

Calibrated probabilities (`CalibratedClassifierCV`), hyperparameter search inside cross-validation, merging overlapping sectors, GitHub Actions CI, Docker, Postgres.

## Author

Khalil Lamrabet — Engineering student, Big Data & Artificial Intelligence, Junia ISEN.
GitHub: [@Xerow42](https://github.com/Xerow42) · LinkedIn: [khalillam12](https://linkedin.com/in/khalillam12)
