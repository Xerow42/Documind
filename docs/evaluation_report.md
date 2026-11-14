# DocuMind — Model Evaluation Report

Dataset: 2482 unique resumes, 24 categories (train 1985 / test 497).

Selected model: **linear_svm**, chosen by 5-fold stratified cross-validation (macro-F1) on the training split only. The test split was evaluated once, after selection.

## Cross-validation (training split)

| Model | Macro-F1 (mean ± std) |
|---|---|
| logistic_regression | 0.581 ± 0.017 |
| linear_svm | 0.655 ± 0.016 |
| multinomial_nb | 0.456 ± 0.016 |

## Test set (single evaluation)

- Accuracy: 0.726
- Macro precision: 0.703
- Macro recall: 0.691
- Macro F1: 0.685

## Per-category results

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ACCOUNTANT | 0.63 | 0.92 | 0.75 | 24 |
| ADVOCATE | 0.70 | 0.67 | 0.68 | 24 |
| AGRICULTURE | 0.80 | 0.62 | 0.70 | 13 |
| APPAREL | 0.56 | 0.26 | 0.36 | 19 |
| ARTS | 0.64 | 0.43 | 0.51 | 21 |
| AUTOMOBILE | 0.67 | 0.57 | 0.62 | 7 |
| AVIATION | 0.87 | 0.87 | 0.87 | 23 |
| BANKING | 0.86 | 0.78 | 0.82 | 23 |
| BPO | 0.00 | 0.00 | 0.00 | 4 |
| BUSINESS-DEVELOPMENT | 0.71 | 0.83 | 0.77 | 24 |
| CHEF | 0.89 | 0.71 | 0.79 | 24 |
| CONSTRUCTION | 0.75 | 0.82 | 0.78 | 22 |
| CONSULTANT | 0.75 | 0.39 | 0.51 | 23 |
| DESIGNER | 0.87 | 0.95 | 0.91 | 21 |
| DIGITAL-MEDIA | 0.75 | 0.63 | 0.69 | 19 |
| ENGINEERING | 0.72 | 0.75 | 0.73 | 24 |
| FINANCE | 0.83 | 0.62 | 0.71 | 24 |
| FITNESS | 0.86 | 0.79 | 0.83 | 24 |
| HEALTHCARE | 0.67 | 0.70 | 0.68 | 23 |
| HR | 0.81 | 0.95 | 0.88 | 22 |
| INFORMATION-TECHNOLOGY | 0.83 | 1.00 | 0.91 | 24 |
| PUBLIC-RELATIONS | 0.58 | 0.82 | 0.68 | 22 |
| SALES | 0.52 | 0.65 | 0.58 | 23 |
| TEACHER | 0.61 | 0.85 | 0.71 | 20 |