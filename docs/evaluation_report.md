# DocuMind — Model Evaluation Report

Selected model: **logistic_regression** (selection criterion: highest macro-F1 on the held-out test split)

Train samples: 72  |  Test samples: 19

## Model comparison

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| logistic_regression | 0.7895 | 0.6111 | 0.6667 | 0.6158 |
| linear_svm | 0.7895 | 0.6111 | 0.6667 | 0.6158 |
| multinomial_nb | 0.3158 | 0.142 | 0.1667 | 0.1224 |

## Notes on the metrics

- Reported on a single held-out test split (`train_test_split`, stratified by category), not cross-validated — a reasonable baseline for a project this size, documented as a possible future improvement (k-fold CV) rather than presented as more rigorous than it is.
- **Macro** precision/recall/F1 (not micro/accuracy alone) are the primary comparison metric because the dataset is class-imbalanced (see docs/dataset_strategy.md) — macro-averaging weights every category equally instead of letting the largest class dominate the score.

## Classification report (selected model)

```
                          precision    recall  f1-score   support

       Business Analysis       0.50      1.00      0.67         1
           Cybersecurity       0.00      0.00      0.00         1
        Data Engineering       1.00      1.00      1.00         3
            Data Science       0.67      1.00      0.80         2
 Database Administration       0.50      0.50      0.50         2
DevOps / Cloud Computing       1.00      0.50      0.67         2
         Human Resources       1.00      1.00      1.00         2
    Software Engineering       0.83      1.00      0.91         5
         Web Development       0.00      0.00      0.00         1

                accuracy                           0.79        19
               macro avg       0.61      0.67      0.62        19
            weighted avg       0.74      0.79      0.74        19

```