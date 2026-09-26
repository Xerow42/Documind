# ml/dataset/

This folder is intentionally empty in the repository (the raw CSV is
gitignored — see root `.gitignore`).

## To train the real model

1. Create a free Kaggle account if you don't have one.
2. Download "Updated Resume Dataset" (`UpdatedResumeDataSet.csv`):
   https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset
   (If that link has moved again, search Kaggle for "Updated Resume Dataset" —
   look for 962 rows, 2 columns: `category`, `resume`. Originally published
   by gauravduttakiit; several mirrors exist under the same content/shape.)
3. Place the file here as: `ml/dataset/UpdatedResumeDataSet.csv`
4. From the repo root:
   ```bash
   python ml/prepare_dataset.py   # writes documind_dataset.csv (filtered/remapped)
   python ml/train.py             # trains + compares 3 models, saves the best
   python ml/evaluate.py          # writes docs/evaluation_report.md + confusion matrix plot
   ```

Full source details, license, category-mapping rationale and known
limitations: [`docs/dataset_strategy.md`](../docs/dataset_strategy.md).
