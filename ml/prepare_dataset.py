"""
Builds the DocuMind training set from the raw public dataset.

Input : ml/dataset/UpdatedResumeDataSet.csv
        ("Resume Dataset", Kaggle, 962 resumes, 25 original job
        categories, columns: Category, Resume. NOT committed to git —
        download it yourself, see ml/dataset/README.md.)
Output: ml/dataset/documind_dataset.csv (Category, Resume), containing
        only the rows whose original category maps to one of
        DocuMind's 9 supported categories.

Full source, license and category-mapping rationale:
docs/dataset_strategy.md
"""
import sys
from pathlib import Path

import pandas as pd

# original label (source dataset) -> DocuMind target category
CATEGORY_MAP = {
    "Data Science": "Data Science",
    "Hadoop": "Data Engineering",
    "ETL Developer": "Data Engineering",
    "Java Developer": "Software Engineering",
    "Python Developer": "Software Engineering",
    "DotNet Developer": "Software Engineering",
    "Web Designing": "Web Development",
    "Network Security Engineer": "Cybersecurity",
    "Database": "Database Administration",
    "DevOps Engineer": "DevOps / Cloud Computing",
    "Business Analyst": "Business Analysis",
    "HR": "Human Resources",
}

MIN_RESUME_LENGTH = 20  # characters; drops empty/near-empty rows


def load_and_map(raw_csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV, keep only mapped categories, remap labels,
    drop empty/too-short resumes. Raises clear errors on malformed input
    rather than silently producing an empty/wrong dataset."""
    df = pd.read_csv(raw_csv_path)

    missing_cols = {"Category", "Resume"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Expected columns 'Category' and 'Resume' in {raw_csv_path}, "
            f"missing: {missing_cols}. Got columns: {list(df.columns)}"
        )

    df = df[df["Category"].isin(CATEGORY_MAP.keys())].copy()
    df["Category"] = df["Category"].map(CATEGORY_MAP)

    df["Resume"] = df["Resume"].astype(str).str.strip()
    df = df[df["Resume"].str.len() >= MIN_RESUME_LENGTH]
    df = df.drop_duplicates(subset=["Resume"])

    return df[["Category", "Resume"]].reset_index(drop=True)


def main():
    ml_dir = Path(__file__).resolve().parent
    raw_path = ml_dir / "dataset" / "UpdatedResumeDataSet.csv"
    out_path = ml_dir / "dataset" / "documind_dataset.csv"

    if not raw_path.exists():
        print(
            f"ERROR: {raw_path} not found.\n\n"
            "Download the source CSV from Kaggle "
            "(https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset,"
            " requires a free Kaggle account) and place it at:\n"
            f"  {raw_path}\n\n"
            "See ml/dataset/README.md and docs/dataset_strategy.md for details.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_and_map(raw_path)

    if df.empty:
        print(
            "ERROR: 0 rows survived the category mapping. The source CSV's "
            "'Category' column values don't match CATEGORY_MAP's expected "
            "labels — check for a schema/label change in the source file.",
            file=sys.stderr,
        )
        sys.exit(1)

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows across {df['Category'].nunique()} categories to {out_path}")
    print(df["Category"].value_counts())


if __name__ == "__main__":
    main()
