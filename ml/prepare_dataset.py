from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "dataset" / "Resume.csv"
OUT = ROOT / "dataset" / "documind_dataset.csv"

df = pd.read_csv(RAW)[["Resume_str", "Category"]].dropna()
df = df.rename(columns={"Resume_str": "text", "Category": "label"})

n0 = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
print(f"{n0} lignes -> {len(df)} après suppression des doublons exacts")
print(df["label"].value_counts())

df.to_csv(OUT, index=False)
print(f"Écrit : {OUT}")