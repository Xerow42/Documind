"""Génère docs/evaluation_report.md et docs/confusion_matrix.png
à partir de ml/model/metrics.json (produit par train_cv.py)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ML_DIR = Path(__file__).resolve().parent
DOCS_DIR = ML_DIR.parent / "docs"


def main():
    path = ML_DIR / "model" / "metrics.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run `python ml/train_cv.py` first.", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(path.read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    rep = meta["test_report"]
    labels = meta["labels"]

    lines = [
        "# DocuMind — Model Evaluation Report", "",
        f"Dataset: {meta['n_samples']} unique resumes, {len(labels)} categories "
        f"(train {meta['n_train']} / test {meta['n_test']}).", "",
        f"Selected model: **{meta['best_model']}**, chosen by 5-fold stratified "
        "cross-validation (macro-F1) on the training split only. The test split "
        "was evaluated once, after selection.", "",
        "## Cross-validation (training split)", "",
        "| Model | Macro-F1 (mean ± std) |", "|---|---|",
    ]
    for name, s in meta["cv_macro_f1"].items():
        lines.append(f"| {name} | {s['mean']:.3f} ± {s['std']:.3f} |")

    lines += [
        "", "## Test set (single evaluation)", "",
        f"- Accuracy: {rep['accuracy']:.3f}",
        f"- Macro precision: {rep['macro avg']['precision']:.3f}",
        f"- Macro recall: {rep['macro avg']['recall']:.3f}",
        f"- Macro F1: {rep['macro avg']['f1-score']:.3f}", "",
        "## Per-category results", "",
        "| Category | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for lab in labels:
        r = rep[lab]
        lines.append(f"| {lab} | {r['precision']:.2f} | {r['recall']:.2f} | "
                     f"{r['f1-score']:.2f} | {int(r['support'])} |")

    (DOCS_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {DOCS_DIR / 'evaluation_report.md'}")

    cm = np.array(meta["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {meta['best_model']}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {DOCS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()