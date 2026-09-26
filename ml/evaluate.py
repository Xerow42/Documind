"""
Regenerates a human-readable evaluation report + confusion matrix plot
from ml/model/metrics.json (produced by train.py).

Usage: python ml/evaluate.py
Outputs:
    docs/confusion_matrix.png
    docs/evaluation_report.md
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe (no display needed)
import matplotlib.pyplot as plt
import numpy as np

ML_DIR = Path(__file__).resolve().parent
DOCS_DIR = ML_DIR.parent / "docs"


def load_metrics():
    metrics_path = ML_DIR / "model" / "metrics.json"
    if not metrics_path.exists():
        print(
            f"ERROR: {metrics_path} not found. Run `python ml/train.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(metrics_path) as f:
        return json.load(f)


def render_comparison_table(results: dict) -> str:
    header = "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]
    for name, m in results.items():
        rows.append(
            f"| {name} | {m['accuracy']} | {m['precision_macro']} | "
            f"{m['recall_macro']} | {m['f1_macro']} |"
        )
    return "\n".join(rows)


def main():
    meta = load_metrics()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# DocuMind — Model Evaluation Report",
        "",
        f"Selected model: **{meta['best_model_name']}** "
        f"(selection criterion: {meta['selection_criterion']})",
        "",
        f"Train samples: {meta['n_train']}  |  Test samples: {meta['n_test']}",
        "",
        "## Model comparison",
        "",
        render_comparison_table(meta["results"]),
        "",
        "## Notes on the metrics",
        "",
        "- Reported on a single held-out test split (`train_test_split`, "
        "stratified by category), not cross-validated — a reasonable "
        "baseline for a project this size, documented as a possible "
        "future improvement (k-fold CV) rather than presented as more "
        "rigorous than it is.",
        "- **Macro** precision/recall/F1 (not micro/accuracy alone) are "
        "the primary comparison metric because the dataset is class-"
        "imbalanced (see docs/dataset_strategy.md) — macro-averaging "
        "weights every category equally instead of letting the largest "
        "class dominate the score.",
    ]

    report_lines.append("")
    report_lines.append("## Classification report (selected model)")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(meta["classification_report"])
    report_lines.append("```")

    with open(DOCS_DIR / "evaluation_report.md", "w") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {DOCS_DIR / 'evaluation_report.md'}")

    cm = np.array(meta["confusion_matrix"])
    labels = meta["labels"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {meta['best_model_name']}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {DOCS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
