"""Evaluate the fraud scoring model with classification metrics and plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

sns.set_theme(style="whitegrid", palette=["#2C7BB6", "#92C5DE", "#D7191C"])


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load model predictions and ground-truth labels."""
    return pd.read_csv(predictions_path)


def evaluate(predictions: pd.DataFrame, output_dir: Path) -> dict:
    """Compute metrics and save evaluation visuals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions["is_flagged"]
    y_pred = predictions["flagged_high_risk"]
    y_score = predictions["risk_score"] / 100

    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    save_path = output_dir / "confusion_matrix.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2C7BB6", label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = metrics.classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pharmacy fraud risk model.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("../outputs/flagged_claims.csv"),
        help="Path to flagged claims output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/charts"),
        help="Directory to save evaluation charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.input_file)
    metrics_summary = evaluate(predictions, args.output_dir)
    print("Evaluation metrics:")
    for name, value in metrics_summary.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
