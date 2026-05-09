"""Evaluate the saved fraud model on the holdout split and emit PNG artefacts.

Loads outputs/fraud_model.pkl and data/processed/scored_claims.csv, recreates the
exact same 80/20 stratified split used in fraud_model.py (random_state=42), then
saves confusion_matrix.png, roc_curve.png, and classification_report.csv to
outputs/charts/.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split


def load_artefacts(model_path: Path, data_path: Path):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    df = pd.read_csv(data_path)
    return bundle, df


def evaluate(bundle: dict, df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    features = bundle["features"]
    model    = bundle["model"]
    scaler   = bundle["scaler"]

    X = df[features].fillna(0)
    y = df["is_fraud_ground_truth"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_s = scaler.transform(X_test)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall    = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1        = metrics.f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = metrics.roc_auc_score(y_test, y_proba)

    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legit", "Actual Fraud"],
        columns=["Predicted Legit", "Predicted Fraud"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#F8F9FA")
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=True, linewidths=0.5, linecolor="white")
    ax.set_title("Confusion Matrix — Fraud Classifier (holdout)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    ax.plot(fpr, tpr, color="#2C7BB6", linewidth=2.5,
            label=f"ROC AUC = {roc_auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2C7BB6")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.5,
            label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Fraud Classifier (holdout)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = metrics.classification_report(
        y_test, y_pred, zero_division=0, output_dict=True,
        target_names=["Legit", "Fraud"],
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    return {
        "precision_fraud": float(precision),
        "recall_fraud":    float(recall),
        "f1_fraud":        float(f1),
        "roc_auc":         float(roc_auc),
        "n_test":          int(len(y_test)),
        "n_fraud_test":    int(y_test.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pharmacy fraud risk model.")
    parser.add_argument("--model-file",  type=Path, default=Path("outputs/fraud_model.pkl"))
    parser.add_argument("--data-file",   type=Path, default=Path("data/processed/scored_claims.csv"))
    parser.add_argument("--output-dir",  type=Path, default=Path("outputs/charts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle, df = load_artefacts(args.model_file, args.data_file)
    summary = evaluate(bundle, df, args.output_dir)
    print("Holdout evaluation summary:")
    for name, value in summary.items():
        if isinstance(value, float):
            print(f"  {name:<18} {value:.4f}")
        else:
            print(f"  {name:<18} {value}")
    print(f"\nArtefacts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
