"""Exploratory data analysis for pharmacy claims fraud detection."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette=["#2C7BB6", "#92C5DE", "#D7191C"])


def load_data(dataset_path: Path) -> pd.DataFrame:
    """Load the processed pharmacy claims dataset."""
    return pd.read_csv(dataset_path)


def save_plot(fig: plt.Figure, chart_path: Path) -> None:
    """Save a matplotlib figure to a file."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run EDA and export charts and aggregated tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["month"] = df["claim_date"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("month").agg(
        claim_count=("claim_id", "count"),
        total_amount=("claim_amount", "sum"),
        flagged_rate=("is_flagged", "mean"),
    ).reset_index()

    provider_agg = (
        df.groupby(["provider_id", "provider_category"])
        .agg(
            claim_count=("claim_id", "count"),
            avg_claim_amount=("claim_amount", "mean"),
            flagged_rate=("is_flagged", "mean"),
        )
        .reset_index()
        .sort_values(by="claim_count", ascending=False)
    )

    corr = df[["claim_amount", "submission_count", "is_flagged"]].corr()

    # Claim amount distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df["claim_amount"], bins=50, kde=True, ax=ax, color="#2C7BB6")
    ax.set_title("Claim amount distribution")
    ax.set_xlabel("Claim amount")
    save_plot(fig, output_dir / "chart_claim_amount_distribution.png")

    # Outlier detection by provider category
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x="provider_category",
        y="claim_amount",
        data=df,
        ax=ax,
        palette="Blues",
    )
    ax.set_title("Claim amount by provider category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    save_plot(fig, output_dir / "chart_claim_amount_by_category.png")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    ax.set_title("Correlation matrix for numeric features")
    save_plot(fig, output_dir / "chart_correlation_heatmap.png")

    # Provider risk scatter
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=provider_agg,
        x="avg_claim_amount",
        y="flagged_rate",
        size="claim_count",
        hue="provider_category",
        ax=ax,
        palette="tab10",
        alpha=0.80,
    )
    ax.set_title("Provider-level flagged rate vs average claim amount")
    save_plot(fig, output_dir / "chart_provider_risk_scatter.png")

    monthly.to_csv(output_dir / "provider_monthly_summary.csv", index=False)
    provider_agg.to_csv(output_dir / "provider_aggregation_summary.csv", index=False)

    return provider_agg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for pharmacy claims fraud detection.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("../data/processed/pharmacy_claims_processed.csv"),
        help="Path to the processed dataset relative to src/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/charts"),
        help="Directory to save chart outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input_file)
    run_eda(df, args.output_dir)
    print(f"EDA complete. Charts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
