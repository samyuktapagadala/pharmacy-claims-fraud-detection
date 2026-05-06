"""Export summary datasets for dashboarding and business reporting."""

import argparse
from pathlib import Path

import pandas as pd


def export_summaries(input_path: Path, export_dir: Path) -> None:
    """Export aggregated CSV files for Power BI or Tableau."""
    df = pd.read_csv(input_path)
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["month"] = df["claim_date"].dt.to_period("M").dt.to_timestamp()

    monthly_claim_volumes = (
        df.groupby("month")
        .agg(
            total_claims=("claim_id", "count"),
            total_claim_amount=("claim_amount", "sum"),
            flagged_rate=("flagged_high_risk", "mean"),
        )
        .reset_index()
    )

    provider_risk_scores = (
        df.groupby(["provider_id", "provider_category"])
        .agg(
            provider_claims=("claim_id", "count"),
            avg_risk_score=("risk_score", "mean"),
            high_risk_share=("flagged_high_risk", "mean"),
        )
        .reset_index()
    )

    anomaly_rates_by_category = (
        df.assign(anomaly=(df["upcoding_score"] > 0.25).astype(int))
        .groupby("provider_category")
        .agg(
            claim_count=("claim_id", "count"),
            anomaly_rate=("anomaly", "mean"),
            average_claim_amount=("claim_amount", "mean"),
        )
        .reset_index()
    )

    export_dir.mkdir(parents=True, exist_ok=True)
    monthly_claim_volumes.to_csv(export_dir / "monthly_claim_volumes.csv", index=False)
    provider_risk_scores.to_csv(export_dir / "provider_risk_scores.csv", index=False)
    anomaly_rates_by_category.to_csv(export_dir / "anomaly_rates_by_category.csv", index=False)
    print(f"Dashboard exports saved to {export_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dashboard-ready data for pharmacy claims fraud project.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("../outputs/flagged_claims.csv"),
        help="Path to flagged claims dataset with risk scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/dashboard_exports"),
        help="Directory for exported dashboard CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_summaries(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
