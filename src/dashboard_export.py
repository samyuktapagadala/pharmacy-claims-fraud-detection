import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/scored_claims.csv", parse_dates=["claim_date"])
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Monthly claim volumes
df["month"] = df["claim_date"].dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(
    total_claims=("claim_id", "count"),
    flagged_claims=("high_risk", "sum"),
    total_amount=("claim_amount", "sum"),
    flagged_amount=("claim_amount", lambda x: x[df.loc[x.index, "high_risk"] == 1].sum())
).reset_index()
monthly["flag_rate"] = (monthly["flagged_claims"] / monthly["total_claims"]).round(4)
monthly["total_amount"] = monthly["total_amount"].round(2)
monthly["flagged_amount"] = monthly["flagged_amount"].round(2)
monthly = monthly[["month", "total_claims", "flagged_claims", "flag_rate", "total_amount", "flagged_amount"]]
monthly.to_csv(out_dir / "monthly_claim_volumes.csv", index=False)
print(f"monthly_claim_volumes.csv: {len(monthly)} rows")

# 2. Provider risk scores
provider_stats = df.groupby(["provider_id", "provider_category"]).agg(
    total_claims=("claim_id", "count"),
    avg_risk_score=("risk_score", "mean"),
    high_risk_count=("high_risk", "sum")
).reset_index()
provider_stats["high_risk_rate"] = (provider_stats["high_risk_count"] / provider_stats["total_claims"]).round(4)
provider_stats["avg_risk_score"] = provider_stats["avg_risk_score"].round(2)
provider_stats.to_csv(out_dir / "provider_risk_scores.csv", index=False)
print(f"provider_risk_scores.csv: {len(provider_stats)} rows")

# 3. Anomaly rates by category
category_stats = df.groupby("provider_category").agg(
    total_claims=("claim_id", "count"),
    fraud_rate=("is_fraud_ground_truth", "mean"),
    avg_claim_amount=("claim_amount", "mean"),
    high_amount_flag_rate=("high_amount_flag", "mean")
).reset_index()
category_stats["fraud_rate"] = category_stats["fraud_rate"].round(4)
category_stats["avg_claim_amount"] = category_stats["avg_claim_amount"].round(2)
category_stats["high_amount_flag_rate"] = category_stats["high_amount_flag_rate"].round(4)
category_stats.to_csv(out_dir / "anomaly_rates_by_category.csv", index=False)
print(f"anomaly_rates_by_category.csv: {len(category_stats)} rows")

print("Dashboard exports saved to outputs/")
