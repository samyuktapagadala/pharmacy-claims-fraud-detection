import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/claims.csv", parse_dates=["claim_date"])

df["duplicate_flag"] = (df["submission_count"] > 1).astype(int)

cat_stats = df.groupby("provider_category")["claim_amount"].agg(["mean", "std"]).reset_index()
cat_stats.columns = ["provider_category", "cat_mean", "cat_std"]
df = df.merge(cat_stats, on="provider_category", how="left")

df["amount_zscore"] = (df["claim_amount"] - df["cat_mean"]) / df["cat_std"].replace(0, 1)
df["high_amount_flag"] = (df["claim_amount"] > 2.5 * df["cat_mean"]).astype(int)

provider_volume = df.groupby("provider_id")["claim_id"].count().reset_index()
provider_volume.columns = ["provider_id", "provider_volume"]
df = df.merge(provider_volume, on="provider_id", how="left")

vol_mean = df["provider_volume"].mean()
vol_std = df["provider_volume"].std()
df["provider_volume_zscore"] = (df["provider_volume"] - vol_mean) / (vol_std if vol_std > 0 else 1)

dup_z = stats.zscore(df["duplicate_flag"].astype(float))
high_z = stats.zscore(df["high_amount_flag"].astype(float))
amt_z_norm = np.clip(df["amount_zscore"], -3, 10)
pvol_z_norm = np.clip(df["provider_volume_zscore"], -3, 10)

raw_score = (
    40 * df["duplicate_flag"] +
    30 * df["high_amount_flag"] +
    20 * np.clip(df["amount_zscore"], 0, 5) / 5 +
    10 * np.clip(df["provider_volume_zscore"], 0, 5) / 5
)

score_min = raw_score.min()
score_max = raw_score.max()
df["risk_score"] = ((raw_score - score_min) / (score_max - score_min) * 100).round(2)

threshold = df["risk_score"].quantile(0.92)
df["high_risk"] = (df["risk_score"] >= threshold).astype(int)

total_claims = len(df)
flagged_claims = df["high_risk"].sum()
flagged_pct = flagged_claims / total_claims * 100
review_reduction = 100 - flagged_pct

print(f"Total claims:    {total_claims:,}")
print(f"Flagged claims:  {flagged_claims:,} ({flagged_pct:.1f}%)")
print(f"Manual review scope reduced by {review_reduction:.0f}% by focusing on top 8% highest risk claims")

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "scored_claims.csv", index=False)
print(f"Saved to data/processed/scored_claims.csv")

features = ["duplicate_flag", "high_amount_flag", "amount_zscore", "provider_volume_zscore"]
X = df[features].fillna(0)
y = df["is_fraud_ground_truth"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fraud_recall = recall_score(y_test, y_pred)
print(f"Recall on fraud cases: {fraud_recall:.3f}")

outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
with open(outputs_dir / "fraud_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "features": features}, f)
print("Model saved to outputs/fraud_model.pkl")
