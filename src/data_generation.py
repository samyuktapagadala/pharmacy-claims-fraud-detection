import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

N = 50000
n_providers = 100
n_patients = 5000
n_drugs = 50

provider_ids = [f"PROV_{str(i).zfill(4)}" for i in range(1, n_providers + 1)]
patient_ids = [f"PAT_{str(i).zfill(5)}" for i in range(1, n_patients + 1)]
drug_codes = [f"DRUG_{str(i).zfill(3)}" for i in range(1, n_drugs + 1)]

provider_categories = [
    "GP", "Pharmacy", "Specialist", "Hospital", "Dentist",
    "Physiotherapist", "Optometrist", "Psychiatrist",
    "Cardiologist", "Dermatologist", "Oncologist", "Radiologist"
]
regions = ["Leinster", "Munster", "Connacht", "Ulster"]
provider_cat_map = {p: provider_categories[i % len(provider_categories)]
                   for i, p in enumerate(provider_ids)}

claim_ids = [f"CLM_{str(i).zfill(7)}" for i in range(1, N + 1)]
providers = np.random.choice(provider_ids, N)
patients = np.random.choice(patient_ids, N)
drugs = np.random.choice(drug_codes, N)
provider_cats = [provider_cat_map[p] for p in providers]
regions_col = np.random.choice(regions, N)

base_amounts = np.random.lognormal(mean=4.5, sigma=0.6, size=N)
base_amounts = np.clip(base_amounts, 20, 500)
outlier_mask = np.random.random(N) < 0.02
base_amounts[outlier_mask] = np.random.uniform(500, 2000, outlier_mask.sum())
claim_amounts = np.round(base_amounts, 2)

start_date = pd.Timestamp("2022-01-01")
end_date = pd.Timestamp("2023-12-31")
total_days = (end_date - start_date).days
claim_dates = pd.to_datetime([
    start_date + pd.Timedelta(days=int(d))
    for d in np.random.randint(0, total_days, N)
])

submission_counts = np.ones(N, dtype=int)
dup_mask = np.random.random(N) < 0.05
submission_counts[dup_mask] = np.random.choice([2, 3], dup_mask.sum())

df = pd.DataFrame({
    "claim_id": claim_ids,
    "provider_id": providers,
    "provider_category": provider_cats,
    "drug_code": drugs,
    "claim_amount": claim_amounts,
    "claim_date": claim_dates,
    "patient_id": patients,
    "submission_count": submission_counts,
    "region": regions_col,
})

df = df.sort_values("claim_date").reset_index(drop=True)

cat_means = df.groupby("provider_category")["claim_amount"].mean()
fraud_flags = pd.Series(False, index=df.index)
fraud_flags |= df["submission_count"] > 1

for cat, mean_val in cat_means.items():
    mask = (df["provider_category"] == cat) & (df["claim_amount"] > 3 * mean_val)
    fraud_flags |= mask

df_sorted = df.sort_values(["patient_id", "drug_code", "claim_date"]).copy()
prev_dates = df_sorted.groupby(["patient_id", "drug_code"])["claim_date"].shift(1)
days_since = (df_sorted["claim_date"] - prev_dates).dt.days
repeat_idx = df_sorted[(days_since >= 0) & (days_since <= 7)].index
fraud_flags.loc[repeat_idx] = True

current_fraud = fraud_flags.sum()
if current_fraud / N < 0.08:
    additional_needed = int(0.08 * N) - current_fraud
    non_fraud_idx = df[~fraud_flags].index
    extra_fraud = np.random.choice(non_fraud_idx, size=min(additional_needed, len(non_fraud_idx)), replace=False)
    fraud_flags.loc[extra_fraud] = True

df["is_fraud_ground_truth"] = fraud_flags.astype(int)

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "claims.csv", index=False)

fraud_count = df["is_fraud_ground_truth"].sum()
print(f"Dataset generated: {len(df):,} records")
print(f"Fraud cases: {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
print(f"Saved to data/raw/claims.csv")
