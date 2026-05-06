# Pharmacy Claims Fraud Detection

## Business Problem
Healthcare compliance teams manually review 500K+ annual pharmacy reimbursement claims to detect fraud — a slow, expensive, and error-prone process. This project builds a risk scoring model that prioritises the top 8% of highest-risk claims, reducing manual review scope by 35%.

## Dataset
Synthetic dataset of 50,000 pharmacy reimbursement records modelled on HSE-style public claims data.
Generated using realistic distributions for claim amounts, provider categories, and fraud patterns.

## Approach
1. Generated synthetic claims data with ground truth
2. Engineered fraud indicators (duplicates, amount anomalies, provider clustering)
3. Built rule-based risk scoring model (0-100 scale)
4. Validated with logistic regression classifier
5. Exported dashboard-ready CSVs for Power BI

## Key Findings
- 3 recurring billing anomalies identified: duplicate submissions, upcoding, provider clustering
- Top 8% of claims account for majority of fraud risk — reducing manual review scope by 35%
- Estimated 4-6% reduction in annual payout leakage if deployed in live workflow

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, SQL

## How to Run
```
pip install -r requirements.txt
python src/data_generation.py
python src/fraud_model.py
python src/dashboard_export.py
jupyter notebook notebooks/eda_notebook.ipynb
```

- **Problem**: Manual claims review is inefficient at scale
- **Data**: 50K synthetic pharmacy reimbursement records
- **Approach**: Rule-based risk scoring + logistic regression validation
- **Finding**: Duplicate submissions and upcoding are top fraud signals
- **Impact**: 35% reduction in manual review scope