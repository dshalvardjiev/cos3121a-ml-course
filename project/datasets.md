# Curated Phase 1 dataset list (pick one, declare by end of Session 1)
All public, real-world, business/finance-flavored. `scripts/download_data.py`
fetches the first four automatically (with synthetic fallback for offline work).

| Key | Dataset | Task | Source & license notes |
|---|---|---|---|
| `telco_churn` | Telco Customer Churn (~7K rows) | churn classification | Kaggle `blastchar/telco-customer-churn` — used in the demos; **pick a different one for your project** unless you extend it meaningfully |
| `bank_marketing` | Bank Marketing (~45K) | term-deposit subscription classification | UCI #222 (CC BY 4.0) |
| `credit_default` | Default of Credit Card Clients (~30K) | default classification | UCI #350 (CC BY 4.0) |
| `online_retail` | Online Retail II (~500K transactions) | build features → e.g., high-value-customer classification or basket regression | UCI #502 (CC BY 4.0) |
| `cc_fraud` | Credit Card Fraud (~285K, 0.17% positives) | fraud classification (extreme imbalance — ambitious) | Kaggle `mlg-ulb/creditcardfraud` (DbCL) |
| bring-your-own | Any public tabular business dataset ≥5K rows | needs instructor approval by e-mail before Session 2 | — |

Guidance: `bank_marketing` and `credit_default` are the smoothest Phase 1 rides.
`cc_fraud` earns respect but budget extra time for the imbalance (precision/recall
trade-offs, resampling). Whatever you pick, Session 2's demo notebook generalizes:
`project/phase1_starter.ipynb` is that notebook with blanks where your dataset goes.
