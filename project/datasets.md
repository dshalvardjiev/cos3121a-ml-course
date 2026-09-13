# Curated Phase 1 dataset list (pick one, declare by end of Session 1)
All public, real-world, business/finance-flavored. `scripts/download_data.py`
fetches all five (falling back to a schema-identical sample whenever a source is unreachable).

| Key | Dataset | Task | Source & license notes |
|---|---|---|---|
| `telco_churn` | Telco Customer Churn (7,000 rows committed) | churn classification | Kaggle `blastchar/telco-customer-churn` — used in the demos; **pick a different one for your project** unless you extend it meaningfully |
| `bank_marketing` | Bank Marketing (4,500 rows committed; ~45K in the full UCI set) | term-deposit subscription classification | UCI #222 (CC BY 4.0) |
| `credit_default` | Default of Credit Card Clients (6,000 rows committed; ~30K full) | default classification | UCI #350 (CC BY 4.0) |
| `online_retail` | Online Retail II (25,000 rows committed; ~500K full) | build features → e.g., high-value-customer classification or basket regression | UCI #502 (CC BY 4.0) |
| `cc_fraud` | Credit Card Fraud (not committed — run `download_data.py`; 20,000-row fallback keeps the 0.17% positive rate, ~285K in the full set) | fraud classification (extreme imbalance — ambitious) | Kaggle `mlg-ulb/creditcardfraud` (DbCL) |
| bring-your-own | Any public tabular business dataset ≥5K rows | needs instructor approval by e-mail before Session 2 | — |


> **A note on sizes.** The CSVs committed to this repository are compact, schema-identical
> samples, so that cloning is fast and no exercise ever waits on a download. Everything in
> the course works on them. `python scripts/download_data.py` replaces any of them with the
> full public dataset when it can reach the source; where it cannot, it regenerates the
> sample. If your report quotes a row count, quote the one you actually trained on.

Guidance: `bank_marketing` and `credit_default` are the smoothest Phase 1 rides.
`cc_fraud` earns respect but budget extra time for the imbalance (precision/recall
trade-offs, resampling). Whatever you pick, Session 2's demo notebook generalizes:
`project/phase1_starter.ipynb` is that notebook with blanks where your dataset goes.
