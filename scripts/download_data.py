#!/usr/bin/env python3
"""Fetch course datasets into data/raw/. Every dataset has a synthetic fallback,
so this script ALWAYS succeeds - offline, in class, or on a restricted network.

Real sources (see project/datasets.md for licenses):
  telco_churn      Kaggle: blastchar/telco-customer-churn
  bank_marketing   UCI 222: Bank Marketing
  online_retail    UCI 502: Online Retail II
  credit_default   UCI 350: Default of Credit Card Clients
  cc_fraud         Kaggle: mlg-ulb/creditcardfraud (large; synthetic fallback keeps the 0.17% imbalance)
"""
from pathlib import Path
import io, sys, zipfile
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)


def _done(name, df, note=""):
    df.to_csv(RAW / f"{name}.csv", index=False)
    print(f"  ok {name}.csv  ({len(df):,} rows) {note}")


def _kaggle(dataset, filename):
    import kagglehub
    path = Path(kagglehub.dataset_download(dataset))
    hits = list(path.rglob(filename))
    return pd.read_csv(hits[0])


def _uci_zip(url, inner_reader):
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return inner_reader(zipfile.ZipFile(io.BytesIO(r.content)))


# ---------------- synthetic fallbacks (same schema as the real data) ----------------
def synth_telco(n=7000):
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    churn_p = 0.42 - 0.004 * tenure + (contracts == "Month-to-month") * 0.18
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    churn_p += senior * 0.10                      # seeded imbalance for the S5 Clarify demo
    churn = rng.random(n) < np.clip(churn_p, 0.03, 0.9)
    total = np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.9, 1.1, n), 2).astype(str)
    total[rng.choice(n, 11, replace=False)] = " "  # the famous 11 blank TotalCharges
    return pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": senior,
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": contracts,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })


def synth_bank(n=4500):
    return pd.DataFrame({
        "age": rng.integers(18, 88, n),
        "job": rng.choice(["admin.", "technician", "blue-collar", "management",
                           "retired", "services", "student", "unknown"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
        "balance": rng.normal(1400, 2800, n).round(0).astype(int),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n, p=[0.16, 0.84]),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n),
        "month": rng.choice(["jan", "feb", "mar", "apr", "may", "jun",
                             "jul", "aug", "sep", "oct", "nov", "dec"], n),
        "duration": rng.integers(0, 3000, n),
        "campaign": rng.integers(1, 15, n),
        "y": rng.choice(["yes", "no"], n, p=[0.12, 0.88]),
    })


def synth_retail(n=25000, customers=1200):
    start = pd.Timestamp("2026-01-01")
    cust = rng.integers(10000, 10000 + customers, n)
    return pd.DataFrame({
        "Invoice": [f"5{i//3:05d}" for i in range(n)],
        "StockCode": rng.integers(10000, 99999, n).astype(str),
        "Description": rng.choice(["MUG", "LAMP", "TOTE BAG", "CANDLE", "NOTEBOOK", "CLOCK"], n),
        "Quantity": rng.integers(1, 24, n),
        "InvoiceDate": start + pd.to_timedelta(rng.integers(0, 180, n), unit="D"),
        "Price": np.round(rng.uniform(0.5, 30, n), 2),
        "CustomerID": cust.astype(float),
        "Country": rng.choice(["United Kingdom", "Germany", "France", "Bulgaria"], n, p=[0.7, 0.1, 0.1, 0.1]),
    })


def synth_credit_default(n=6000):
    return pd.DataFrame({
        "LIMIT_BAL": rng.choice(range(10000, 500001, 10000), n),
        "SEX": rng.choice([1, 2], n), "EDUCATION": rng.choice([1, 2, 3, 4], n),
        "MARRIAGE": rng.choice([1, 2, 3], n), "AGE": rng.integers(21, 70, n),
        "PAY_0": rng.integers(-2, 9, n), "BILL_AMT1": rng.normal(50000, 60000, n).round(0),
        "PAY_AMT1": rng.normal(5000, 8000, n).clip(0).round(0),
        "default_payment_next_month": rng.choice([0, 1], n, p=[0.78, 0.22]),
    })


def synth_cc_fraud(n=20000, positive_rate=0.0017):
    """Same shape as the real Kaggle set: anonymised PCA components V1..V28 plus
    Time, Amount and Class. The point of this dataset is the imbalance, so that is
    what the fallback reproduces faithfully - roughly 0.17% positives."""
    n_pos = max(2, int(round(n * positive_rate)))
    cols = {f"V{i}": rng.normal(0, 1, n).round(4) for i in range(1, 29)}
    cols["Time"] = np.sort(rng.integers(0, 172800, n))
    cols["Amount"] = np.round(rng.lognormal(3.0, 1.2, n), 2)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    # give the fraud rows a faint, learnable signature - otherwise no model can
    # beat the majority class and the exercise teaches nothing
    for c in ("V3", "V10", "V14"):
        cols[c] = cols[c] + y * rng.normal(-2.2, 0.6, n).round(4)
    cols["Class"] = y
    return pd.DataFrame(cols)[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]]


JOBS = [
    ("telco_churn", lambda: _kaggle("blastchar/telco-customer-churn",
                                    "WA_Fn-UseC_-Telco-Customer-Churn.csv"), synth_telco),
    ("bank_marketing", lambda: _uci_zip(
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        lambda z: pd.read_csv(io.BytesIO(zipfile.ZipFile(io.BytesIO(z.read("bank.zip")))
                                         .read("bank-full.csv")), sep=";")), synth_bank),
    ("online_retail", lambda: _uci_zip(
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        lambda z: pd.read_excel(io.BytesIO(z.read(z.namelist()[0])),
                                sheet_name="Year 2010-2011")), synth_retail),
    ("credit_default", lambda: _uci_zip(
        "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        lambda z: pd.read_excel(io.BytesIO(z.read(z.namelist()[0])), header=1)), synth_credit_default),
    ("cc_fraud", lambda: _kaggle("mlg-ulb/creditcardfraud", "creditcard.csv"), synth_cc_fraud),
]


def main(synthetic_only=False):
    for name, real, synth in JOBS:
        if (RAW / f"{name}.csv").exists():
            print(f"  -- {name}.csv already present, skipping")
            continue
        if not synthetic_only:
            try:
                _done(name, real(), "(real source)")
                continue
            except Exception as e:
                print(f"  !! {name}: real download failed ({type(e).__name__}) -> synthetic fallback")
        _done(name, synth(), "(synthetic)")
    print("done. Real sources are tried first; anything unreachable falls back to synthetic\n"
          "      data with the same schema, so this script always succeeds.")


if __name__ == "__main__":
    main(synthetic_only="--synthetic" in sys.argv)
