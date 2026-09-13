"""Data loading, cleaning, profiling, splitting. Used from Session 1 onward."""
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"


def load_raw(name: str) -> pd.DataFrame:
    """Load data/raw/<name>.csv (run scripts/download_data.py first)."""
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/download_data.py")
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Persist a cleaned/derived frame; later sessions load exactly this file."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    try:
        path = PROCESSED / f"{name}.parquet"
        df.to_parquet(path, index=False)
    except Exception:                       # pyarrow missing -> csv fallback
        path = PROCESSED / f"{name}.csv"
        df.to_csv(path, index=False)
    print(f"saved -> {path}")
    return path


def load_processed(name: str) -> pd.DataFrame:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = PROCESSED / f"{name}{ext}"
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"no processed file '{name}' - run the earlier session notebook that creates it")


def profile_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One-screen data profile: dtype, missing, unique, sample - Session 1 workhorse."""
    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
        "sample": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} cols | duplicated rows: {df.duplicated().sum()}")
    return prof


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """The Session 1 demo cleaning, packaged: TotalCharges to numeric, drop the 11 blanks, map target."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df.drop(columns=["customerID"], errors="ignore")


def prep_features(df: pd.DataFrame, target: str = "Churn"):
    """One-hot encode categoricals -> (X, y) ready for sklearn/XGBoost."""
    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    return X, y


def make_splits(X, y, seed: int = 42):
    """Stratified 70/15/15 train/val/test."""
    from sklearn.model_selection import train_test_split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return {"X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val, "X_test": X_te, "y_test": y_te}


def make_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    """Recency / Frequency / Monetary per customer from Online Retail-style transactions.
    Expects columns: CustomerID, InvoiceNo (or Invoice), InvoiceDate, Quantity, UnitPrice (or Price)."""
    t = transactions.copy()
    inv = "InvoiceNo" if "InvoiceNo" in t else "Invoice"
    price = "UnitPrice" if "UnitPrice" in t else "Price"
    t = t.dropna(subset=["CustomerID"])
    t = t[(t["Quantity"] > 0) & (t[price] > 0)]
    t["InvoiceDate"] = pd.to_datetime(t["InvoiceDate"])
    t["amount"] = t["Quantity"] * t[price]
    now = t["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = t.groupby("CustomerID").agg(
        recency_days=("InvoiceDate", lambda s: (now - s.max()).days),
        frequency=(inv, "nunique"),
        monetary=("amount", "sum"),
    ).reset_index()
    return rfm
