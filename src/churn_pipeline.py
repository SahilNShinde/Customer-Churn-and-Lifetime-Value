from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


@dataclass(frozen=True)
class PipelineConfig:
    churn_recency_days: int = 90
    test_size: float = 0.2
    random_state: int = 42
    rf_n_estimators: int = 200
    rf_max_depth: int | None = 10


REQUIRED_RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def load_raw_transactions(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")
    return df


def clean_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Parse time
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Normalize numeric columns
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price", "Customer ID"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Remove cancellations (Invoice values starting with "C" exist in this dataset)
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C", na=False)]

    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df


def build_customer_features(df_txn: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df_txn.copy()

    # Snapshot date for recency (end of dataset)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    grp = df.groupby("Customer ID")

    rfm = grp.agg(
        FirstPurchaseDate=("InvoiceDate", "min"),
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("Invoice", pd.Series.nunique),
        Monetary=("TotalPrice", "sum"),
        Variety=("StockCode", pd.Series.nunique),
    ).reset_index()

    rfm["Tenure"] = (rfm["LastPurchaseDate"] - rfm["FirstPurchaseDate"]).dt.days.clip(lower=0)
    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"]).dt.days.clip(lower=0)

    # Avoid divide-by-zero; Frequency is >=1 for grouped customers, but keep robust.
    rfm["AvgTicketSize"] = rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)

    # Purchase velocity (orders per day of tenure). If tenure=0, use 1 day to avoid inf.
    denom_days = rfm["Tenure"].replace(0, 1)
    rfm["PurchaseVelocity"] = rfm["Frequency"] / denom_days

    # Churn label from project convention
    rfm["Churn"] = (rfm["Recency"] > cfg.churn_recency_days).astype(int)

    return rfm[
        [
            "Customer ID",
            "FirstPurchaseDate",
            "Frequency",
            "Monetary",
            "Tenure",
            "Variety",
            "PurchaseVelocity",
            "AvgTicketSize",
            "Recency",
            "Churn",
        ]
    ]


FEATURE_COLUMNS = ["Frequency", "Monetary", "AvgTicketSize", "Tenure", "Variety", "PurchaseVelocity"]


def train_churn_model(df_features: pd.DataFrame, cfg: PipelineConfig) -> tuple[RandomForestClassifier, str]:
    # Keep the model leakage-resistant by excluding Recency (label definition depends on it).
    X = df_features[FEATURE_COLUMNS].copy()
    y = df_features["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    return model, report


def predict_with_buckets(
    model: RandomForestClassifier,
    df_features: pd.DataFrame,
    *,
    churn_threshold: float = 0.5,
    value_threshold_quantile: float = 0.5,
) -> pd.DataFrame:
    X = df_features[FEATURE_COLUMNS].copy()
    churn_proba = model.predict_proba(X)[:, 1]

    out = df_features.copy()
    out["Churn_Probability"] = churn_proba
    out["Churn_Pred"] = (out["Churn_Probability"] >= churn_threshold).astype(int)

    # Value: high/low by Monetary quantile (default: median split)
    value_threshold = float(out["Monetary"].quantile(value_threshold_quantile))
    out["Value_Level"] = np.where(out["Monetary"] >= value_threshold, "High Value", "Low Value")

    # Risk: high/low by churn probability threshold
    out["Risk_Level"] = np.where(out["Churn_Probability"] >= churn_threshold, "High Risk", "Low Risk")

    def bucket(v: str, r: str) -> str:
        if v == "High Value" and r == "High Risk":
            return "High Value High Risk"
        if v == "High Value" and r == "Low Risk":
            return "High Value Low Risk"
        if v == "Low Value" and r == "High Risk":
            return "Low Value High Risk"
        return "Low Value Low Risk"

    out["Value_Risk_Bucket"] = [bucket(v, r) for v, r in zip(out["Value_Level"], out["Risk_Level"])]

    return out


def save_model(model: RandomForestClassifier, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)


def load_model(path: str | Path) -> RandomForestClassifier:
    return load(path)

