from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class RfmModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    rf_n_estimators: int = 300
    rf_max_depth: int | None = 12


RFM_FEATURE_COLUMNS = ["Frequency", "Monetary", "AvgOrderValue"]


def train_rfm_churn_model(df_rfm: pd.DataFrame, cfg: RfmModelConfig) -> tuple[RandomForestClassifier, str]:
    missing = [c for c in (RFM_FEATURE_COLUMNS + ["Churn"]) if c not in df_rfm.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    X = df_rfm[RFM_FEATURE_COLUMNS].copy()
    y = df_rfm["Churn"].astype(int)

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


def predict_rfm_with_buckets(
    model: RandomForestClassifier,
    df_rfm: pd.DataFrame,
    *,
    churn_threshold: float = 0.5,
    value_threshold_quantile: float = 0.5,
    customer_id_column: str = "Customer ID",
) -> pd.DataFrame:
    missing = [c for c in RFM_FEATURE_COLUMNS if c not in df_rfm.columns]
    if missing:
        raise ValueError(f"Missing required RFM columns for prediction: {missing}")

    X = df_rfm[RFM_FEATURE_COLUMNS].copy()
    churn_proba = model.predict_proba(X)[:, 1]

    out = df_rfm.copy()
    out["Churn_Probability"] = churn_proba
    out["Churn_Pred"] = (out["Churn_Probability"] >= churn_threshold).astype(int)

    value_threshold = float(out["Monetary"].quantile(value_threshold_quantile))
    out["Value_Level"] = np.where(out["Monetary"] >= value_threshold, "High Value", "Low Value")
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

    # Keep a stable output order if Customer ID exists
    cols = []
    if customer_id_column in out.columns:
        cols.append(customer_id_column)
    cols += [c for c in out.columns if c not in cols]
    return out[cols]


def save_model(model: RandomForestClassifier, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)


def load_model(path: str | Path) -> RandomForestClassifier:
    return load(path)

