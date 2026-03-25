from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.churn_pipeline import (
    PipelineConfig,
    build_customer_features,
    clean_transactions,
    load_model,
    load_raw_transactions,
    predict_with_buckets,
    save_model,
    train_churn_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train churn model and generate customer churn predictions.")
    p.add_argument(
        "--data",
        default="online_retail_II 2009-2010.csv",
        help="Path to raw Online Retail II CSV (default: repo root file).",
    )
    p.add_argument(
        "--model",
        default="models/churn_model.joblib",
        help="Path to save/load the churn model (joblib).",
    )
    p.add_argument(
        "--out",
        default="customer_predictions.csv",
        help="Output CSV path for predictions.",
    )
    p.add_argument(
        "--churn-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for churn prediction and risk bucketing.",
    )
    p.add_argument(
        "--value-quantile",
        type=float,
        default=0.5,
        help="Quantile threshold for value split using Monetary (0.5 = median).",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="If set, trains a new model and overwrites --model.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PipelineConfig()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    raw = load_raw_transactions(data_path)
    tx = clean_transactions(raw)
    features = build_customer_features(tx, cfg)

    model_path = Path(args.model)
    if args.train or not model_path.exists():
        model, report = train_churn_model(features, cfg)
        save_model(model, model_path)
        (Path("reports")).mkdir(parents=True, exist_ok=True)
        Path("reports/classification_report.txt").write_text(report, encoding="utf-8")
    else:
        model = load_model(model_path)

    pred = predict_with_buckets(
        model,
        features,
        churn_threshold=float(args.churn_threshold),
        value_threshold_quantile=float(args.value_quantile),
    )

    out_path = Path(args.out)
    pred.to_csv(out_path, index=False)

    # Also export a small summary table
    summary = (
        pred.groupby(["Value_Risk_Bucket", "Churn_Pred"], dropna=False)
        .size()
        .reset_index(name="Customers")
        .sort_values(["Value_Risk_Bucket", "Churn_Pred"])
    )
    summary.to_csv("bucket_summary.csv", index=False)

    print(f"Wrote predictions to: {out_path}")
    print("Wrote bucket summary to: bucket_summary.csv")
    print("Model path:", model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

