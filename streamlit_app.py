from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.rfm_model import (
    RfmModelConfig,
    load_model,
    predict_rfm_with_buckets,
    save_model,
    train_rfm_churn_model,
)


APP_TITLE = "Customer Churn Prediction (RFM Upload)"
DEFAULT_MODEL_PATH = Path("models/rfm_churn_model.joblib")
DEFAULT_TRAINING_DATA = Path("rfm_features.csv")
DEFAULT_REPORT_PATH = Path("reports/rfm_classification_report.txt")


def ensure_rfm_model(model_path: Path) -> None:
    if model_path.exists():
        return

    if not DEFAULT_TRAINING_DATA.exists():
        raise FileNotFoundError(
            "No trained model found and no training data available. "
            f"Expected either {model_path.as_posix()} or {DEFAULT_TRAINING_DATA.as_posix()}."
        )

    df_train = pd.read_csv(DEFAULT_TRAINING_DATA)
    model, report = train_rfm_churn_model(df_train, RfmModelConfig())
    save_model(model, model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.markdown(
        """
Upload a CSV containing **RFM features** and get:

- **Churn prediction** (`Churn_Pred`) and **probability** (`Churn_Probability`)
- Customer category (**Value_Risk_Bucket**):
  - High Value High Risk
  - High Value Low Risk
  - Low Value High Risk
  - Low Value Low Risk

**Required columns in your upload:**
- `Customer ID` (optional but recommended)
- `Frequency`
- `Monetary`
- `AvgOrderValue`
"""
    )

    with st.sidebar:
        st.header("Settings")
        churn_threshold = st.slider("Churn threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        value_quantile = st.slider("Value split quantile (Monetary)", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        model_path_str = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
        model_path = Path(model_path_str)

        if st.button("Train model (if missing)"):
            ensure_rfm_model(model_path)
            st.success(f"Model ready at {model_path.as_posix()}")

    uploaded = st.file_uploader("Upload RFM CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to start.")
        return

    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Ensure model exists
    try:
        ensure_rfm_model(model_path)
        model = load_model(model_path)
    except Exception as e:
        st.error(str(e))
        return

    try:
        pred = predict_rfm_with_buckets(
            model,
            df,
            churn_threshold=float(churn_threshold),
            value_threshold_quantile=float(value_quantile),
            customer_id_column="Customer ID",
        )
    except Exception as e:
        st.error(str(e))
        st.code(
            "Your CSV must contain at least: Frequency, Monetary, AvgOrderValue "
            "(and optionally Customer ID)."
        )
        return

    st.subheader("Predictions")
    st.dataframe(pred.head(50), use_container_width=True)

    st.subheader("High Value High Risk (priority retention list)")
    hvhr = pred[pred["Value_Risk_Bucket"] == "High Value High Risk"].copy()
    st.write(f"Customers in **High Value High Risk**: {len(hvhr)}")
    st.dataframe(hvhr.head(50), use_container_width=True)

    st.subheader("Bucket summary")
    summary = pred.groupby(["Value_Risk_Bucket", "Churn_Pred"], dropna=False).size().reset_index(name="Customers")
    st.dataframe(summary, use_container_width=True)

    csv_bytes = pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name="rfm_predictions.csv",
        mime="text/csv",
    )

    hvhr_bytes = hvhr.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download High Value High Risk CSV",
        data=hvhr_bytes,
        file_name="high_value_high_risk_customers.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

