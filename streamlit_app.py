from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.rfm_model import RfmModelConfig, load_model, predict_rfm_with_buckets, save_model, train_rfm_churn_model


APP_TITLE = "Customer Churn Predictor"
APP_TAGLINE = "AI-powered retention analytics"
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


def inject_styles() -> None:
    st.markdown(
        """
<style>
/* ---- Base ---- */
:root {
  --bg0: #070a12;
  --bg1: #0b1220;
  --card: rgba(255,255,255,0.04);
  --card-border: rgba(255,255,255,0.08);
  --text: #e5e7eb;
  --muted: rgba(229,231,235,0.72);
  --green: #22c55e;
  --amber: #f59e0b;
  --red: #ef4444;
  --cyan: #06b6d4;
  --indigo: #6366f1;
}

.stApp {
  background:
    radial-gradient(900px circle at 15% 12%, rgba(34,197,94,0.14), transparent 42%),
    radial-gradient(900px circle at 82% 18%, rgba(99,102,241,0.18), transparent 45%),
    radial-gradient(700px circle at 70% 75%, rgba(6,182,212,0.12), transparent 40%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
  color: var(--text);
}

/* Hide Streamlit default header + footer spacing */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
  background: rgba(7,10,18,0.6);
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] > div {
  padding-top: 18px;
}

/* ---- Cards (general) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  backdrop-filter: blur(10px);
}

/* ---- Buttons ---- */
.stButton>button, .stDownloadButton>button {
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: linear-gradient(135deg, rgba(34,197,94,0.85), rgba(99,102,241,0.85)) !important;
  color: white !important;
  font-weight: 650 !important;
  padding: 0.65rem 1rem !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}

/* ---- Inputs ---- */
div[data-baseweb="input"] input, textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}

/* ---- Dataframe ---- */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.07);
}

/* ---- Small badges ---- */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.28rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(34,197,94,0.35);
  background: rgba(34,197,94,0.10);
  color: rgba(34,197,94,0.95);
  font-weight: 600;
  font-size: 0.82rem;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: rgba(34,197,94,0.95);
  box-shadow: 0 0 0 3px rgba(34,197,94,0.15);
}

/* Metric title/values */
div[data-testid="stMetric"] label {
  color: rgba(229,231,235,0.70) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def format_currency(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def _get_customer_column(pred: pd.DataFrame) -> str:
    if "Customer ID" in pred.columns:
        return "Customer ID"
    # Fallback for unexpected schemas: use first column.
    return str(pred.columns[0])


def make_display_df(pred: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table matching the UI: Customer, Churn Prob, Value, Risk, Segment."""
    pred = pred.copy()
    customer_col = _get_customer_column(pred)

    display = pd.DataFrame(
        {
            "Customer": pred[customer_col].astype(str),
            "Churn Prob.": pred["Churn_Probability"].astype(float),
            "Value": pred.get("Value_Level", pd.Series(["Low Value"] * len(pred))),
            "Risk Level": pred.get("Risk_Level", pd.Series(["Low Risk"] * len(pred))),
            "Segment": pred.get("Value_Risk_Bucket", pd.Series(["Low Value Low Risk"] * len(pred))),
            "Churn Pred": pred.get("Churn_Pred", pd.Series([0] * len(pred))).astype(int),
        }
    )
    return display


def render_predictions_table_html(df: pd.DataFrame, *, churn_threshold: float, max_rows: int = 50) -> str:
    df = df.head(max_rows).copy()

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def pill_bg(text: str) -> str:
        # Return inline style background color for a pill
        t = text.lower()
        if "high value" in t:
            return "rgba(34,197,94,0.18)"
        if "low value" in t:
            return "rgba(245,158,11,0.18)"
        if "high risk" in t:
            return "rgba(239,68,68,0.18)"
        if "low risk" in t:
            return "rgba(34,197,94,0.12)"
        return "rgba(255,255,255,0.08)"

    def pill_border(text: str) -> str:
        t = text.lower()
        if "high value" in t:
            return "rgba(34,197,94,0.45)"
        if "low value" in t:
            return "rgba(245,158,11,0.45)"
        if "high risk" in t:
            return "rgba(239,68,68,0.45)"
        if "low risk" in t:
            return "rgba(34,197,94,0.28)"
        return "rgba(255,255,255,0.18)"

    rows_html = []
    for _, r in df.iterrows():
        prob = float(r["Churn Prob."])
        bar_pct = max(0.0, min(1.0, prob)) * 100.0
        risk_color = "rgba(239,68,68,0.95)" if prob >= churn_threshold else "rgba(34,197,94,0.95)"
        risk_bar = "rgba(239,68,68,0.18)" if prob >= churn_threshold else "rgba(34,197,94,0.14)"

        value = str(r["Value"])
        risk = str(r["Risk Level"])
        segment = str(r["Segment"])

        rows_html.append(
            f"""
            <tr>
              <td style="padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.06);min-width:160px;">
                <div style="font-weight:700;color:rgba(229,231,235,0.95);">{r['Customer']}</div>
              </td>
              <td style="padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.06);min-width:220px;">
                <div style="font-weight:700;color:rgba(229,231,235,0.95);">{pct(prob)}</div>
                <div style="height:7px;background:{risk_bar};border-radius:999px;overflow:hidden;margin-top:8px;">
                  <div style="height:100%;width:{bar_pct:.1f}%;background:{risk_color};border-radius:999px;"></div>
                </div>
              </td>
              <td style="padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.06);min-width:140px;">
                <span style="display:inline-block;padding:6px 10px;border-radius:999px;border:1px solid {pill_border(value)};background:{pill_bg(value)};color:rgba(229,231,235,0.95);font-weight:700;">
                  {value}
                </span>
              </td>
              <td style="padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.06);min-width:140px;">
                <span style="display:inline-block;padding:6px 10px;border-radius:999px;border:1px solid {pill_border(risk)};background:{pill_bg(risk)};color:rgba(229,231,235,0.95);font-weight:700;">
                  {risk}
                </span>
              </td>
              <td style="padding:12px 10px;border-bottom:1px solid rgba(255,255,255,0.06);min-width:160px;color:rgba(229,231,235,0.82);font-weight:650;">
                {segment}
              </td>
            </tr>
            """
        )

    table = f"""<div style="border:1px solid rgba(255,255,255,0.08);border-radius:16px;overflow:hidden;">
<table style="width:100%;border-collapse:collapse;">
<thead>
<tr>
<th style="text-align:left;padding:10px 10px;background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.9rem;color:rgba(229,231,235,0.65);">Customer</th>
<th style="text-align:left;padding:10px 10px;background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.9rem;color:rgba(229,231,235,0.65);">Churn Prob.</th>
<th style="text-align:left;padding:10px 10px;background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.9rem;color:rgba(229,231,235,0.65);">Value</th>
<th style="text-align:left;padding:10px 10px;background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.9rem;color:rgba(229,231,235,0.65);">Risk Level</th>
<th style="text-align:left;padding:10px 10px;background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.9rem;color:rgba(229,231,235,0.65);">Segment</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</div>"""
    # Critical: Streamlit markdown treats any line starting with indentation
    # (typically 4 spaces) as an indented code block.
    # So we left-strip *every* line.
    html = table.lstrip()
    return "\n".join(line.lstrip() for line in html.splitlines())


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_styles()

    top_left, top_right = st.columns([0.78, 0.22], vertical_alignment="center")
    with top_left:
        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.35rem;">
  <div style="width:40px;height:40px;border-radius:999px;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.25);display:flex;align-items:center;justify-content:center;">
    <div style="width:18px;height:18px;border-radius:6px;background:linear-gradient(135deg, rgba(34,197,94,0.95), rgba(99,102,241,0.95));"></div>
  </div>
  <div>
    <div style="font-size:1.55rem;font-weight:750;line-height:1.1;">{APP_TITLE}</div>
    <div style="font-size:0.95rem;color:rgba(229,231,235,0.68);margin-top:0.1rem;">{APP_TAGLINE}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown(
            '<div style="display:flex;justify-content:flex-end;"><span class="badge"><span class="dot"></span> Model Active</span></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    with st.sidebar:
        st.markdown("### Configuration")
        st.caption("Upload data and set parameters")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.write("")

        churn_threshold = st.slider("Churn Threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        st.caption("Flag customers above this churn probability")

        value_quantile = st.slider(
            "Value Quantile",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
        )
        st.caption("High-value customers in the top percentile by Monetary")

        model_path = DEFAULT_MODEL_PATH

        st.write("")
        run_pred = st.button("Run Prediction", use_container_width=True)

    if not uploaded:
        st.info("Upload a CSV to start. Required columns: Frequency, Monetary, AvgOrderValue (Customer ID optional).")
        return

    df = pd.read_csv(uploaded)
    if not run_pred and "pred" not in st.session_state:
        st.caption("Upload complete. Click **Run Prediction** in the sidebar.")
        st.dataframe(df.head(20), use_container_width=True, height=320)
        return

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
        st.session_state["pred"] = pred
    except Exception as e:
        st.error(str(e))
        st.code(
            "Your CSV must contain at least: Frequency, Monetary, AvgOrderValue "
            "(and optionally Customer ID)."
        )
        return

    pred = st.session_state["pred"]

    # ---- KPI cards ----
    total_customers = int(len(pred))
    churned = int(pred["Churn_Pred"].sum())
    hvhr = pred[pred["Value_Risk_Bucket"] == "High Value High Risk"].copy()
    hvhr_count = int(len(hvhr))
    avg_prob = float(np.mean(pred["Churn_Probability"])) if len(pred) else 0.0
    churn_rate = churned / total_customers if total_customers else 0.0
    hvhr_rate = hvhr_count / total_customers if total_customers else 0.0

    display_cards = f"""
    <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-top:4px;">
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="font-weight:800;color:rgba(229,231,235,0.65);">Total Customers</div>
          <div style="width:28px;height:28px;border-radius:10px;background:rgba(34,197,94,0.14);border:1px solid rgba(34,197,94,0.25);display:flex;align-items:center;justify-content:center;color:rgba(34,197,94,0.95);font-weight:900;">👥</div>
        </div>
        <div style="font-size:1.75rem;font-weight:900;margin-top:6px;color:rgba(229,231,235,0.95);">{total_customers:,}</div>
        <div style="margin-top:6px;color:rgba(229,231,235,0.62);font-weight:650;font-size:0.9rem;">▲ {hvhr_rate:.1%} in priority bucket</div>
      </div>
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="font-weight:800;color:rgba(229,231,235,0.65);">Predicted Churn</div>
          <div style="width:28px;height:28px;border-radius:10px;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.25);display:flex;align-items:center;justify-content:center;color:rgba(239,68,68,0.95);font-weight:900;">⚠️</div>
        </div>
        <div style="font-size:1.75rem;font-weight:900;margin-top:6px;color:rgba(229,231,235,0.95);">{churned:,}</div>
        <div style="margin-top:6px;color:rgba(229,231,235,0.62);font-weight:650;font-size:0.9rem;">{churn_rate:.1%} churn rate</div>
      </div>
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="font-weight:800;color:rgba(229,231,235,0.65);">High Value at Risk</div>
          <div style="width:28px;height:28px;border-radius:10px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.25);display:flex;align-items:center;justify-content:center;color:rgba(245,158,11,0.95);font-weight:900;">🎯</div>
        </div>
        <div style="font-size:1.75rem;font-weight:900;margin-top:6px;color:rgba(229,231,235,0.95);">{hvhr_count:,}</div>
        <div style="margin-top:6px;color:rgba(229,231,235,0.62);font-weight:650;font-size:0.9rem;">{hvhr_rate:.1%} priority share</div>
      </div>
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:14px;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="font-weight:800;color:rgba(229,231,235,0.65);">Avg Churn Probability</div>
          <div style="width:28px;height:28px;border-radius:10px;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);display:flex;align-items:center;justify-content:center;color:rgba(99,102,241,0.95);font-weight:900;">📈</div>
        </div>
        <div style="font-size:1.75rem;font-weight:900;margin-top:6px;color:rgba(229,231,235,0.95);">{avg_prob:.1%}</div>
        <div style="margin-top:6px;color:rgba(229,231,235,0.62);font-weight:650;font-size:0.9rem;">Model confidence proxy</div>
      </div>
    </div>
    """
    st.markdown(display_cards, unsafe_allow_html=True)

    st.write("")

    tab_pred, tab_hvhr, tab_sum = st.tabs(["Predictions", "High Value High Risk", "Summary"])

    with tab_pred:
        c1, c2 = st.columns([0.72, 0.28], vertical_alignment="center")
        with c1:
            q = st.text_input("Search customers…", placeholder="Type Customer ID (or any text match)")
        with c2:
            csv_bytes = pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="rfm_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        view = pred
        if q:
            mask = pd.Series(False, index=view.index)
            for col in [c for c in ["Customer ID"] if c in view.columns]:
                mask = mask | view[col].astype(str).str.contains(q, case=False, na=False)
            # fallback: search all columns if Customer ID not present or no match
            if not mask.any():
                mask = view.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)
            view = view[mask].copy()

        display_view = make_display_df(view)
        html_table = render_predictions_table_html(display_view, churn_threshold=float(churn_threshold), max_rows=50)
        st.markdown(html_table, unsafe_allow_html=True)
        st.caption("Showing up to 50 rows. Download the full CSV for all customers.")

    with tab_hvhr:
        c1, c2 = st.columns([0.72, 0.28], vertical_alignment="center")
        with c1:
            st.markdown("#### Priority retention list")
            st.caption("High Value + High Risk customers to target first.")
        with c2:
            hvhr_bytes = hvhr.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download HVHR CSV",
                data=hvhr_bytes,
                file_name="high_value_high_risk_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )
        hvhr_display = make_display_df(hvhr)
        hvhr_html = render_predictions_table_html(hvhr_display, churn_threshold=float(churn_threshold), max_rows=50)
        st.markdown(hvhr_html, unsafe_allow_html=True)
        st.caption("Showing up to 50 rows. Download the HVHR CSV for full details.")

    with tab_sum:
        # Summary dashboard (retention analytics)
        monetary_col = "Monetary" if "Monetary" in pred.columns else ("Monetary_History" if "Monetary_History" in pred.columns else None)

        total_customers = len(pred)
        churned = int(pred["Churn_Pred"].sum()) if "Churn_Pred" in pred.columns else 0
        churn_rate = churned / total_customers if total_customers else 0.0
        retention_rate = 1.0 - churn_rate if total_customers else 0.0

        avg_churn_probability = float(pred["Churn_Probability"].mean()) if "Churn_Probability" in pred.columns and total_customers else 0.0

        hvhr = pred[pred.get("Value_Risk_Bucket", pd.Series([None] * len(pred))) == "High Value High Risk"].copy()
        hvhr_count = int(len(hvhr))
        hvhr_value_total = float(hvhr[monetary_col].sum()) if monetary_col and len(hvhr) else 0.0
        hvhr_value_avg = hvhr_value_total / hvhr_count if hvhr_count else 0.0

        # Derive 3 risk levels for UI (High/Medium/Low) from churn probability
        # Keep HighRisk aligned with the churn threshold used for prediction.
        mid_low = max(0.0, float(churn_threshold) * 0.5)
        mid_high = float(churn_threshold)

        def risk_band(p: float) -> str:
            if p >= mid_high:
                return "High Risk"
            if p >= mid_low:
                return "Medium Risk"
            return "Low Risk"

        risk_probs = pred["Churn_Probability"].astype(float) if "Churn_Probability" in pred.columns else pd.Series([0.0] * total_customers)
        risk_labels = risk_probs.apply(risk_band)
        pred = pred.copy()
        pred["Risk_UI"] = risk_labels

        risk_stats = (
            pred.groupby("Risk_UI", dropna=False)
            .size()
            .reset_index(name="Customers")
        )
        risk_stats["Share"] = risk_stats["Customers"] / total_customers if total_customers else 0.0

        risk_order = ["High Risk", "Medium Risk", "Low Risk"]
        risk_stats = risk_stats.set_index("Risk_UI").reindex(risk_order).fillna(0).reset_index()

        # Segment column (try common names)
        segment_col = None
        if "Segment_Name" in pred.columns:
            segment_col = "Segment_Name"
        elif "Segment" in pred.columns:
            segment_col = "Segment"
        else:
            segment_col = "Value_Risk_Bucket" if "Value_Risk_Bucket" in pred.columns else pred.columns[0]

        # Churn by segment (top segments by churn count)
        seg_stats = (
            pred.groupby(segment_col, dropna=False)
            .agg(Customers=("Churn_Pred", "size"), Churned=("Churn_Pred", "sum"))
            .reset_index()
        )
        seg_stats["ChurnShare"] = seg_stats["Churned"] / seg_stats["Customers"].replace(0, np.nan)
        seg_stats = seg_stats.sort_values("Churned", ascending=False).head(6)
        max_seg_churn = float(seg_stats["Churned"].max()) if len(seg_stats) else 0.0

        def fmt_pct(x: float) -> str:
            return f"{x * 100:.1f}%"

        def card_html(title: str, value: str, subtitle: str, icon_html: str, accent: str) -> str:
            return (
                f'<div style="background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.08);'
                f'border-radius:18px;padding:14px 16px;min-height:90px;">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;">'
                f'  <div>'
                f'    <div style="font-weight:800;color:rgba(229,231,235,0.72);font-size:0.95rem;">{title}</div>'
                f'    <div style="margin-top:6px;font-size:1.8rem;font-weight:900;color:rgba(229,231,235,0.96);">{value}</div>'
                f'    <div style="margin-top:6px;color:rgba(229,231,235,0.62);font-weight:650;font-size:0.88rem;">{subtitle}</div>'
                f'  </div>'
                f'  <div style="width:36px;height:36px;border-radius:14px;background:{accent};border:1px solid rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;">'
                f'    {icon_html}'
                f'  </div>'
                f'</div>'
                f'</div>'
            )

        def bar_row_html(label: str, customers: int, share: float, color: str) -> str:
            width = max(0.0, min(1.0, share)) * 100.0
            return (
                f'<div style="margin-top:14px;">'
                f'  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
                f'    <div style="font-weight:800;color:rgba(229,231,235,0.88);font-size:0.95rem;">{label}</div>'
                f'    <div style="font-weight:800;color:rgba(229,231,235,0.7);font-size:0.9rem;">{customers} ({fmt_pct(share)})</div>'
                f'  </div>'
                f'  <div style="height:10px;border-radius:999px;background:rgba(255,255,255,0.05);overflow:hidden;border:1px solid rgba(255,255,255,0.07);">'
                f'    <div style="height:100%;width:{width:.1f}%;background:{color};border-radius:999px;"></div>'
                f'  </div>'
                f'</div>'
            )

        def segment_row_html(seg: str, churned_count: int, share: float, accent_color: str, bar_color: str, max_value: float) -> str:
            width = (churned_count / max_value * 100.0) if max_value else 0.0
            return (
                f'<div style="margin-top:14px;">'
                f'  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
                f'    <div style="font-weight:800;color:rgba(229,231,235,0.88);font-size:0.95rem;">{seg}</div>'
                f'    <div style="font-weight:800;color:rgba(229,231,235,0.7);font-size:0.9rem;">{churned_count} ({fmt_pct(share)})</div>'
                f'  </div>'
                f'  <div style="height:10px;border-radius:999px;background:rgba(255,255,255,0.05);overflow:hidden;border:1px solid rgba(255,255,255,0.07);">'
                f'    <div style="height:100%;width:{width:.1f}%;background:{bar_color};border-radius:999px;"></div>'
                f'  </div>'
                f'</div>'
            )

        # Layout: top KPI cards
        top1, top2, top3 = st.columns(3, gap="large")
        with top1:
            st.markdown(
                card_html(
                    "Retention Rate",
                    fmt_pct(retention_rate),
                    f"{total_customers - churned} customers likely to stay",
                    "🔁",
                    "rgba(34,197,94,0.12)",
                ),
                unsafe_allow_html=True,
            )
        with top2:
            st.markdown(
                card_html(
                    "Churn Rate",
                    fmt_pct(churn_rate),
                    f"{churned} customers at risk",
                    "⚠️",
                    "rgba(239,68,68,0.12)",
                ),
                unsafe_allow_html=True,
            )
        with top3:
            st.markdown(
                card_html(
                    "Value at Risk",
                    format_currency(hvhr_value_total),
                    f"From {hvhr_count} high-value customers",
                    "$",
                    "rgba(245,158,11,0.12)",
                ),
                unsafe_allow_html=True,
            )

        st.write("")

        # Risk Distribution + Churn by Segment panels
        left_panel, right_panel = st.columns(2, gap="large")
        with left_panel:
            left_html = (
                '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:16px;min-height:280px;">'
                '<div style="font-weight:900;color:rgba(229,231,235,0.85);font-size:1.05rem;margin-bottom:8px;">Risk Distribution</div>'
            )
            left_html += bar_row_html(
                "High Risk",
                int(risk_stats.loc[risk_stats["Risk_UI"] == "High Risk", "Customers"].iloc[0]) if not risk_stats.empty else 0,
                float(risk_stats.loc[risk_stats["Risk_UI"] == "High Risk", "Share"].iloc[0]) if not risk_stats.empty else 0.0,
                "rgba(239,68,68,0.95)",
            )
            left_html += bar_row_html(
                "Medium Risk",
                int(risk_stats.loc[risk_stats["Risk_UI"] == "Medium Risk", "Customers"].iloc[0]) if not risk_stats.empty else 0,
                float(risk_stats.loc[risk_stats["Risk_UI"] == "Medium Risk", "Share"].iloc[0]) if not risk_stats.empty else 0.0,
                "rgba(245,158,11,0.95)",
            )
            left_html += bar_row_html(
                "Low Risk",
                int(risk_stats.loc[risk_stats["Risk_UI"] == "Low Risk", "Customers"].iloc[0]) if not risk_stats.empty else 0,
                float(risk_stats.loc[risk_stats["Risk_UI"] == "Low Risk", "Share"].iloc[0]) if not risk_stats.empty else 0.0,
                "rgba(34,197,94,0.95)",
            )
            left_html += "</div>"
            st.markdown(left_html, unsafe_allow_html=True)

        with right_panel:
            right_html = (
                '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:16px;min-height:280px;">'
                '<div style="font-weight:900;color:rgba(229,231,235,0.85);font-size:1.05rem;margin-bottom:8px;">Churn by Segment</div>'
            )
            seg_html = '<div style="background:transparent;">'
            for _, row in seg_stats.iterrows():
                seg = str(row[segment_col])
                churned_count = int(row["Churned"])
                share = float(row["ChurnShare"]) if not pd.isna(row["ChurnShare"]) else 0.0
                # Color intensity based on churn share
                if share >= float(churn_threshold):
                    bar_color = "rgba(239,68,68,0.95)"
                elif share >= (float(churn_threshold) * 0.5):
                    bar_color = "rgba(245,158,11,0.95)"
                else:
                    bar_color = "rgba(34,197,94,0.95)"
                seg_html += segment_row_html(seg, churned_count, share, "", bar_color, max_seg_churn)
            seg_html += "</div>"
            right_html += seg_html
            right_html += "</div>"
            st.markdown(right_html, unsafe_allow_html=True)

        st.write("")

        # Key Insights cards row
        k1, k2, k3 = st.columns(3, gap="large")
        with k1:
            st.markdown(
                card_html(
                    "Average churn probability",
                    fmt_pct(avg_churn_probability),
                    "Average churn probability across uploaded customers",
                    "∿",
                    "rgba(99,102,241,0.12)",
                ),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                card_html(
                    "High-value customers at risk",
                    f"{hvhr_count}",
                    "Customers in the priority bucket",
                    "🎯",
                    "rgba(245,158,11,0.12)",
                ),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                card_html(
                    "Avg value per at-risk customer",
                    format_currency(hvhr_value_avg),
                    "Helps prioritize retention spend",
                    "$",
                    "rgba(34,197,94,0.12)",
                ),
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

