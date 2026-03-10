"""
app.py — Pepper Price Forecasting Dashboard
Streamlit UI with district/market dropdowns, EDA charts, model validation,
and 30-day recursive price-band forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config — must be FIRST streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Pepper Price Forecast",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model  import run_pipeline, TARGETS
from charts import (
    fig_historical_bands,
    fig_forecast,
    fig_forecast_table,
    fig_validation,
    fig_residuals,
    fig_feature_importance,
    fig_seasonality,
    fig_yearly_trend,
    fig_price_spread,
    fig_arrivals_vs_price,
    fig_weather_correlation,
    fig_decomposition,
    metrics_summary,
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global dark theme override */
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    background-color: #0f1117 !important;
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1d27 !important;
    border-right: 1px solid #2a2d3a;
}
/* Metric cards */
[data-testid="metric-container"] {
    background-color: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"]  { color: #90caf9 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 1.35rem !important; }
[data-testid="stMetricDelta"]  { font-size: 0.82rem !important; }
/* Tab styling */
[data-testid="stTabs"] button {
    color: #90caf9 !important;
    font-weight: 600;
    font-size: 0.88rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #4fc3f7 !important;
}
/* Headings */
h1 { color: #4fc3f7 !important; letter-spacing: -0.5px; }
h2 { color: #90caf9 !important; }
h3 { color: #b0bec5 !important; }
/* Selectbox / dropdown */
[data-testid="stSelectbox"] label { color: #90caf9 !important; font-weight: 600; }
/* Spinner */
[data-testid="stSpinner"] p { color: #66bb6a !important; }
/* Divider */
hr { border-color: #2a2d3a !important; }
/* Info/warning boxes */
.stAlert { border-radius: 8px; }
/* Section badge */
.badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.75rem; font-weight:700; letter-spacing:0.5px;
    background:#1F3864; color:#90caf9; border:1px solid #2E75B6;
    margin-bottom:6px;
}
/* KPI section header */
.kpi-header {
    font-size:0.72rem; font-weight:700; letter-spacing:1px;
    color:#546e7a; text-transform:uppercase; margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data/final.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Model caching (per market) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_result(market: str):
    df = load_data()
    return run_pipeline(df, market)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## 🌿 Pepper Forecast")
    st.sidebar.markdown("**Karnataka APMC Markets**")
    st.sidebar.markdown("---")

    # District dropdown
    districts = sorted(df["District"].dropna().unique())
    district = st.sidebar.selectbox("📍 Select District", ["All Districts"] + districts)

    # Market dropdown — sorted by record count DESC so the most data-rich
    # market (SIRSI) is always the default first option.
    if district == "All Districts":
        subset = df
    else:
        subset = df[df["District"] == district]

    market_counts = (
        subset.groupby("Market")
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )
    markets = market_counts["Market"].tolist()   # most records first

    market_labels = {
        row["Market"]: f"{row['Market']}  ({row['records']:,} records)"
        for _, row in market_counts.iterrows()
    }
    market = st.sidebar.selectbox(
        "🏪 Select Market",
        options=markets,
        format_func=lambda m: market_labels[m],
    )

    st.sidebar.markdown("---")

    # Market info card
    mdf_info = df[df["Market"] == market]
    if not mdf_info.empty:
        dist_name = mdf_info["District"].iloc[0]
        rec_count = len(mdf_info)
        date_min  = mdf_info["date"].min().strftime("%b %Y")
        date_max  = mdf_info["date"].max().strftime("%b %Y")
        latest    = mdf_info.sort_values("date").iloc[-1]

        st.sidebar.markdown(f"""
<div style='background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:14px;'>
  <div style='color:#90caf9;font-size:0.7rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;'>Market Info</div>
  <div style='color:#e0e0e0;font-size:0.85rem;'><b>{market}</b></div>
  <div style='color:#b0bec5;font-size:0.75rem;'>📍 {dist_name}</div>
  <hr style='border-color:#2a2d3a;margin:8px 0;'>
  <div style='color:#b0bec5;font-size:0.72rem;'>📅 {date_min} → {date_max}</div>
  <div style='color:#b0bec5;font-size:0.72rem;'>📊 {rec_count:,} records</div>
  <hr style='border-color:#2a2d3a;margin:8px 0;'>
  <div style='color:#66bb6a;font-size:0.72rem;font-weight:700;'>Latest Prices</div>
  <div style='color:#e0e0e0;font-size:0.8rem;'>Min: ₹{latest["Min"]:,.0f}</div>
  <div style='color:#e0e0e0;font-size:0.8rem;'>Max: ₹{latest["Max"]:,.0f}</div>
  <div style='color:#e0e0e0;font-size:0.8rem;'>Modal: ₹{latest["Modal"]:,.0f}</div>
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
<div style='color:#546e7a;font-size:0.7rem;text-align:center;'>
Model: GBM + RF + Ridge Ensemble<br>
Forecast: 30-day recursive<br>
Features: 23 per target<br><br>
<span style='color:#2a2d3a;'>─────────────────</span><br>
Data: Karnataka APMC 2019–2025
</div>
""", unsafe_allow_html=True)

    return market


# ── KPI strip ──────────────────────────────────────────────────────────────────
def render_kpi_strip(result: dict, market: str):
    kpis = metrics_summary(result)

    st.markdown("### 📊 Model Performance & 30-Day Forecast Snapshot")

    cols = st.columns(9)
    label_map = {
        "MAPE":      ("MAPE %",       "{:.2f}%"),
        "MAE":       ("MAE (₹/q)",    "₹{:,.0f}"),
        "Day1":      ("Day 1 (₹/q)",  "₹{:,.0f}"),
        "Day30":     ("Day 30 (₹/q)", "₹{:,.0f}"),
        "Trend_pct": ("30d Trend",    "{:+.1f}%"),
    }
    colour_map = {"Min": "#4fc3f7", "Max": "#ef5350", "Modal": "#66bb6a"}
    idx = 0

    for col_name in ["Min", "Max", "Modal"]:
        for kpi_key in ["Day1", "Day30", "Trend_pct"]:
            lbl, fmt = label_map[kpi_key]
            val = kpis[col_name][kpi_key]
            delta_str = None
            if kpi_key == "Trend_pct":
                delta_str = fmt.format(val)
            with cols[idx]:
                st.metric(
                    label=f"{col_name} · {lbl}",
                    value=fmt.format(val) if kpi_key != "Trend_pct" else "—",
                    delta=delta_str,
                    delta_color="normal",
                )
            idx += 1

    # Accuracy row
    st.markdown("##### Model Accuracy on Held-out Test Set")
    acc_cols = st.columns(6)
    for i, col_name in enumerate(["Min", "Max", "Modal"]):
        with acc_cols[i * 2]:
            st.metric(f"{col_name} MAPE", f"{kpis[col_name]['MAPE']:.2f}%")
        with acc_cols[i * 2 + 1]:
            st.metric(f"{col_name} MAE",  f"₹{kpis[col_name]['MAE']:,.0f}")

    # Colour-coded accuracy badge
    modal_mape = kpis["Modal"]["MAPE"]
    if modal_mape < 5:
        badge_col, badge_txt = "#1b5e20", f"🟢 High Accuracy  Modal MAPE {modal_mape:.2f}%"
    elif modal_mape < 10:
        badge_col, badge_txt = "#e65100", f"🟡 Moderate Accuracy  Modal MAPE {modal_mape:.2f}%"
    else:
        badge_col, badge_txt = "#b71c1c", f"🔴 Low Accuracy — interpret forecasts with caution  Modal MAPE {modal_mape:.2f}%"

    st.markdown(f"""
<div style='background:{badge_col};border-radius:8px;padding:8px 16px;
     display:inline-block;color:#ffffff;font-size:0.82rem;font-weight:600;margin-top:4px;'>
  {badge_txt}
</div>
""", unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    market = render_sidebar(df)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"# 🌿 Pepper Price Forecast")
    st.markdown(f"<h3 style='color:#90caf9;margin-top:-10px;'>{market} Market &nbsp;·&nbsp; 30-Day Price Intelligence</h3>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner(f"Training ensemble models for {market}…"):
        result = get_result(market)

    # Handle insufficient data
    if result is None or result.get("status") == "insufficient_data":
        rows = result.get("rows", 0) if result else 0
        st.warning(f"""
**⚠️ Insufficient data for {market}**

This market has only **{rows} usable rows** after feature engineering (minimum required: 30).

Markets typically excluded are those with very few historical trading days recorded in the APMC database.
Please select a different market from the sidebar.
        """)
        # Still show whatever raw EDA we can
        mdf_raw = df[df["Market"] == market].sort_values("date")
        if len(mdf_raw) > 2:
            st.markdown("#### Raw Historical Data (limited)")
            st.dataframe(mdf_raw[["date","Min","Max","Modal","Arrivals"]].tail(50),
                         use_container_width=True)
        return

    mdf = result["mdf"]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    render_kpi_strip(result, market)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🔮 Forecast",
        "📈 Historical",
        "🧭 EDA",
        "✅ Validation",
        "📊 Residuals",
        "🔑 Features",
        "📋 Forecast Table",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1 — FORECAST
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="badge">30-DAY RECURSIVE FORECAST</div>', unsafe_allow_html=True)
        st.markdown("##### Min · Max · Modal price bands with 90% confidence ribbons")
        st.plotly_chart(fig_forecast(result, market), use_container_width=True, key="forecast_main")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Price Spread (Max − Min)")
            st.plotly_chart(fig_price_spread(mdf, result, market), use_container_width=True, key="forecast_spread")
        with c2:
            st.markdown("##### Seasonal Context")
            st.plotly_chart(fig_seasonality(mdf, market), use_container_width=True, key="seasonality_1")

        # Forecast summary callout
        fc_modal = result["forecasts"]["Modal"]
        d1  = fc_modal["prices"][0]
        d30 = fc_modal["prices"][-1]
        chg = (d30 / d1 - 1) * 100
        arrow = "📈" if chg > 0 else "📉"
        color = "#1b5e20" if chg > 0 else "#b71c1c"
        st.markdown(f"""
<div style='background:{color};border-radius:10px;padding:16px 22px;margin-top:10px;'>
  <span style='font-size:1.1rem;font-weight:700;color:#fff;'>
  {arrow} {market} Modal price forecast: ₹{d1:,.0f} → ₹{d30:,.0f}
  &nbsp; ({chg:+.1f}% over 30 days)
  </span><br>
  <span style='font-size:0.8rem;color:#ccc;'>
  Confidence band: ₹{fc_modal["low"][-1]:,.0f} – ₹{fc_modal["high"][-1]:,.0f} on Day 30
  </span>
</div>
""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2 — HISTORICAL
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="badge">HISTORICAL PRICE BANDS</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_historical_bands(mdf, market), use_container_width=True, key="historical_bands")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_yearly_trend(mdf, market), use_container_width=True, key="yearly_trend")
        with c2:
            st.plotly_chart(fig_arrivals_vs_price(mdf, market), use_container_width=True, key="arrivals_price")

        st.plotly_chart(fig_decomposition(mdf, market), use_container_width=True, key="decomposition")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3 — EDA
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="badge">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

        st.plotly_chart(fig_seasonality(mdf, market), use_container_width=True, key="seasonality_eda")
        st.plotly_chart(fig_weather_correlation(mdf, market), use_container_width=True, key="weather_corr")

        # Summary stats table
        st.markdown("##### Descriptive Statistics")
        stats = mdf[["Min","Max","Modal","Arrivals",
                     "avg_rainfall","temperature_2m_avg"]].describe().round(1)
        st.dataframe(stats, use_container_width=True)

        # Raw data preview
        with st.expander("📂 Raw Data Preview (last 60 rows)"):
            show_cols = ["Min","Max","Modal","Arrivals",
                         "avg_rainfall","temperature_2m_avg",
                         "soil_moisture_0_to_7cm_avg"]
            st.dataframe(mdf[show_cols].tail(60).reset_index()
                           .rename(columns={"date":"Date"}),
                         use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4 — VALIDATION
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="badge">MODEL VALIDATION — HELD-OUT TEST SET</div>', unsafe_allow_html=True)
        st.markdown(f"""
<p style='color:#b0bec5;font-size:0.85rem;'>
The model is trained on all data up to the last {len(result["test"])} days,
then evaluated on the held-out test set without re-fitting.
This mirrors real-world deployment conditions.
</p>
""", unsafe_allow_html=True)
        st.plotly_chart(fig_validation(result, market), use_container_width=True, key="validation")

        # Per-target metric cards
        st.markdown("##### Per-Target Accuracy Breakdown")
        c1, c2, c3 = st.columns(3)
        for col_widget, col_name, color in zip(
                [c1, c2, c3], ["Min", "Max", "Modal"],
                ["#4fc3f7", "#ef5350", "#66bb6a"]):
            m = result["metrics"][col_name]
            with col_widget:
                st.markdown(f"""
<div style='background:#1a1d27;border:1px solid {color};border-radius:10px;
     padding:14px;text-align:center;'>
  <div style='color:{color};font-size:1rem;font-weight:700;'>{col_name} Price</div>
  <hr style='border-color:#2a2d3a;margin:8px 0;'>
  <div style='color:#e0e0e0;font-size:0.9rem;'>MAPE: <b>{m["MAPE"]:.2f}%</b></div>
  <div style='color:#e0e0e0;font-size:0.9rem;'>MAE:  <b>₹{m["MAE"]:,.0f}</b></div>
  <div style='color:#e0e0e0;font-size:0.9rem;'>RMSE: <b>₹{m["RMSE"]:,.0f}</b></div>
</div>
""", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5 — RESIDUALS
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="badge">RESIDUAL DIAGNOSTICS</div>', unsafe_allow_html=True)
        st.markdown("""
<p style='color:#b0bec5;font-size:0.85rem;'>
Residuals = Actual − Predicted on the test set.
Ideally centred on zero with no systematic pattern over time.
</p>
""", unsafe_allow_html=True)
        st.plotly_chart(fig_residuals(result, market), use_container_width=True, key="residuals")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6 — FEATURES
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="badge">FEATURE IMPORTANCE — GBM</div>', unsafe_allow_html=True)
        st.markdown("""
<p style='color:#b0bec5;font-size:0.85rem;'>
Top-15 features by GBM importance for each price band.
Lag and rolling-mean features consistently dominate,
confirming the strong autocorrelative structure of Pepper prices.
</p>
""", unsafe_allow_html=True)
        st.plotly_chart(fig_feature_importance(result, market), use_container_width=True, key="feature_imp")

        # Feature engineering reference table
        with st.expander("📖 Full Feature Engineering Reference"):
            feat_rows = [
                ("Calendar",          "dayofyear, month, weekofyear, year",             "4",  "Capture annual & intra-annual seasonality"),
                ("Cyclical encoding", "sin_doy, cos_doy",                               "2",  "Circular representation; prevents day 365 ≠ day 1 discontinuity"),
                ("Lag prices",        "lag_1, lag_3, lag_7, lag_14, lag_21, lag_30",    "6",  "Momentum signals at multiple timescales"),
                ("YoY lag",           "lag_365",                                         "1",  "Same-season prior-year reference; captures harvest cycle"),
                ("Rolling mean",      "roll_mean_7, roll_mean_14, roll_mean_30",         "3",  "Trend direction proxy at 3 timescales"),
                ("Rolling std dev",   "roll_std_7, roll_std_14, roll_std_30",            "3",  "Local volatility / market uncertainty"),
                ("Weather",           "temperature_2m_avg, avg_rainfall",               "2",  "Seasonal demand and supply disruption signals"),
                ("Soil",              "soil_moisture_avg, soil_temperature_avg",         "2",  "Crop physiology covariates"),
                ("Total",             "",                                                "23", "Per target  ×  3 targets  =  69 features"),
            ]
            feat_df = pd.DataFrame(feat_rows,
                                   columns=["Category","Features","Count","Rationale"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7 — FORECAST TABLE
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="badge">30-DAY FORECAST TABLE</div>', unsafe_allow_html=True)

        # Build clean DataFrame for display + download
        rows = []
        for i, d in enumerate(result["forecasts"]["Modal"]["dates"]):
            rows.append({
                "Day":           i + 1,
                "Date":          d.strftime("%d %b %Y"),
                "Min Forecast":  round(result["forecasts"]["Min"]["prices"][i], 0),
                "Min Low (CI)":  round(result["forecasts"]["Min"]["low"][i], 0),
                "Min High (CI)": round(result["forecasts"]["Min"]["high"][i], 0),
                "Max Forecast":  round(result["forecasts"]["Max"]["prices"][i], 0),
                "Max Low (CI)":  round(result["forecasts"]["Max"]["low"][i], 0),
                "Max High (CI)": round(result["forecasts"]["Max"]["high"][i], 0),
                "Modal Forecast":  round(result["forecasts"]["Modal"]["prices"][i], 0),
                "Modal Low (CI)":  round(result["forecasts"]["Modal"]["low"][i], 0),
                "Modal High (CI)": round(result["forecasts"]["Modal"]["high"][i], 0),
            })
        fc_df = pd.DataFrame(rows)

        st.plotly_chart(fig_forecast_table(result), use_container_width=True, key="forecast_table")

        # CSV download
        csv = fc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Forecast CSV",
            data=csv,
            file_name=f"{market}_30day_forecast.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
