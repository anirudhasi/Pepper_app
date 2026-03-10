"""
charts.py — All Plotly chart builders for the Pepper Forecasting App.
NOTE: add_vline() is NEVER used anywhere in this file because it crashes
      on date-based x-axes in Plotly with pandas 2.x.
      Vertical lines are drawn via update_layout(shapes=[...]) instead.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "Min":    "#4fc3f7",
    "Max":    "#ef5350",
    "Modal":  "#66bb6a",
    "bg":     "#0f1117",
    "panel":  "#1a1d27",
    "grid":   "#2a2d3a",
    "text":   "#e0e0e0",
    "accent": "#ffd54f",
    "white":  "#ffffff",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["panel"],
    font=dict(color=C["text"], family="Arial, sans-serif"),
    xaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False),
    legend=dict(bgcolor=C["panel"], bordercolor=C["grid"], borderwidth=1),
    margin=dict(l=50, r=30, t=60, b=50),
)


def _apply_layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=C["text"])),
        height=height,
        **LAYOUT_DEFAULTS,
    )
    return fig


def _vline_shape(x_str, color=None):
    """Return a shapes-dict vertical line for update_layout(shapes=[...]).
    Uses xref='x' so it tracks the data axis — NO add_vline() call."""
    return dict(
        type="line",
        x0=x_str, x1=x_str,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=color or C["accent"], width=1.5, dash="dot"),
    )


def _vline_annotation(x_str, text, color=None):
    """Return an annotation dict to accompany a vertical line."""
    return dict(
        x=x_str, y=1,
        xref="x", yref="paper",
        text=text, showarrow=False,
        font=dict(color=color or C["accent"], size=11),
        xanchor="left", yanchor="top",
        bgcolor=C["panel"], bordercolor=color or C["accent"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Historical Price Bands
# ─────────────────────────────────────────────────────────────────────────────

def fig_historical_bands(mdf: pd.DataFrame, market: str) -> go.Figure:
    fig = go.Figure()
    for col, dash in [("Min", "dot"), ("Max", "dash"), ("Modal", "solid")]:
        fig.add_trace(go.Scatter(
            x=mdf.index, y=mdf[col],
            name=col, line=dict(color=C[col], width=1.2, dash=dash),
            opacity=0.85,
            hovertemplate=f"{col}: \u20b9%{{y:,.0f}}<extra></extra>",
        ))
        ma30 = mdf[col].rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=mdf.index, y=ma30,
            name=f"{col} 30d MA",
            line=dict(color=C[col], width=2, dash="longdash"),
            opacity=0.5, showlegend=False,
            hovertemplate=f"{col} 30d MA: \u20b9%{{y:,.0f}}<extra></extra>",
        ))
    return _apply_layout(fig, f"\U0001f4c8 Historical Price Bands \u2014 {market} (2019\u20132025)", height=440)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 30-Day Forecast
# ─────────────────────────────────────────────────────────────────────────────

def fig_forecast(result: dict, market: str) -> go.Figure:
    fig = go.Figure()
    mdf = result["mdf"]
    hist_tail = mdf.iloc[-90:]

    for col in ["Min", "Max", "Modal"]:
        fc = result["forecasts"][col]

        fig.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail[col],
            name=f"{col} (hist)", line=dict(color=C[col], width=1.3),
            opacity=0.7,
            hovertemplate=f"{col}: \u20b9%{{y:,.0f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=list(fc["dates"]) + list(fc["dates"])[::-1],
            y=list(fc["high"]) + list(fc["low"])[::-1],
            fill="toself", fillcolor=C[col],
            opacity=0.12, line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["dates"], y=fc["prices"],
            name=f"{col} Forecast",
            line=dict(color=C[col], width=2.2),
            mode="lines+markers",
            marker=dict(size=4, color=C[col]),
            hovertemplate=f"{col} Forecast: \u20b9%{{y:,.0f}}<extra></extra>",
        ))

    # ── Vertical line via shapes — avoids add_vline bug entirely ──────────────
    last_str = str(result["last_date"].date())
    fig.update_layout(
        shapes=[_vline_shape(last_str)],
        annotations=[_vline_annotation(last_str, "Forecast Start")],
    )
    return _apply_layout(fig, f"\U0001f52e 30-Day Price Forecast \u2014 {market}", height=480)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Forecast Table
# ─────────────────────────────────────────────────────────────────────────────

def fig_forecast_table(result: dict) -> go.Figure:
    dates = result["forecasts"]["Modal"]["dates"]
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "Day":            i + 1,
            "Date":           d.strftime("%d %b %Y"),
            "Min (\u20b9)":       f"\u20b9{result['forecasts']['Min']['prices'][i]:,.0f}",
            "Min Low":        f"\u20b9{result['forecasts']['Min']['low'][i]:,.0f}",
            "Max (\u20b9)":       f"\u20b9{result['forecasts']['Max']['prices'][i]:,.0f}",
            "Max High":       f"\u20b9{result['forecasts']['Max']['high'][i]:,.0f}",
            "Modal (\u20b9)":     f"\u20b9{result['forecasts']['Modal']['prices'][i]:,.0f}",
            "Modal Low":      f"\u20b9{result['forecasts']['Modal']['low'][i]:,.0f}",
            "Modal High":     f"\u20b9{result['forecasts']['Modal']['high'][i]:,.0f}",
        })
    df = pd.DataFrame(rows)

    fig = go.Figure(data=[go.Table(
        columnwidth=[50, 90, 100, 100, 100, 100, 100, 100, 100],
        header=dict(
            values=list(df.columns),
            fill_color="#1F3864",
            font=dict(color="white", size=11),
            align="center", height=32,
        ),
        cells=dict(
            values=[df[c].tolist() for c in df.columns],
            fill_color=[["#1a1d27" if i % 2 == 0 else "#23263a" for i in range(len(df))]],
            font=dict(color=C["text"], size=10),
            align="center", height=26,
        ),
    )])
    fig.update_layout(
        paper_bgcolor=C["bg"],
        margin=dict(l=0, r=0, t=10, b=0),
        height=850,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Validation
# ─────────────────────────────────────────────────────────────────────────────

def fig_validation(result: dict, market: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Min Price", "Max Price", "Modal Price"],
                        vertical_spacing=0.08)
    test = result["test"]
    for i, col in enumerate(["Min", "Max", "Modal"], 1):
        y_true = test[col].values
        y_pred = result["test_preds"][col]
        m = result["metrics"][col]
        fig.add_trace(go.Scatter(
            x=test.index, y=y_true, name=f"{col} Actual",
            line=dict(color=C[col], width=1.8),
            hovertemplate=f"Actual: \u20b9%{{y:,.0f}}<extra></extra>",
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=test.index, y=y_pred, name=f"{col} Predicted",
            line=dict(color=C["white"], width=1.4, dash="dash"),
            hovertemplate=f"Predicted: \u20b9%{{y:,.0f}}<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(title_text="\u20b9/quintal", gridcolor=C["grid"], row=i, col=1)
        ann_text = f"MAE: \u20b9{m['MAE']:,.0f} | RMSE: \u20b9{m['RMSE']:,.0f} | MAPE: {m['MAPE']:.2f}%"
        yref = "y" if i == 1 else f"y{i}"
        fig.add_annotation(
            xref="paper", yref=yref,
            x=0.01, y=float(y_true.max()),
            text=ann_text, showarrow=False,
            font=dict(size=10, color=C["accent"]),
            bgcolor=C["panel"], bordercolor=C["grid"],
        )
    fig.update_layout(
        title=dict(text=f"\u2705 Model Validation \u2014 {market} (last {len(test)} days)",
                   font=dict(size=14, color=C["text"])),
        height=620,
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        legend=dict(bgcolor=C["panel"], bordercolor=C["grid"]),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=C["grid"], showgrid=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Residual Analysis
# ─────────────────────────────────────────────────────────────────────────────

def fig_residuals(result: dict, market: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Min Residuals", "Max Residuals", "Modal Residuals",
                        "Min Distribution", "Max Distribution", "Modal Distribution"],
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )
    test = result["test"]
    for j, col in enumerate(["Min", "Max", "Modal"], 1):
        resid = test[col].values - result["test_preds"][col]
        resid_mean = float(np.mean(resid))

        # Time series
        fig.add_trace(go.Scatter(
            x=test.index, y=resid, name=col,
            line=dict(color=C[col], width=1.2), showlegend=False,
        ), row=1, col=j)

        # Zero line via shape (numeric y-axis, add_hline is safe here but use shape for consistency)
        fig.add_trace(go.Scatter(
            x=[test.index[0], test.index[-1]], y=[0, 0],
            mode="lines", line=dict(color=C["white"], width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=j)

        # Histogram
        fig.add_trace(go.Histogram(
            x=resid, name=col,
            marker_color=C[col], opacity=0.8,
            showlegend=False, nbinsx=20,
        ), row=2, col=j)

        # Mean line on histogram — numeric axis, safe to use a Scatter
        fig.add_trace(go.Scatter(
            x=[resid_mean, resid_mean], y=[0, 1],
            mode="lines",
            line=dict(color=C["accent"], width=1.5, dash="dash"),
            showlegend=False, hoverinfo="skip",
            yaxis=f"y{3 + j}",   # histogram yaxis
        ), row=2, col=j)

    fig.update_layout(
        title=dict(text=f"\U0001f4ca Residual Analysis \u2014 {market}",
                   font=dict(size=14, color=C["text"])),
        height=500,
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        margin=dict(l=50, r=20, t=70, b=40),
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_importance(result: dict, market: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Min Price", "Max Price", "Modal Price"],
                        horizontal_spacing=0.08)
    for j, col in enumerate(["Min", "Max", "Modal"], 1):
        fi = result["fi"][col].head(15)
        short = [f.replace(f"{col}_", "").replace("roll_", "r").replace("lag_", "l")
                 for f in fi.index]
        fig.add_trace(go.Bar(
            x=fi.values[::-1], y=short[::-1],
            orientation="h", name=col,
            marker=dict(color=C[col], opacity=0.85),
            showlegend=False,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ), row=1, col=j)
    fig.update_layout(
        title=dict(text=f"\U0001f511 Feature Importances (GBM) \u2014 {market}",
                   font=dict(size=14, color=C["text"])),
        height=460,
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        margin=dict(l=80, r=20, t=70, b=40),
    )
    fig.update_xaxes(gridcolor=C["grid"], title_text="Importance")
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Monthly Seasonality
# ─────────────────────────────────────────────────────────────────────────────

def fig_seasonality(mdf: pd.DataFrame, market: str) -> go.Figure:
    mdf2 = mdf.copy()
    mdf2["month"] = mdf2.index.month
    monthly = mdf2.groupby("month")[["Min", "Max", "Modal"]].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x - 1])
    fig = go.Figure()
    for col in ["Min", "Max", "Modal"]:
        fig.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly[col],
            name=col, marker_color=C[col], opacity=0.85,
            hovertemplate=f"{col}: \u20b9%{{y:,.0f}}<extra></extra>",
        ))
    return _apply_layout(fig, f"\U0001f4c5 Monthly Price Seasonality \u2014 {market}", height=400)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Year-on-Year Trend
# ─────────────────────────────────────────────────────────────────────────────

def fig_yearly_trend(mdf: pd.DataFrame, market: str) -> go.Figure:
    mdf2 = mdf.copy()
    mdf2["year"] = mdf2.index.year
    yearly = mdf2.groupby("year")[["Min", "Max", "Modal"]].mean().reset_index()
    fig = go.Figure()
    for col in ["Min", "Max", "Modal"]:
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly[col],
            name=col, line=dict(color=C[col], width=2.5),
            mode="lines+markers", marker=dict(size=7),
            hovertemplate=f"{col}: \u20b9%{{y:,.0f}}<extra></extra>",
        ))
    return _apply_layout(fig, f"\U0001f4c6 Year-on-Year Average Price \u2014 {market}", height=380)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Price Spread (Max − Min)
# ─────────────────────────────────────────────────────────────────────────────

def fig_price_spread(mdf: pd.DataFrame, result: dict, market: str) -> go.Figure:
    hist_spread = (mdf["Max"] - mdf["Min"]).iloc[-180:]
    fc_spread   = result["forecasts"]["Max"]["prices"] - result["forecasts"]["Min"]["prices"]
    fc_dates    = result["forecasts"]["Max"]["dates"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_spread.index, y=hist_spread.values,
        name="Historical Spread", line=dict(color="#ce93d8", width=1.2), opacity=0.85,
        hovertemplate="Spread: \u20b9%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_spread,
        name="Forecast Spread", line=dict(color=C["accent"], width=2.2),
        mode="lines+markers", marker=dict(size=4),
        hovertemplate="Forecast Spread: \u20b9%{y:,.0f}<extra></extra>",
    ))
    # Historical average line — draw as a Scatter, NOT add_hline
    avg_val = float(hist_spread.mean())
    all_x = list(hist_spread.index) + list(fc_dates)
    fig.add_trace(go.Scatter(
        x=[all_x[0], all_x[-1]], y=[avg_val, avg_val],
        mode="lines", name=f"Hist avg \u20b9{avg_val:,.0f}",
        line=dict(color=C["white"], width=1, dash="dash"),
        opacity=0.4, hoverinfo="skip",
    ))
    # Vertical line at forecast start via shapes
    last_str = str(result["last_date"].date())
    fig.update_layout(shapes=[_vline_shape(last_str)])
    return _apply_layout(fig, f"\U0001f4c9 Price Spread (Max\u2212Min) \u2014 {market}", height=360)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Arrivals vs Price
# ─────────────────────────────────────────────────────────────────────────────

def fig_arrivals_vs_price(mdf: pd.DataFrame, market: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=mdf.index, y=mdf["Arrivals"],
        name="Arrivals (q)", marker_color="#7986cb", opacity=0.5,
        hovertemplate="Arrivals: %{y:,.0f} q<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=mdf.index, y=mdf["Modal"],
        name="Modal Price", line=dict(color=C["Modal"], width=1.5),
        hovertemplate="Modal: \u20b9%{y:,.0f}<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text=f"\U0001f4e6 Arrivals vs Modal Price \u2014 {market}",
                   font=dict(size=14, color=C["text"])),
        height=400,
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        legend=dict(bgcolor=C["panel"], bordercolor=C["grid"]),
        margin=dict(l=50, r=60, t=60, b=50),
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(title_text="Arrivals (quintals)", gridcolor=C["grid"], secondary_y=False)
    fig.update_yaxes(title_text="Modal Price (\u20b9)", gridcolor=C["grid"], secondary_y=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. Weather Correlation
# ─────────────────────────────────────────────────────────────────────────────

def fig_weather_correlation(mdf: pd.DataFrame, market: str) -> go.Figure:
    weather_cols = ["avg_rainfall", "temperature_2m_avg",
                    "soil_moisture_0_to_7cm_avg", "soil_temperature_0_to_7cm_avg"]
    labels = ["Rainfall (mm)", "Temperature (\u00b0C)", "Soil Moisture", "Soil Temp (\u00b0C)"]
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.07,
                        subplot_titles=labels)
    for j, (wc, lbl) in enumerate(zip(weather_cols, labels), 1):
        valid = mdf[[wc, "Modal"]].dropna()
        fig.add_trace(go.Scatter(
            x=valid[wc], y=valid["Modal"],
            mode="markers",
            marker=dict(color=C["Modal"], size=3, opacity=0.35),
            name=lbl, showlegend=False,
            hovertemplate=f"{lbl}: %{{x:.2f}}<br>Modal: \u20b9%{{y:,.0f}}<extra></extra>",
        ), row=1, col=j)
        try:
            m_coef, b = np.polyfit(valid[wc], valid["Modal"], 1)
            xr = np.linspace(float(valid[wc].min()), float(valid[wc].max()), 50)
            r  = float(np.corrcoef(valid[wc], valid["Modal"])[0, 1])
            fig.add_trace(go.Scatter(
                x=xr, y=m_coef * xr + b,
                line=dict(color=C["accent"], width=1.5, dash="dash"),
                showlegend=False,
                hovertemplate=f"Trend r={r:.2f}<extra></extra>",
            ), row=1, col=j)
            fig.update_xaxes(title_text=lbl, gridcolor=C["grid"], row=1, col=j)
        except Exception:
            pass
    fig.update_layout(
        title=dict(text=f"\U0001f326 Weather vs Modal Price Correlation \u2014 {market}",
                   font=dict(size=14, color=C["text"])),
        height=360,
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        margin=dict(l=50, r=20, t=70, b=50),
    )
    fig.update_yaxes(title_text="Modal \u20b9", gridcolor=C["grid"], col=1)
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. Price Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def fig_decomposition(mdf: pd.DataFrame, market: str) -> go.Figure:
    trend    = mdf["Modal"].rolling(365, center=True).mean()
    residual = mdf["Modal"] - trend
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=["Observed + Trend", "Long-Term Trend",
                                        "Seasonal / Residual"],
                        shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=mdf.index, y=mdf["Modal"], name="Observed",
                             line=dict(color=C["Modal"], width=0.9), opacity=0.8),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=mdf.index, y=trend, name="365d MA",
                             line=dict(color=C["accent"], width=2, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=mdf.index, y=trend, name="Trend",
                             line=dict(color=C["accent"], width=2), showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=mdf.index, y=residual, name="Residual",
                             line=dict(color="#ce93d8", width=0.8), opacity=0.8),
                  row=3, col=1)
    # Zero line on residual panel as a Scatter trace
    fig.add_trace(go.Scatter(
        x=[mdf.index[0], mdf.index[-1]], y=[0, 0],
        mode="lines", line=dict(color=C["white"], width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.update_layout(
        title=dict(text=f"\U0001f4c9 Price Decomposition \u2014 {market} (Modal)",
                   font=dict(size=14, color=C["text"])),
        height=550,
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        legend=dict(bgcolor=C["panel"], bordercolor=C["grid"]),
        margin=dict(l=50, r=20, t=70, b=40),
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13. Metrics Summary
# ─────────────────────────────────────────────────────────────────────────────

def metrics_summary(result: dict) -> dict:
    out = {}
    for col in ["Min", "Max", "Modal"]:
        m  = result["metrics"][col]
        fc = result["forecasts"][col]
        out[col] = {
            "MAPE":      m["MAPE"],
            "MAE":       m["MAE"],
            "RMSE":      m["RMSE"],
            "Day1":      round(float(fc["prices"][0]), 0),
            "Day30":     round(float(fc["prices"][-1]), 0),
            "Trend_pct": round((float(fc["prices"][-1]) / float(fc["prices"][0]) - 1) * 100, 1),
        }
    return out
