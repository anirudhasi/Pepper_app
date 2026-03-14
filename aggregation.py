"""
aggregation.py — District-level price aggregation for Pepper Price Forecasting App.

Strategy: Arrival-weighted average prices per district per day.
  - Markets with higher arrivals on a given day contribute more to the district price.
  - Missing arrivals default to 1 (equal weight) to avoid zero-division.
  - Days where only 1 market traded are still included (single-market day).
  - Weekly resampling is used for correlation analysis to fill cross-market gaps.
"""

import numpy as np
import pandas as pd

TARGETS = ["Min", "Max", "Modal"]
MIN_DISTRICT_RECORDS = 30   # minimum daily rows to consider a district usable


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_district(df: pd.DataFrame, district: str) -> pd.DataFrame | None:
    """
    Compute daily arrival-weighted Min / Max / Modal for one district.

    Returns a DataFrame indexed by date with columns:
        Min, Max, Modal,           ← weighted average prices
        Total_Arrivals,            ← sum of arrivals across active markets
        Markets_Active,            ← number of markets that traded that day
        Min_raw, Max_raw,          ← unweighted (simple) mean for reference
        Modal_raw
    Returns None if district not found or insufficient data.
    """
    sub = df[df["District"] == district].copy()
    if sub.empty:
        return None

    # Safe arrivals: fill NaN with 1 so every market gets at least equal weight
    sub["arr_safe"] = sub["Arrivals"].fillna(1).clip(lower=1)

    def _wavg(g: pd.DataFrame) -> pd.Series:
        w = g["arr_safe"].values
        result = {}
        for col in TARGETS:
            vals = g[col].values
            mask = ~np.isnan(vals)
            if mask.sum() == 0:
                result[col] = np.nan
            else:
                result[col] = np.average(vals[mask], weights=w[mask])
        result["Total_Arrivals"] = g["Arrivals"].sum()
        result["Markets_Active"] = len(g)
        # Raw (unweighted) mean for display reference
        for col in TARGETS:
            result[f"{col}_raw"] = g[col].mean()
        return pd.Series(result)

    daily = sub.groupby("date").apply(_wavg).reset_index()
    daily = daily.sort_values("date").set_index("date")
    daily.index = pd.to_datetime(daily.index)

    if len(daily) < MIN_DISTRICT_RECORDS:
        return None

    return daily


def aggregate_all_districts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict {district_name: daily_df} for all districts with sufficient data."""
    results = {}
    for district in df["District"].dropna().unique():
        agg = aggregate_district(df, district)
        if agg is not None:
            results[district] = agg
    return results


# ─────────────────────────────────────────────────────────────────────────────
# District summary stats
# ─────────────────────────────────────────────────────────────────────────────

def district_summary(df: pd.DataFrame, district: str) -> dict:
    """Return key stats for the district info card."""
    sub = df[df["District"] == district]
    if sub.empty:
        return {}
    markets = sub["Market"].unique().tolist()
    record_counts = sub.groupby("Market").size().sort_values(ascending=False)
    latest = sub.sort_values("date").groupby("Market").last()[["Min", "Max", "Modal"]]
    return {
        "district":      district,
        "markets":       markets,
        "n_markets":     len(markets),
        "total_records": len(sub),
        "date_min":      sub["date"].min(),
        "date_max":      sub["date"].max(),
        "record_counts": record_counts.to_dict(),
        "latest_prices": latest.to_dict(orient="index"),
        "dominant_market": record_counts.index[0],  # most data
    }


# ─────────────────────────────────────────────────────────────────────────────
# Weekly resampled pivot (for correlation — fills cross-market date gaps)
# ─────────────────────────────────────────────────────────────────────────────

def weekly_pivot(df: pd.DataFrame, markets: list[str],
                 price_col: str = "Modal") -> pd.DataFrame:
    """
    Build a weekly-resampled price matrix for a list of markets.
    Each column = one market, each row = one week (Sunday anchor).
    NaN weeks are forward-filled up to 2 periods to handle holiday gaps,
    then dropped if still missing.
    """
    frames = []
    for mkt in markets:
        s = (df[df["Market"] == mkt]
             .set_index("date")[price_col]
             .resample("W").mean()
             .rename(mkt))
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    pivot = pd.concat(frames, axis=1)
    pivot = pivot.ffill(limit=2)       # fill ≤2-week gaps (holiday closures)
    pivot = pivot.dropna(how="all")
    return pivot


def district_weekly_pivot(df: pd.DataFrame, district: str,
                          price_col: str = "Modal") -> pd.DataFrame:
    """Weekly pivot for all markets in a district."""
    markets = df[df["District"] == district]["Market"].unique().tolist()
    return weekly_pivot(df, markets, price_col)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrices
# ─────────────────────────────────────────────────────────────────────────────

def correlation_matrix(pivot: pd.DataFrame,
                       min_periods: int = 20) -> pd.DataFrame:
    """
    Pearson correlation matrix from a weekly pivot.
    Pairs with fewer than min_periods shared observations are set to NaN.
    """
    return pivot.corr(method="pearson", min_periods=min_periods)


def rolling_correlation(pivot: pd.DataFrame,
                        market_a: str, market_b: str,
                        window: int = 26) -> pd.Series:
    """
    26-week (≈6-month) rolling Pearson correlation between two markets.
    Returns a Series indexed by date.
    """
    if market_a not in pivot.columns or market_b not in pivot.columns:
        return pd.Series(dtype=float)
    pair = pivot[[market_a, market_b]].dropna()
    return pair[market_a].rolling(window, min_periods=window // 2).corr(pair[market_b])


def all_market_correlation(df: pd.DataFrame,
                           price_col: str = "Modal",
                           min_records: int = 50) -> pd.DataFrame:
    """
    Full cross-market correlation matrix using all markets with ≥ min_records.
    Uses weekly resampling to handle different market schedules.
    """
    eligible = (df.groupby("Market").size()
                  .loc[lambda s: s >= min_records]
                  .index.tolist())
    pivot = weekly_pivot(df, eligible, price_col)
    return correlation_matrix(pivot)
