"""
model.py — Pepper Price Forecasting Pipeline
Handles data prep, feature engineering, ensemble training, and 30-day recursive forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

TARGETS      = ["Min", "Max", "Modal"]
W_GBM        = 0.45
W_RF         = 0.40
W_RIDGE      = 0.15
CI_SIGMA     = 1.5
RANDOM_STATE = 42
MIN_ROWS     = 30
TEST_DAYS    = 60
FORECAST_DAYS = 30

BASE_FEATURES = [
    "dayofyear", "month", "weekofyear", "year",
    "sin_doy", "cos_doy",
    "temperature_2m_avg", "avg_rainfall",
    "soil_moisture_0_to_7cm_avg", "soil_temperature_0_to_7cm_avg",
]

NUM_COLS = [
    "Min", "Max", "Modal", "Arrivals", "avg_rainfall",
    "temperature_2m_avg", "soil_moisture_0_to_7cm_avg",
    "soil_temperature_0_to_7cm_avg",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_market_data(df: pd.DataFrame, market: str) -> pd.DataFrame | None:
    """Filter to one market, resolve duplicates, reindex daily, interpolate."""
    mdf = df[df["Market"] == market].sort_values("date").reset_index(drop=True)
    if mdf.empty:
        return None
    mdf = mdf.groupby("date").mean(numeric_only=True).reset_index()
    full_idx = pd.date_range(mdf["date"].min(), mdf["date"].max(), freq="D")
    mdf = mdf.set_index("date").reindex(full_idx)
    mdf.index.name = "date"
    mdf[NUM_COLS] = mdf[NUM_COLS].interpolate(method="time")
    return mdf


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(mdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add lag, rolling, calendar, and seasonality features. Return (enriched_df, feature_sets)."""
    mdf = mdf.copy()
    mdf["dayofyear"]  = mdf.index.dayofyear
    mdf["month"]      = mdf.index.month
    mdf["weekofyear"] = mdf.index.isocalendar().week.astype(int)
    mdf["year"]       = mdf.index.year

    for col in TARGETS:
        for lag in [1, 3, 7, 14, 21, 30]:
            mdf[f"{col}_lag_{lag}"] = mdf[col].shift(lag)
        mdf[f"{col}_lag_365"] = mdf[col].shift(365)
        for w in [7, 14, 30]:
            mdf[f"{col}_roll_mean_{w}"] = mdf[col].shift(1).rolling(w).mean()
            mdf[f"{col}_roll_std_{w}"]  = mdf[col].shift(1).rolling(w).std()

    mdf["sin_doy"] = np.sin(2 * np.pi * mdf["dayofyear"] / 365)
    mdf["cos_doy"] = np.cos(2 * np.pi * mdf["dayofyear"] / 365)
    mdf.dropna(inplace=True)

    feature_sets = {}
    for col in TARGETS:
        lags  = [f"{col}_lag_{l}" for l in [1, 3, 7, 14, 21, 30]] + [f"{col}_lag_365"]
        rolls = ([f"{col}_roll_mean_{w}" for w in [7, 14, 30]] +
                 [f"{col}_roll_std_{w}"  for w in [7, 14, 30]])
        feature_sets[col] = BASE_FEATURES + lags + rolls

    return mdf, feature_sets


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(X_tr, y_tr):
    """Fit GBM + RF + Ridge; return tuple of three fitted models."""
    n = len(X_tr)
    n_est = 300 if n > 500 else (200 if n > 200 else 100)
    lr     = 0.05 if n > 500 else 0.10

    gbm = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=4, learning_rate=lr,
        subsample=0.8, random_state=RANDOM_STATE)
    rf  = RandomForestRegressor(
        n_estimators=n_est, max_depth=8,
        random_state=RANDOM_STATE, n_jobs=-1)
    rid = Ridge(alpha=10)

    gbm.fit(X_tr, y_tr)
    rf.fit(X_tr,  y_tr)
    rid.fit(X_tr, y_tr)
    return gbm, rf, rid


def ensemble_predict(models, X):
    gbm, rf, rid = models
    return W_GBM * gbm.predict(X) + W_RF * rf.predict(X) + W_RIDGE * rid.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# Recursive forecast
# ─────────────────────────────────────────────────────────────────────────────

def recursive_forecast(target_col, models, scaler, feature_list, base_series, n_days=30):
    """One-step-ahead recursive forecast for `n_days`."""
    simulated = base_series.copy()
    last_date  = simulated.index[-1]
    fc_dates   = pd.date_range(last_date + pd.Timedelta("1D"), periods=n_days, freq="D")
    fc_prices  = []

    for fd in fc_dates:
        ser = simulated[target_col]
        row = {
            "dayofyear":  fd.dayofyear,
            "month":      fd.month,
            "weekofyear": fd.isocalendar()[1],
            "year":       fd.year,
            "sin_doy":    np.sin(2 * np.pi * fd.dayofyear / 365),
            "cos_doy":    np.cos(2 * np.pi * fd.dayofyear / 365),
            "temperature_2m_avg":            simulated["temperature_2m_avg"].iloc[-7:].mean(),
            "avg_rainfall":                  simulated["avg_rainfall"].iloc[-30:].mean(),
            "soil_moisture_0_to_7cm_avg":    simulated["soil_moisture_0_to_7cm_avg"].iloc[-7:].mean(),
            "soil_temperature_0_to_7cm_avg": simulated["soil_temperature_0_to_7cm_avg"].iloc[-7:].mean(),
            f"{target_col}_lag_1":   ser.iloc[-1],
            f"{target_col}_lag_3":   ser.iloc[-3],
            f"{target_col}_lag_7":   ser.iloc[-7],
            f"{target_col}_lag_14":  ser.iloc[-14],
            f"{target_col}_lag_21":  ser.iloc[-21],
            f"{target_col}_lag_30":  ser.iloc[-30],
            f"{target_col}_lag_365": ser.iloc[-365] if len(ser) >= 365 else ser.mean(),
            f"{target_col}_roll_mean_7":  ser.iloc[-7:].mean(),
            f"{target_col}_roll_std_7":   ser.iloc[-7:].std(),
            f"{target_col}_roll_mean_14": ser.iloc[-14:].mean(),
            f"{target_col}_roll_std_14":  ser.iloc[-14:].std(),
            f"{target_col}_roll_mean_30": ser.iloc[-30:].mean(),
            f"{target_col}_roll_std_30":  ser.iloc[-30:].std(),
        }
        X_new = scaler.transform(pd.DataFrame([row])[feature_list])
        p = ensemble_predict(models, X_new)[0]
        fc_prices.append(p)

        new_full = {c: np.nan for c in simulated.columns}
        new_full.update(row)
        new_full[target_col] = p
        simulated = pd.concat([simulated, pd.DataFrame(new_full, index=[fd])])

    return fc_dates, np.array(fc_prices)


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, market: str) -> dict | None:
    """
    Full pipeline for one market: prepare → features → split → train → eval → forecast.
    Returns a result dict or None if the market has insufficient data.
    """
    mdf = prepare_market_data(df, market)
    if mdf is None:
        return None

    mdf, feature_sets = build_features(mdf)
    if len(mdf) < MIN_ROWS:
        return {"status": "insufficient_data", "rows": len(mdf), "market": market}

    test_n = min(TEST_DAYS, len(mdf) // 4)
    cutoff = mdf.index[-test_n]
    train  = mdf[mdf.index <= cutoff]
    test   = mdf[mdf.index >  cutoff]

    if len(train) < 10 or len(test) < 5:
        return {"status": "insufficient_data", "rows": len(mdf), "market": market}

    models_dict  = {}
    scalers_dict = {}
    metrics_dict = {}
    test_preds   = {}
    resid_stds   = {}
    forecasts    = {}
    fi_dict      = {}

    for col in TARGETS:
        sc     = StandardScaler()
        X_tr   = sc.fit_transform(train[feature_sets[col]])
        X_te   = sc.transform(test[feature_sets[col]])
        scalers_dict[col] = sc

        mdl = train_ensemble(X_tr, train[col])
        models_dict[col] = mdl

        y_pred = ensemble_predict(mdl, X_te)
        y_true = test[col].values
        test_preds[col] = y_pred

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        resid_std = np.std(y_true - y_pred)
        resid_stds[col] = resid_std

        metrics_dict[col] = {
            "MAE":  round(mae, 0),
            "RMSE": round(rmse, 0),
            "MAPE": round(mape, 2),
        }

        # Feature importances from GBM
        gbm = mdl[0]
        fi_dict[col] = pd.Series(
            gbm.feature_importances_, index=feature_sets[col]
        ).sort_values(ascending=False)

        # 30-day forecast
        fc_dates, fc_prices = recursive_forecast(
            col, mdl, sc, feature_sets[col], mdf.copy(), FORECAST_DAYS
        )
        ci = CI_SIGMA * resid_std
        forecasts[col] = {
            "dates":  fc_dates,
            "prices": fc_prices,
            "low":    fc_prices - ci,
            "high":   fc_prices + ci,
        }

    return {
        "status":      "ok",
        "market":      market,
        "mdf":         mdf,
        "train":       train,
        "test":        test,
        "test_preds":  test_preds,
        "metrics":     metrics_dict,
        "forecasts":   forecasts,
        "fi":          fi_dict,
        "last_date":   mdf.index[-1],
        "feature_sets": feature_sets,
    }
