# 🌿 Pepper Price Forecasting Dashboard

A production-ready Streamlit application for forecasting **Min, Max, and Modal pepper prices**
across Karnataka APMC markets — with district/market dropdowns, EDA charts,
model validation diagnostics, and a 30-day recursive ensemble forecast.

---

## 📁 Project Structure

```
pepper_app/
├── app.py               ← Main Streamlit application (entry point)
├── model.py             ← ML pipeline: data prep, feature engineering, training, forecasting
├── charts.py            ← All Plotly chart builders
├── requirements.txt     ← Python dependencies
├── data/
│   └── final.csv        ← Karnataka APMC transaction data (2019–2025)
└── .streamlit/
    └── config.toml      ← Streamlit theme and server config
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or 3.11  *(3.12 also works)*
- pip

### Step 1 — Clone / download the project folder

Place the entire `pepper_app/` folder somewhere on your machine.

### Step 2 — Create a virtual environment *(recommended)*

```bash
# Navigate into the project folder
cd pepper_app

# Create virtual environment
python -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate
# Windows (CMD):
venv\Scripts\activate.bat
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.35.0 | Web UI framework |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.5.0 | ML models |
| plotly | 5.22.0 | Interactive charts |

### Step 4 — Verify data file

Ensure `data/final.csv` is present in the project folder.
The file should contain the Karnataka APMC pepper dataset (10,530 rows, 20 columns).

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** in your browser.

---

## 🖥️ Using the App

### Sidebar Controls
1. **Select District** — Choose from 13 Karnataka districts (or "All Districts")
2. **Select Market** — Dropdown filters to show only markets in the selected district
3. **Market Info card** — Shows record count, date range, and latest prices at a glance

### Main Dashboard Tabs

| Tab | What You See |
|-----|-------------|
| 🔮 **Forecast** | 30-day Min/Max/Modal forecast with CI ribbons + price spread + seasonal context |
| 📈 **Historical** | Full historical price bands, year-on-year trend, arrivals vs price, decomposition |
| 🧭 **EDA** | Monthly seasonality, weather correlations, descriptive stats, raw data preview |
| ✅ **Validation** | Actual vs predicted on held-out test set with MAPE/MAE/RMSE per target |
| 📊 **Residuals** | Residual time series + distribution histograms for all three targets |
| 🔑 **Features** | GBM feature importance bars + full feature engineering reference table |
| 📋 **Forecast Table** | Full 30-row interactive table with CI bounds + CSV download button |

### KPI Strip
Below the header, 9 metric cards show **Day 1 price, Day 30 price, and 30-day trend %**
for each of Min, Max, and Modal. An accuracy badge (green/amber/red) shows overall model quality.

---

## 🤖 Model Architecture

```
For each (market, target_variable):
  ├── GradientBoostingRegressor  (weight 45%)
  ├── RandomForestRegressor      (weight 40%)
  └── Ridge Regression           (weight 15%)

Features (23 per target):
  ├── Calendar:  dayofyear, month, weekofyear, year
  ├── Cyclical:  sin_doy, cos_doy
  ├── Lags:      lag_1, lag_3, lag_7, lag_14, lag_21, lag_30, lag_365
  ├── Rolling:   roll_mean & roll_std at 7/14/30-day windows
  └── Weather:   temperature, rainfall, soil_moisture, soil_temperature

Forecast strategy: Recursive (one-step-ahead repeated 30×)
Confidence intervals: ±1.5σ of test-set residuals
```

---

## 📊 Supported Markets

22 markets have sufficient data for reliable forecasting:

**High accuracy (MAPE < 5%):** SIRSI, SAGAR, KARKALA, SRINGERI, BENGALURU,
CHANNAGIRI, SULYA, BANTWALA, YELLAPURA, PUTTUR

**Moderate accuracy (MAPE 5–10%):** SIDDAPURA, MUDIGERE, BELUR, BELTHANGADI,
MANGALURU, CHIKKAMAGALURU, KUNDAPUR

**Data-limited (MAPE > 10%):** SAKLESHPUR, GONIKOPPAL, SOMWARPET, KOPPA, MADIKERI

Markets with < 30 usable rows show a warning and are excluded from modelling.

---

## ⚡ Performance Notes

- **First load** for any market takes ~15–45 seconds (model training).
- **Subsequent selections** of the same market are instant — results are cached.
- Cache persists for the lifetime of the Streamlit session.
- To force re-training, press **C** (clear cache) in the browser or restart the app.

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: data/final.csv` | Ensure `data/final.csv` exists and you're running from `pepper_app/` |
| App opens but shows blank | Wait ~30s for the first model to train; check terminal for errors |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Slow on older machines | Reduce `n_estimators` in `model.py` line ~60 from 300 → 100 |

---

## 📦 Packaging for Sharing

To share with colleagues without them installing Python:

### Option A — Streamlit Community Cloud (free hosting)
1. Push the `pepper_app/` folder to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Click **Deploy** — live URL in ~2 minutes

### Option B — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
```bash
docker build -t pepper-forecast .
docker run -p 8501:8501 pepper-forecast
```

---

*Karnataka APMC pepper Price Intelligence · 2025*
