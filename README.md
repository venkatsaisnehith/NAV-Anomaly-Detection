# NAV Valuation Anomaly & Fraud Detection Engine

## The Problem

Luxembourg administers over **€5.8 trillion** in fund assets, making it Europe's largest and the world's second-largest fund domicile. Transfer agents and fund administrators process thousands of Net Asset Value (NAV) calculations daily. A single pricing error, a stale data feed, or an unreconciled cash entry can misstate fund valuations by millions of euros, triggering regulatory sanctions under CSSF Circular 18/698 and investor compensation claims.

Traditional rule-based checks (static tolerance bands, manual reconciliation) fail to catch anomalies that fall within normal thresholds individually but are statistically abnormal in combination. This engine addresses that gap.

## What This Project Does

This is a **production-grade unsupervised machine learning pipeline** that:

1. **Simulates** realistic daily fund operations — asset valuations, investor cash flows, fee accruals, and NAV calculations — using stochastic financial models.
2. **Injects** three categories of real-world operational errors into the simulation to create labeled ground truth for evaluation.
3. **Engineers** domain-specific financial features designed to expose valuation discrepancies that raw data alone cannot reveal.
4. **Trains** two complementary anomaly detection models that work in parallel to maximize detection coverage.
5. **Explains** every flagged anomaly using game-theory attribution (SHAP), producing structured audit logs that compliance officers can review without needing ML expertise.

---

## Pipeline Results

The engine was evaluated against 400 simulated trading days with a 5% anomaly injection rate:

| Metric | Value |
|---|---|
| **Precision** | 66.67% |
| **Recall (Capture Rate)** | 80.00% |
| **F1-Score** | 72.73% |

**Detection rates by anomaly category:**

| Anomaly Type | Description | Capture Rate |
|---|---|---|
| Fat-Finger Errors | Decimal shifts or data entry typos in NAV or asset prices | **100.0%** (4/4) |
| Cash Reconciliation Breaks | Missing subscription cash not deposited into fund accounts | **91.7%** (11/12) |
| Stale Pricing | Frozen price feeds while markets continue to move | **25.0%** (1/4) |

The ensemble approach (Isolation Forest + Autoencoder voting) captures every fat-finger error and nearly all cash mismatches. Stale pricing remains the hardest category because the price deviation from a single frozen day can be within normal volatility bands — a known challenge in real fund administration.

---

## Technical Architecture

### Fund Simulator (`data_generator.py`)

Generates realistic daily time-series for a €100M Luxembourg-style investment fund:

- **Asset pricing** follows Geometric Brownian Motion (GBM) with asset-specific drift and volatility parameters calibrated to equity-like returns (4–12% annualized return, 10–25% annualized volatility).
- **Investor cash flows** (subscriptions and redemptions) are modeled using Log-Normal distributions to reflect the skewed, non-negative nature of real fund flows.
- **Portfolio rebalancing** maintains target asset weights through daily end-of-day trades, with a 2% cash buffer reserve.
- **Management fees** are accrued daily at a 1.5% annual rate (divided by 252 trading days).
- **Three anomaly types** are injected at controlled indices:
  - *Fat-Finger:* Asset price multiplied by 5x, simulating a keying error.
  - *Stale Pricing:* Previous day's asset prices carried forward while markets move.
  - *Cash Mismatch:* 90% of subscription cash fails to appear in the cash buffer.

### Feature Engineering (`features.py`)

Transforms raw operational data into 10 normalized features optimized for outlier detection:

| Feature | What It Captures |
|---|---|
| `NAV_Return` | Daily percentage change in calculated NAV |
| `Tracking_Residual` | Divergence between NAV movement and underlying asset returns — the single most powerful anomaly signal |
| `NAV_Vol_5d` / `NAV_Vol_20d` | Short-term and medium-term rolling volatility of NAV returns |
| `Cash_Ratio` | Cash buffer as a proportion of total AUM |
| `Cash_Ratio_Deviation` | Deviation from the expected 2% cash buffer target |
| `Sub_Ratio` / `Red_Ratio` | Subscriptions and redemptions scaled by AUM |
| `Net_Flow_Ratio` | Net investor cash flow direction |
| `NAV_ZScore_20d` | Rolling 20-day Z-score of NAV price — flags statistically extreme valuations |

All features are standardized using `StandardScaler` (zero mean, unit variance) before model training.

### Anomaly Detection Models (`models.py`)

Two models operate in parallel. A day is flagged if **either** model identifies it as anomalous (high-sensitivity ensemble):

**Model A — Isolation Forest (Scikit-Learn)**
- Ensemble of 100 isolation trees
- Contamination parameter set to match the expected anomaly rate
- Isolates outliers based on shorter average path lengths in random partitioning
- Strengths: Fast training, handles non-linear feature interactions, no assumption about data distribution

**Model B — Deep Autoencoder (PyTorch)**
- Architecture: `Input(10) → Dense(8, ReLU) → Latent(4, ReLU) → Dense(8, ReLU) → Output(10)`
- Trained for 40 epochs with Adam optimizer (learning rate: 0.001, batch size: 16)
- Loss function: Mean Squared Error (MSE) between input and reconstruction
- Anomaly threshold: Set at the 95th percentile of training reconstruction losses
- Strengths: Learns compressed representations of "normal" fund behavior; anomalies produce high reconstruction error

### Explainability Engine (`explainability.py`)

Every flagged anomaly is accompanied by a structured SHAP explanation:

- Uses `shap.Explainer` with a background sample of 50 representative trading days
- For Isolation Forest: explains the negated decision function (positive SHAP = increased anomaly likelihood)
- For Autoencoder: explains the reconstruction loss directly
- Produces human-readable narratives, e.g.: *"'Cash_Ratio' value of -2.15 decreased the anomaly score (impact: -0.062)"*
- Top 3 contributing features are extracted and ranked by absolute SHAP magnitude

### REST API (`api.py`)

FastAPI service with automatic model training on startup:

| Endpoint | Method | Purpose |
|---|---|---|
| `/train` | POST | Retrain models on a fresh simulated dataset |
| `/data` | GET | Fetch the full simulated fund time-series for dashboard visualization |
| `/validate` | POST | Submit a single day's NAV record and receive anomaly flag + SHAP explanation |

### Surveillance Dashboard (`templates/dashboard.html`)

Interactive dark-mode glassmorphism web interface for risk managers:
- Historical fund performance visualization
- Real-time anomaly alert timeline
- SHAP feature impact breakdown for each flagged day

---

## Project Structure

```
├── src/
│   ├── api.py                 # FastAPI endpoints (/train, /data, /validate)
│   ├── data_generator.py      # GBM fund simulation & anomaly injection
│   ├── features.py            # Financial feature engineering pipeline
│   ├── models.py              # Isolation Forest & PyTorch Autoencoder
│   └── explainability.py      # SHAP-based audit trail generation
├── templates/
│   └── dashboard.html         # Surveillance dashboard UI
├── run_pipeline.py            # End-to-end pipeline orchestrator
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## Setup & Local Execution

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone and navigate
git clone https://github.com/venkatsaisnehith/NAV-Anomaly-Detection.git
cd NAV-Anomaly-Detection

# 2. Create virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (simulate → engineer → train → evaluate → explain)
python run_pipeline.py

# 5. Launch the REST API
cd src
uvicorn api:app --port 8000 --reload

# 6. Open the dashboard
# Navigate to templates/dashboard.html in your browser
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation & Data | NumPy, Pandas |
| Classical ML | Scikit-Learn (Isolation Forest, StandardScaler) |
| Deep Learning | PyTorch (Autoencoder, Adam, MSELoss) |
| Explainability | SHAP (Shapley Additive Explanations) |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript |
