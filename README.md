# NAV Valuation Anomaly & Fraud Detection Engine

An unsupervised machine learning pipeline and compliance surveillance dashboard designed for Luxembourg’s fund administration ecosystem. The engine simulates daily fund operations (AUM tracking, share subscriptions/redemptions, underlying asset returns) and applies a dual-model detection framework to automatically flag and explain valuation anomalies (e.g., stale price feeds, human "fat-finger" data entry errors, cash mismatches).

---

## 🚀 KEY FEATURES

*   **Financial Market & Fund Simulator:** Generates realistic daily fund histories. Asset prices are modeled using Geometric Brownian Motion (GBM), subscription/redemption cash flows are modeled using Log-Normal distributions, and operational anomalies are systematically injected to test detection rates:
    1.  *Fat-Finger Errors:* Arbitrary decimal shifts or typos in calculated NAV or asset entries.
    2.  *Stale Pricing:* Freeze in pricing data feeds for specific assets while market indices shift.
    3.  *Cash Reconciliation Breaks:* Discrepancies between physical bank cash buffers and calculated flows.
*   **Targeted Feature Engineering:** Translates raw operational streams into features optimized for outlier detection:
    *   *Tracking Residuals:* Divergence between physical AUM shifts and weighted asset returns.
    *   *Cash Buffer Ratios:* Unexplained deviations in liquid cash buffers relative to subscriptions/redemptions.
    *   *Volatility Ratios:* Relative changes in asset pricing volatility flags.
*   **Dual Outlier Detection Models:**
    *   *Isolation Forest (Scikit-Learn):* An ensemble of isolation trees that isolates anomalous data points based on path length shortcuts.
    *   *PyTorch Deep Autoencoder:* A neural network trained to compress and reconstruct normal fund patterns. Valuation entries with a reconstruction loss exceeding a dynamic threshold are flagged as anomalous.
*   **SHAP Explainable Compliance Audit Trail:** Integrates game-theory explanations (Shapley values) to break down exactly which features triggered the anomaly flag, generating structured log entries for human-in-the-loop compliance reviews.
*   **Surveillance Web Dashboard:** A premium, interactive dark-mode glassmorphism dashboard that allows risk managers to visualize historical trends, view active anomaly alerts, inspect features, and review automated explanations.

---

## 📁 SYSTEM ARCHITECTURE

```
project2_anomaly_detection/
├── src/
│   ├── __init__.py
│   ├── api.py                 # FastAPI service endpoints (/train, /data, /validate)
│   ├── data_generator.py      # Geometric Brownian Motion simulation & anomaly injection
│   ├── features.py            # Financial feature engineering & scaling pipeline
│   ├── models.py              # Isolation Forest & PyTorch Autoencoder architectures
│   └── explainability.py      # SHAP engine for local feature impact explanations
├── templates/
│   └── dashboard.html         # Asynchronous dashboard UI with visual metrics
├── run_pipeline.py            # Offline validation & pipeline orchestrator script
└── requirements.txt           # Project dependencies
```

---

## 🛠️ INSTALLATION & LOCAL SETUP

### Prerequisites
*   Python 3.8 or higher
*   PIP (Python Package Installer)

### Step-by-Step Execution
1.  **Navigate into the directory & create a Virtual Environment:**
    ```powershell
    cd project2_anomaly_detection
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Run the offline pipeline script:**
    This script runs the entire end-to-end pipeline (simulates fund data, extracts features, trains both ML models, calculates evaluation metrics, and generates SHAP explanations for top anomalies).
    ```powershell
    python run_pipeline.py
    ```
4.  **Launch the live REST API:**
    ```powershell
    cd src
    uvicorn api:app --port 8000 --reload
    ```
5.  **Open the Web Dashboard:**
    Open [dashboard.html](file:///c:/Users/saisn/Documents/finance%20project/project2_anomaly_detection/templates/dashboard.html) directly in any browser to visualize the fund performance, track live alerts, and review SHAP explanation reports!
