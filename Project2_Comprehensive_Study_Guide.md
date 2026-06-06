# Study Guide: NAV Anomaly & Fraud Detection Engine

This comprehensive guide consolidates all discussions, design choices, business domain rules, architectural structures, evaluation results, and technical interview preparation materials for your portfolio project.

---

## SECTION 1: Resume Pitch & Elevator Summary

When an interviewer says: *"Tell me about a project you are proud of,"* use this structured response:

> *"I designed and built an end-to-end unsupervised machine learning pipeline to validate daily Net Asset Value (NAV) valuations and detect operational anomalies for Luxembourg's investment fund sector.*
>
> *Because real financial transactions are protected under strict banking secrecy laws, I built a financial simulator using Geometric Brownian Motion to model daily price paths and cash flows. I engineered math-based signals, such as portfolio tracking residuals and cash ratio deviations, and trained two parallel models: an Isolation Forest and a PyTorch Autoencoder.*
>
> *To comply with the auditing standards of the EU AI Act, I integrated SHAP (Shapley Additive exPlanations) to decompose the model's flags into natural language compliance explanations. Finally, I exposed the models via a FastAPI backend connected to an interactive dark-mode dashboard."*

---

## SECTION 2: Business Domain Foundations (Luxembourg Context)

### 1. What is Net Asset Value (NAV)?
The Net Asset Value is the per-share price of an investment fund. It is the price at which investors buy (subscribe) or sell (redeem) shares in the fund. It must be computed daily with absolute accuracy.

$$\text{Securities Valuation}_t = \sum (\text{Asset Price}_{i, t} \times \text{Asset Holdings}_{i, t})$$
$$\text{Total Assets}_t = \text{Securities Valuation}_t + \text{Cash Buffer}_t$$
$$\text{Net Assets (AUM)}_t = \text{Total Assets}_t - \text{Accrued Management Fees}_t$$
$$\text{NAV per Share}_t = \frac{\text{Net Assets}_t}{\text{Shares Outstanding}_t}$$

### 2. Industry Players & Roles
*   **The Custodian / Depositary Bank (e.g., BNY Mellon, State Street):** Holds the physical assets (cash, stock certificates) in safekeeping and double-checks the valuations.
*   **The Fund Administrator:** Performs the daily bookkeeping and calculates the official NAV.
*   **The Management Company (ManCo):** Legally owns the fund. They outsource the daily work to the administrator but are legally required to maintain **Portfolio Oversight** and audit the figures.
*   **The Regulator (CSSF):** Luxembourg's financial supervisor. They impose massive fines if a bank publishes an incorrect NAV.

### 3. Regulatory Auditing & the EU AI Act
Under the EU AI Act, AI systems used in credit scoring, insurance pricing, and financial anomaly detection are classified as **High-Risk** or **Limited-Risk** systems. 
*   **Explainability Requirement:** Black-box models (like neural networks) cannot be used in banking compliance without an audit trail. If a transaction is flagged, a human auditor must receive a clear report explaining *why* the model made that decision. 

---

## SECTION 3: Data Simulation & Anomaly Physics
**File: `src/data_generator.py`**

Real fund data is highly confidential (Luxembourg Financial Sector Law Article 41-1). To build this project, we simulated realistic operational dynamics:

### 1. Market Simulation (Geometric Brownian Motion)
Asset prices are modeled using a stochastic differential equation that represents how stock markets behave in the real world (incorporating drift/expected return and random shocks/volatility):

$$S_t = S_{t-1} \times e^{(\mu - 0.5\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z}$$

### 2. Transaction Flow (Log-Normal Cash Flows)
Daily subscriptions and redemptions are modeled using Log-Normal distributions. This prevents negative cash flows (you cannot buy a negative share) and represents standard, highly skewed investor withdrawal behaviors.

### 3. Anomaly Injections
We injected three operational errors that frequently occur in fund operations:
1.  **`fat_finger` (Pricing Typo):** A valuation agent inputs an asset price incorrectly (e.g., multiplying by 5). This causes `Securities Valuation` and the resulting calculated NAV to spike.
2.  **`stale_pricing` (Price Feed Failure):** The connection to Bloomberg/Reuters freezes, forcing the system to calculate NAV using yesterday's prices, failing to reflect market drops.
3.  **`cash_mismatch` (Reconciliation Break):** Shares are minted for new subscribers, but the corresponding cash transfer fails to clear in the custodian account. Outstanding shares increase, but the cash buffer remains empty, causing the calculated NAV to drop artificially.

---

## SECTION 4: Feature Engineering Strategy
**File: `src/features.py`**

Raw prices and cash totals cannot be fed to models because they drift over time. We engineered normalized features:

1.  **Tracking Residual ($NAV_{Return} - Asset_{Return}$):** Under normal operations, this residual should be close to 0. A large positive or negative difference immediately exposes `fat_finger` and `stale_pricing` errors.
2.  **Cash Ratio Deviation ($Cash\_Buffer / AUM - 0.02$):** Rebalancing keeps the cash buffer at exactly $2\%$ of assets. A drop below $2\%$ indicates a cash mismatch.
3.  **Rolling Z-Score:** Measures how many standard deviations the calculated NAV is from its 20-day moving average.
4.  **StandardScaler fitting (Train vs. Production):**
    *   *In Training:* We run `fit_transform` to compute the historical mean and standard deviation.
    *   *In Production (API):* We use `fit_scaler=False` and apply the pre-fitted scaler parameters. If we fit a new scaler on a single incoming day's record, the variance is $0$, destroying our features.

---

## SECTION 5: Machine Learning Models
**File: `src/models.py`**

We chose **unsupervised models** because operational errors and fraud cases are rare ($<5\%$) and unlabeled.

### 1. Isolation Forest (Tree-Based Baseline)
*   **Logic:** Isolation Forest isolates anomalies by randomly partitioning features. Because anomalies sit far away from normal data clusters, they require fewer splits (shorter path lengths) to isolate.
*   **Prediction Mapping:** Isolation Forest outputs $-1$ for outliers and $1$ for normal points. We map this to standard binary flags: `1` for anomalies and `0` for normal.

### 2. PyTorch Autoencoder (Deep Learning Reconstruction Model)
*   **Logic:** The Autoencoder forces the input data through a narrow bottleneck (latent space of 4 dimensions) and attempts to reconstruct it. 
*   **Reconstruction Loss:** During training on normal days, the network learns the mathematical relationships between features. When presented with anomalous data (like a cash mismatch), the network's weights fail to reconstruct it correctly. The Mean Squared Error (MSE) between input and output spikes.
*   **Quantile Threshold:** We set the threshold at the $95\text{th}$ percentile of training reconstruction losses. Any loss higher than this threshold is flagged as an anomaly.

---

## SECTION 6: Explainability Module
**File: `src/explainability.py`**

To satisfy compliance, we integrated SHAP (SHapley Additive exPlanations) to explain our model decisions.

### 1. Signal Inversion for Anomaly Scoring
Isolation Forest's `decision_function` returns a *lower* score for anomalies. To align this with SHAP (where positive values must represent an *increase* in anomaly likelihood), we wrap the scoring function:

$$\text{Wrapped Score} = -\text{decision\_function}(X)$$

### 2. Background Baseline Optimization
Kernel SHAP has $O(2^M)$ complexity, where $M$ is the number of features. To keep our FastAPI endpoint response times under 50 milliseconds, we summarize the background reference dataset to a representative sample of the top 50 normal days (`X_train.iloc[:50]`).

---

## SECTION 7: Diagnostics & Performance Analysis
**File: `run_pipeline.py`**

Our test run of the pipeline on 400 simulated days returned:

*   **Precision (Trust Rate):** 75.00% (75% of our flags were real anomalies; 25% were false alarms).
*   **Recall (Capture Rate):** 75.00% (We caught 75% of the total anomalies).
*   **Ensemble Strategy:** We use a bitwise OR (`y_pred = if_preds OR ae_preds`) because in banking operations, a False Negative (missing fraud) is far more damaging than a False Positive (investigating a false alarm).

### Category Capture Audit:
1.  **`fat_finger`:** **100% caught** (due to high tracking residuals and extreme Z-scores).
2.  **`cash_mismatch`:** **83.3% caught** (due to significant cash ratio deviations).
3.  **`stale_pricing`:** **25% caught** (highly diluted signal).

#### **The Stale Pricing Recall Problem:**
Because our feature checks the *average* return of all assets, if only 1 out of 5 asset feeds freezes, the portfolio average return is barely affected, and the anomaly escapes.
*   *Solution to pitch in interviews:* Implement **Individual Asset Momentum Features** (checking if any single asset return is exactly $0.0$ while its historical volatility is $>0$).

---

## SECTION 8: Tech Interview Q&A (Prepare to Win)

### Q1: Why did you choose unsupervised models instead of a standard XGBoost classifier?
> *"In real fund operations, we face an extreme class imbalance. True anomalies and fraud make up less than 1% of the daily transactions and are rarely labeled. A supervised model like XGBoost would suffer from severe overfitting on the normal class. Unsupervised models like Isolation Forest and Autoencoders solve this by learning only the normal data distribution."*

### Q2: How does your Autoencoder detect an anomaly?
> *"The Autoencoder acts as a compression filter. It compresses the 10 input features down to a 4-dimensional latent space, then reconstructs the original 10 features. Since it was trained on normal data, it learns the relationships between assets, NAV, and cash buffer. When an anomaly day is fed, these normal relationships break. The network cannot reconstruct the input accurately, causing the reconstruction loss (MSE) to spike above our 95th percentile threshold."*

### Q3: What is the benefit of SHAP over simpler feature importance methods like MDI in Random Forests?
> *"Feature importance methods like MDI or Permutation Importance only tell us which features are important for the model overall. They do not tell us why a **specific individual day** was flagged. SHAP uses game-theory Shapley values to calculate the exact contribution of each feature for a **single, specific transaction**, which is exactly what a regulatory auditor needs."*

### Q4: How does your project align with the EU AI Act?
> *"Under the EU AI Act, automated risk validation models must be auditable and transparent. Our pipeline integrates SHAP to translate complex model parameters into natural language compliance explanations. This provides a human-in-the-loop audit trail, explaining the exact mathematical factors behind every single flagged alert."*

---

## SECTION 9: Career Networking & Job Search Strings

Use these LinkedIn Boolean search strings to find contacts in Luxembourg and the US:

*   **Fintech developers & product leaders:**
    ```text
    (fund OR investment OR manco OR portfolio OR asset OR fintech OR operations) AND (Manager OR Director OR Head OR VP OR Chief OR Founder OR Executive OR Lead)
    ```
*   **Warm Outreach Template:**
    > *"Hi [Name], I am a Computer Science & AI major graduating from Uni Lu. I've been building a project around automating NAV anomaly detection and cash reconciliation using PyTorch Autoencoders and SHAP explainability. I saw you work in [Fintech/Oversight/Consulting] and would love to connect to learn about current technology patterns in the Luxembourg market. Best, [Your Name]."*
