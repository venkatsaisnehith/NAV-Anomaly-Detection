"""
Pipeline Orchestration Script (run_pipeline.py)
Executes the entire fund anomaly detection workflow:
1. Simulates a history of fund operations.
2. Extracts engineered features and scales them.
3. Trains the ML models (Isolation Forest & PyTorch Autoencoder).
4. Evaluates model performance (F1-score, Precision, Recall).
5. Generates compliance explanations (SHAP) for sample anomalies.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_generator import FundSimulator
from features import FeatureEngineer
from models import IsolationForestDetector, AutoencoderDetector, TORCH_AVAILABLE
from explainability import AnomalyExplainer


def run_pipeline():
    print("=" * 60)
    print("      NAV ANOMALY & FRAUD DETECTION PIPELINE RUN       ")
    print("=" * 60)
    
    # 1. Simulate Fund Data
    print("\n[Step 1/5] Simulating fund operations data...")
    sim = FundSimulator(num_assets=5, initial_aum=120_000_000.0, seed=42)
    # Generate 400 trading days
    df_fund, df_assets = sim.generate_time_series(days=400, anomaly_ratio=0.05)
    
    # Keep track of ground truth labels
    y_true = df_fund["Is_Anomaly"].values
    anomaly_types = df_fund["Anomaly_Type"].values
    
    # 2. Feature Engineering
    print("\n[Step 2/5] Engineering financial features...")
    engineer = FeatureEngineer()
    df_features = engineer.construct_features(df_fund, df_assets, fit_scaler=True)
    print(f"Feature matrix shape: {df_features.shape}")
    
    # 3. Model Training
    print("\n[Step 3/5] Training anomaly detection models...")
    
    # Model A: Isolation Forest
    if_detector = IsolationForestDetector(contamination=0.05, random_state=42)
    if_detector.fit(df_features)
    if_preds, if_scores = if_detector.predict(df_features)
    
    # Model B: PyTorch Autoencoder
    ae_preds = None
    if TORCH_AVAILABLE:
        ae_detector = AutoencoderDetector(epochs=40, batch_size=16, lr=0.001)
        ae_detector.fit(df_features, contamination=0.05)
        ae_preds, ae_losses = ae_detector.predict(df_features)
    else:
        print(" -> PyTorch not found. Skipping Autoencoder training.")

    # Combined Ensembled Predictions (flagged if either model flags)
    if ae_preds is not None:
        y_pred = np.bitwise_or(if_preds, ae_preds)
    else:
        y_pred = if_preds

    # 4. Evaluation
    print("\n[Step 4/5] Evaluating detection accuracy against ground truth...")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("-" * 40)
    print(f"Precision (True Positive Rate): {precision * 100:.2f}%")
    print(f"Recall (Capture Rate):         {recall * 100:.2f}%")
    print(f"F1-Score:                      {f1 * 100:.2f}%")
    print("-" * 40)
    print("\nClassification Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Analyze capture rates by anomaly type
    df_eval = pd.DataFrame({
        "Type": anomaly_types,
        "True": y_true,
        "Pred": y_pred
    })
    
    print("\nCapture rate by anomaly category:")
    for name, group in df_eval[df_eval["True"] == 1].groupby("Type"):
        captured = group["Pred"].sum()
        total = len(group)
        print(f" - {name:15s}: {captured} / {total} caught ({captured/total*100:.1f}%)")

    # 5. Explainability Inspection
    print("\n[Step 5/5] Generating SHAP explainability audit logs for samples...")
    explainer = AnomalyExplainer(if_detector, df_features, engineer.feature_columns)
    
    # Find the first few flagged anomalies
    flagged_indices = np.where(y_pred == 1)[0]
    
    if len(flagged_indices) > 0:
        print("\n--- Compliance Explanation Report (Top 2 Flags) ---")
        for idx in flagged_indices[:2]:
            date_label = df_fund.index[idx].strftime("%Y-%m-%d")
            true_type = anomaly_types[idx]
            row_features = df_features.iloc[idx]
            
            print(f"\n[FLAG DATE: {date_label}] (Injected Bug Type: {true_type})")
            explanation = explainer.explain_instance(row_features)
            
            if explanation["status"] == "success":
                for note in explanation["top_contributors"]:
                    print(f"  - {note}")
            else:
                print(f"  - Explanation Status: {explanation['status']}")
    else:
        print("\nNo anomalies were flagged during this run.")

    print("\n" + "=" * 60)
    print("                      PIPELINE RUN COMPLETE                   ")
    print("=" * 60)
    print("\nTo launch the real-time API and browse the Dashboard UI:")
    print(" 1. Run command: uvicorn project2_anomaly_detection.src.api:app --reload")
    print(" 2. Open file in Web Browser: project2_anomaly_detection/templates/dashboard.html")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
