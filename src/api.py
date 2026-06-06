"""
FastAPI Validation Service (api.py)
Exposes REST endpoints to query simulated fund data, train models,
evaluate incoming valuation records, and return anomaly flags and explanations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

from data_generator import FundSimulator
from features import FeatureEngineer
from models import IsolationForestDetector, AutoencoderDetector, TORCH_AVAILABLE
from explainability import AnomalyExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIService")

app = FastAPI(
    title="NAV Anomaly & Fraud Detection Engine API",
    description="Production-grade API to validate daily Net Asset Value calculations and explain anomalies.",
    version="1.0.0"
)

# Enable CORS for frontend dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances of our trained models and utilities
GLOBAL_STATE = {
    "simulator": None,
    "engineer": None,
    "if_model": None,
    "ae_model": None,
    "explainer": None,
    "df_fund": None,
    "df_assets": None,
    "df_features": None,
    "is_trained": False
}


class AssetPrices(BaseModel):
    Asset_1: float
    Asset_2: float
    Asset_3: float
    Asset_4: float
    Asset_5: float


class NAVValidationRecord(BaseModel):
    AUM: float
    Cash_Buffer: float
    Subscriptions: float
    Redemptions: float
    Calculated_NAV: float
    Asset_Prices: AssetPrices


@app.on_event("startup")
def startup_event():
    """Initializes and trains the models on startup to ensure API is ready to use."""
    logger.info("Initializing models on startup...")
    try:
        train_pipeline()
    except Exception as e:
        logger.error("Failed to train models on startup: %s", e)


def train_pipeline():
    """Simulates fund data, engineers features, and fits the ML models."""
    logger.info("Starting background training pipeline...")
    
    # 1. Simulate data
    sim = FundSimulator(num_assets=5, seed=42)
    df_fund, df_assets = sim.generate_time_series(days=300, anomaly_ratio=0.04)
    
    # 2. Extract features
    engineer = FeatureEngineer()
    df_features = engineer.construct_features(df_fund, df_assets, fit_scaler=True)
    
    # 3. Train Isolation Forest
    if_model = IsolationForestDetector(contamination=0.04)
    if_model.fit(df_features)
    
    # 4. Train Autoencoder (if PyTorch is available)
    ae_model = None
    if TORCH_AVAILABLE:
        ae_model = AutoencoderDetector(epochs=30, batch_size=16)
        ae_model.fit(df_features, contamination=0.04)
    else:
        logger.warning("PyTorch not available, running API without Autoencoder model.")
        
    # 5. Initialize Explainer (we use Isolation Forest as the main explainer model)
    explainer = AnomalyExplainer(if_model, df_features, engineer.feature_columns)
    
    # Update global state
    GLOBAL_STATE["simulator"] = sim
    GLOBAL_STATE["engineer"] = engineer
    GLOBAL_STATE["if_model"] = if_model
    GLOBAL_STATE["ae_model"] = ae_model
    GLOBAL_STATE["explainer"] = explainer
    GLOBAL_STATE["df_fund"] = df_fund
    GLOBAL_STATE["df_assets"] = df_assets
    GLOBAL_STATE["df_features"] = df_features
    GLOBAL_STATE["is_trained"] = True
    
    logger.info("Training pipeline completed successfully.")


@app.post("/train", summary="Retrain the models on a new simulated fund dataset")
def retrain():
    try:
        train_pipeline()
        return {"status": "success", "message": "Models retrained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/data", summary="Fetch simulated timeseries data for the dashboard UI")
def get_dashboard_data():
    if not GLOBAL_STATE["is_trained"]:
        raise HTTPException(status_code=503, detail="Service is initializing, try again shortly.")
        
    # Unpack simulated fund dataframe
    fund_df = GLOBAL_STATE["df_fund"].copy()
    
    # Convert index (dates) to string for JSON serialization
    fund_df["Date"] = fund_df.index.strftime("%Y-%m-%d")
    
    # Return raw records
    return fund_df.to_dict(orient="records")


@app.post("/validate", summary="Validate a single day's NAV calculation and check for anomalies")
def validate_nav(record: NAVValidationRecord):
    if not GLOBAL_STATE["is_trained"]:
        raise HTTPException(status_code=503, detail="Models are not trained or initialized.")
        
    engineer = GLOBAL_STATE["engineer"]
    if_model = GLOBAL_STATE["if_model"]
    ae_model = GLOBAL_STATE["ae_model"]
    explainer = GLOBAL_STATE["explainer"]

    # 1. Structure the incoming record to match our data format
    raw_fund_dict = {
        "AUM": [record.AUM],
        "Cash_Buffer": [record.Cash_Buffer],
        "Subscriptions": [record.Subscriptions],
        "Redemptions": [record.Redemptions],
        "Calculated_NAV": [record.Calculated_NAV]
    }
    df_single_fund = pd.DataFrame(raw_fund_dict, index=[pd.Timestamp.now()])

    raw_assets_dict = {
        "Asset_1": [record.Asset_Prices.Asset_1],
        "Asset_2": [record.Asset_Prices.Asset_2],
        "Asset_3": [record.Asset_Prices.Asset_3],
        "Asset_4": [record.Asset_Prices.Asset_4],
        "Asset_5": [record.Asset_Prices.Asset_5]
    }
    df_single_assets = pd.DataFrame(raw_assets_dict, index=[pd.Timestamp.now()])

    # 2. Extract and scale features
    try:
        # Pass through the feature engineer using the fitted scaler
        df_single_features = engineer.construct_features(
            df_single_fund,
            df_single_assets,
            fit_scaler=False
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {str(e)}")

    # 3. Model Predictions
    if_flag, if_score = if_model.predict(df_single_features)
    
    ae_flag, ae_score = 0, 0.0
    if ae_model:
        ae_flag, ae_score = ae_model.predict(df_single_features)
        
    # Anomaly condition: flagged if either model identifies it (high sensitivity)
    is_anomaly = bool(if_flag[0] == 1 or ae_flag[0] == 1)

    # 4. Generate SHAP explanation if anomalous
    explanation = None
    if is_anomaly and explainer:
        single_row = df_single_features.iloc[0]
        explanation = explainer.explain_instance(single_row)

    return {
        "is_anomaly": is_anomaly,
        "models": {
            "isolation_forest": {
                "flag": int(if_flag[0]),
                "score": float(if_score[0])
            },
            "autoencoder": {
                "flag": int(ae_flag),
                "score": float(ae_score)
            }
        },
        "explanation": explanation
    }


if __name__ == "__main__":
    import uvicorn
    # Local dev server run
    uvicorn.run(app, host="127.0.0.1", port=8000)
