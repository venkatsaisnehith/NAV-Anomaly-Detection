"""
Feature Engineering Module (features.py)
Preprocesses raw simulated fund operations and asset prices.
Computes mathematical residuals, rolling Z-scores, and transactional features
specifically designed to expose financial anomalies to machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureEngineer")


class FeatureEngineer:
    """
    Constructs model features from raw fund data and individual asset prices.
    Scales features and prepares them for training or real-time API inference.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []

    def construct_features(
        self,
        df_fund: pd.DataFrame,
        df_assets: pd.DataFrame,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Engineers features to capture anomalies.

        Args:
            df_fund: Fund operational data.
            df_assets: Daily prices of underlying assets.
            fit_scaler: If True, fits the scaler on this dataset. If False, transforms using fitted scaler.

        Returns:
            DataFrame of scaled features ready for machine learning.
        """
        logger.info("Starting feature engineering process...")
        
        # Ensure indices match
        common_idx = df_fund.index.intersection(df_assets.index)
        df_fund = df_fund.loc[common_idx].copy()
        df_assets = df_assets.loc[common_idx].copy()

        # 1. Compute Returns
        # Daily percentage change in Calculated NAV
        df_fund["NAV_Return"] = df_fund["Calculated_NAV"].pct_change().fillna(0)
        
        # Daily percentage changes in underlying assets
        df_asset_returns = df_assets.pct_change().fillna(0)

        # 2. Compute Benchmark (Weighted Asset) Return
        # Find the starting asset weights by looking at their proportion on day 1
        first_row_assets = df_assets.iloc[0].values
        # To simulate weights, we'll approximate using the initial Dirichlet weights.
        # But dynamically, we can estimate weights as the ratio of asset holdings value to total portfolio value.
        # For simplicity, we calculate the average daily returns of the asset class.
        mean_asset_return = df_asset_returns.mean(axis=1)

        # 3. Compute Tracking Residual
        # This is the single most powerful feature!
        # Under normal conditions, NAV_Return - mean_asset_return should be very close to 0.
        # A large deviation points directly to a fat-finger or stale price.
        df_fund["Tracking_Residual"] = df_fund["NAV_Return"] - mean_asset_return

        # 4. Compute Rolling Volatility of NAV
        df_fund["NAV_Vol_5d"] = df_fund["NAV_Return"].rolling(window=5).std().fillna(0)
        df_fund["NAV_Vol_20d"] = df_fund["NAV_Return"].rolling(window=20).std().fillna(0)

        # 5. Cash Account Features
        # Cash buffer ratio: Cash should stay around 2% of AUM.
        df_fund["Cash_Ratio"] = df_fund["Cash_Buffer"] / df_fund["AUM"]
        df_fund["Cash_Ratio_Deviation"] = df_fund["Cash_Ratio"] - 0.02

        # 6. Transaction Flow Features
        # Subscriptions and redemptions scaled by AUM
        df_fund["Sub_Ratio"] = df_fund["Subscriptions"] / df_fund["AUM"]
        df_fund["Red_Ratio"] = df_fund["Redemptions"] / df_fund["AUM"]
        
        # Net Cash Flow Ratio
        df_fund["Net_Flow_Ratio"] = df_fund["Sub_Ratio"] - df_fund["Red_Ratio"]

        # 7. Roll-up Z-Scores
        # Rolling Z-score of NAV deviation to identify out-of-bounds prices
        rolling_mean = df_fund["Calculated_NAV"].rolling(window=20).mean().fillna(method="bfill")
        rolling_std = df_fund["Calculated_NAV"].rolling(window=20).std().fillna(method="bfill")
        # Avoid division by zero
        rolling_std = np.where(rolling_std == 0, 1e-6, rolling_std)
        df_fund["NAV_ZScore_20d"] = (df_fund["Calculated_NAV"] - rolling_mean) / rolling_std

        # Compile final features to send to model
        model_features = [
            "NAV_Return",
            "Tracking_Residual",
            "NAV_Vol_5d",
            "NAV_Vol_20d",
            "Cash_Ratio",
            "Cash_Ratio_Deviation",
            "Sub_Ratio",
            "Red_Ratio",
            "Net_Flow_Ratio",
            "NAV_ZScore_20d"
        ]
        
        self.feature_columns = model_features
        df_features_raw = df_fund[model_features].copy()
        
        # Replace any infs or NaNs that escaped
        df_features_raw = df_features_raw.replace([np.inf, -np.inf], 0).fillna(0)

        # Scale features
        if fit_scaler:
            logger.info("Fitting and transforming features using StandardScaler...")
            scaled_array = self.scaler.fit_transform(df_features_raw)
        else:
            logger.info("Transforming features using pre-fitted StandardScaler...")
            scaled_array = self.scaler.transform(df_features_raw)

        df_scaled = pd.DataFrame(
            scaled_array, 
            index=df_features_raw.index, 
            columns=df_features_raw.columns
        )

        return df_scaled


if __name__ == "__main__":
    # Test feature builder
    from data_generator import FundSimulator
    
    sim = FundSimulator(seed=42)
    fund, assets = sim.generate_time_series(days=100)
    
    engineer = FeatureEngineer()
    features = engineer.construct_features(fund, assets)
    
    print("\n--- Features Shape ---")
    print(features.shape)
    
    print("\n--- Feature Matrix Sample (First 3 Days) ---")
    print(features.head(3))
