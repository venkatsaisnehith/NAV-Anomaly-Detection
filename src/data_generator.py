"""
Fund Simulator (data_generator.py)
Generates highly realistic daily timeseries data for an investment fund.
Models assets, cash flows, accrued fees, shares outstanding, and calculated NAV.
Injects controlled financial anomalies (fat-finger, stale pricing, cash flow errors).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FundSimulator")


class FundSimulator:
    """
    Simulates daily operations of a Luxembourg-style investment fund.
    Calculates NAV, manages cash accounts, and handles asset valuations.
    """

    def __init__(
        self,
        num_assets: int = 5,
        initial_aum: float = 100_000_000.0,  # €100 Million initial Assets Under Management
        management_fee_annual: float = 0.015,  # 1.5% annual fee
        initial_nav_per_share: float = 100.0,  # Starting NAV price per share
        seed: int = 42
    ):
        self.num_assets = num_assets
        self.initial_aum = initial_aum
        self.fee_daily_rate = management_fee_annual / 252.0  # Assumes 252 trading days per year
        self.initial_nav = initial_nav_per_share
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Define assets (a mix of blue-chip equities and fixed income)
        self.asset_names = [f"Asset_{i+1}" for i in range(num_assets)]
        self.asset_weights = np.random.dirichlet(np.ones(num_assets))  # Random weights summing to 1
        
        # Set expected annual return and volatility for each asset
        self.asset_expected_returns = np.random.uniform(0.04, 0.12, num_assets) / 252.0
        self.asset_vols = np.random.uniform(0.10, 0.25, num_assets) / np.sqrt(252.0)

    def generate_time_series(
        self,
        days: int = 500,
        anomaly_ratio: float = 0.04  # ~4% of days will contain anomalies
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates daily time-series dataset of the fund operations.

        Returns:
            Tuple of:
            - df_fund: Fund operational data (NAV, Shares, Cash, Fees).
            - df_assets: Individual asset price histories.
        """
        logger.info(f"Simulating {days} days of fund operations...")

        # Initialize tracking containers
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="B")  # Business days
        
        # Asset prices initialization (starting at €100.0)
        asset_prices = np.zeros((days, self.num_assets))
        asset_prices[0] = 100.0
        
        # Simulate asset price paths using Geometric Brownian Motion
        for t in range(1, days):
            for i in range(self.num_assets):
                # Daily change with drift and volatility
                drift = self.asset_expected_returns[i]
                shock = self.asset_vols[i] * np.random.normal()
                asset_prices[t, i] = asset_prices[t-1, i] * np.exp(drift - 0.5 * (self.asset_vols[i]**2) + shock)
        
        df_assets = pd.DataFrame(asset_prices, index=dates, columns=self.asset_names)

        # Fund portfolios and shares initialization
        shares_outstanding = np.zeros(days)
        shares_outstanding[0] = self.initial_aum / self.initial_nav
        
        # Target asset holdings (number of units of each asset in the portfolio)
        # Allocate initial AUM minus a small cash buffer (e.g., 2% cash buffer)
        cash_buffer_ratio = 0.02
        cash = np.zeros(days)
        cash[0] = self.initial_aum * cash_buffer_ratio
        
        invested_aum = self.initial_aum * (1 - cash_buffer_ratio)
        asset_holdings = (invested_aum * self.asset_weights) / asset_prices[0]
        
        # Operational variables
        aum = np.zeros(days)
        aum[0] = self.initial_aum
        
        nav_per_share = np.zeros(days)
        nav_per_share[0] = self.initial_nav
        
        fees_accrued = np.zeros(days)
        subscriptions = np.zeros(days)
        redemptions = np.zeros(days)
        
        # Ground truth labels for anomalies
        is_anomaly = np.zeros(days, dtype=int)
        anomaly_type = ["None"] * days

        # Determine indices where we will inject anomalies
        anomaly_indices = np.random.choice(range(10, days - 10), size=int(days * anomaly_ratio), replace=False)

        # Daily fund cycle loop
        for t in range(1, days):
            # 1. Simulate standard daily client cash flows (subscriptions & redemptions)
            # Log-normal distribution to avoid negative subscriptions
            sub_ratio = np.random.lognormal(-5.0, 0.5)  # typical small subscription percentage
            red_ratio = np.random.lognormal(-5.0, 0.5)  # typical small redemption percentage
            
            # Absolute Cash Values
            subscriptions[t] = aum[t-1] * sub_ratio
            redemptions[t] = aum[t-1] * red_ratio
            
            # Net Cash flow into the fund
            net_cash_flow = subscriptions[t] - redemptions[t]
            
            # Adjust Shares Outstanding (at yesterday's NAV per share)
            shares_created = subscriptions[t] / nav_per_share[t-1]
            shares_redeemed = redemptions[t] / nav_per_share[t-1]
            shares_outstanding[t] = shares_outstanding[t-1] + shares_created - shares_redeemed
            
            # 2. Portfolio Valuation
            # Value of underlying securities at today's close
            securities_valuation = np.sum(asset_holdings * asset_prices[t])
            
            # Update cash (add net cash flow, minus any security purchases to rebalance)
            # For simplicity, we assume we keep the subscription cash in cash buffer for today, 
            # and rebalance at the end of the day.
            cash[t] = cash[t-1] + net_cash_flow
            
            total_assets = securities_valuation + cash[t]
            
            # 3. Accrue management fee
            fees_accrued[t] = total_assets * self.fee_daily_rate
            net_assets = total_assets - fees_accrued[t]
            
            # Deduct fee from cash
            cash[t] -= fees_accrued[t]
            
            # 4. Calculate Net Asset Value (NAV) per share
            aum[t] = net_assets
            nav_per_share[t] = net_assets / shares_outstanding[t]

            # Rebalance fund holdings back to target weights to simulate realistic portfolio maintenance
            # We do this at the end of the day by trading some assets to match target weights
            target_invested = net_assets * (1 - cash_buffer_ratio)
            asset_holdings = (target_invested * self.asset_weights) / asset_prices[t]
            cash[t] = net_assets * cash_buffer_ratio

        # 5. Inject Anomalies
        # Create a copy of prices to modify for the anomaly days
        corrupted_nav = nav_per_share.copy()
        corrupted_cash = cash.copy()
        
        for idx in anomaly_indices:
            is_anomaly[idx] = 1
            choice = np.random.choice(["fat_finger", "stale_pricing", "cash_mismatch"])
            anomaly_type[idx] = choice
            
            if choice == "fat_finger":
                # Ingress a data entry error: e.g. keying a 10x multiplier on Asset 1 price
                # This causes the valuation of that day to spike abnormally
                corrupted_price = df_assets.iloc[idx, 0] * 5.0
                # Re-calculate securities valuation for that day using corrupted price
                temp_asset_prices = df_assets.iloc[idx].values.copy()
                temp_asset_prices[0] = corrupted_price
                
                temp_securities = np.sum(asset_holdings * temp_asset_prices)
                temp_assets = temp_securities + cash[idx]
                temp_net_assets = temp_assets - fees_accrued[idx]
                corrupted_nav[idx] = temp_net_assets / shares_outstanding[idx]
                
            elif choice == "stale_pricing":
                # Price feed freezes: Asset prices fail to update from the previous day, 
                # despite volatility.
                # Re-calculate using yesterday's prices
                temp_securities = np.sum(asset_holdings * df_assets.iloc[idx-1].values)
                temp_assets = temp_securities + cash[idx]
                temp_net_assets = temp_assets - fees_accrued[idx]
                corrupted_nav[idx] = temp_net_assets / shares_outstanding[idx]

            elif choice == "cash_mismatch":
                # Inflow cash mismatch: Subscriptions are recorded, shares are minted, 
                # but cash fails to be deposited into the cash buffer.
                lost_cash = subscriptions[idx] * 0.90  # 90% of subscriptions cash goes missing
                corrupted_cash[idx] -= lost_cash
                
                # Recalculate AUM and NAV
                temp_securities = np.sum(asset_holdings * df_assets.iloc[idx].values)
                temp_assets = temp_securities + corrupted_cash[idx]
                temp_net_assets = temp_assets - fees_accrued[idx]
                corrupted_nav[idx] = temp_net_assets / shares_outstanding[idx]

        # Construct final fund operations DataFrame
        df_fund = pd.DataFrame(
            {
                "AUM": aum,
                "Shares_Outstanding": shares_outstanding,
                "Cash_Buffer": corrupted_cash,
                "Fees_Accrued": fees_accrued,
                "Subscriptions": subscriptions,
                "Redemptions": redemptions,
                "Calculated_NAV": corrupted_nav,  # Contains the injected bugs!
                "True_NAV": nav_per_share,        # The uncorrupted NAV for debugging/validation
                "Is_Anomaly": is_anomaly,
                "Anomaly_Type": anomaly_type
            },
            index=dates
        )
        
        logger.info(f"Generated {days} days of data. Injected {is_anomaly.sum()} anomalies.")
        return df_fund, df_assets


if __name__ == "__main__":
    # Test generation and print summary
    simulator = FundSimulator(num_assets=5, seed=42)
    fund_df, assets_df = simulator.generate_time_series(days=100, anomaly_ratio=0.05)
    
    print("\n--- Injected Anomalies Summary ---")
    print(fund_df[fund_df["Is_Anomaly"] == 1][["Calculated_NAV", "True_NAV", "Anomaly_Type"]].head(10))
