"""
Explainability Module (explainability.py)
Provides post-hoc explanations for flagged anomalies using SHAP (SHapley Additive exPlanations).
Helps compliance officers audit why the ML models flagged a day's NAV calculation.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any

# Import SHAP dynamically to handle systems without it installed initially
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Explainability")


class AnomalyExplainer:
    """
    Computes SHAP values to explain the anomaly scores produced by the models.
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, feature_names: List[str]):
        """
        Args:
            model: The fitted anomaly detector (IsolationForestDetector or AutoencoderDetector).
            X_train: The baseline feature matrix used to compute expected values.
            feature_names: List of column names in the feature matrix.
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library is not installed. AnomalyExplainer will return empty explanations.")
            return

        # Setup explainer
        logger.info("Initializing SHAP explainer...")
        try:
            # We want to explain the score (decision_function or reconstruction loss)
            # For Isolation Forest: decision_function (lower is more anomalous, so we explain the negative score to show what INCREASED anomaly)
            # For Autoencoder: reconstruction loss (higher is more anomalous)
            if hasattr(model, 'decision_function'):
                # Wrap the decision function
                def score_func(X):
                    # We invert the score so that positive SHAP values correspond to making the row MORE anomalous
                    return -self.model.decision_function(X)
                
                # Use KernelExplainer or generic Explainer on the wrapper score function
                self.explainer = shap.Explainer(score_func, X_train.iloc[:50]) # Use a representative background sample to run fast
            elif hasattr(model, 'net') or hasattr(model, 'predict'):
                # For Autoencoder or custom model: wrap the prediction loss function
                def ae_loss_func(X):
                    # Predict returns (binary_preds, losses)
                    _, losses = self.model.predict(pd.DataFrame(X, columns=self.feature_names))
                    return losses
                
                self.explainer = shap.Explainer(ae_loss_func, X_train.iloc[:50])
        except Exception as e:
            logger.error("Failed to initialize SHAP explainer: %s", e)

    def explain_instance(self, X_row: pd.Series) -> Dict[str, Any]:
        """
        Computes SHAP values for a single row and returns a structured explanation dictionary.

        Args:
            X_row: A pandas Series representing the day's features.

        Returns:
            Dictionary containing feature names, raw values, SHAP impact scores, and English descriptions.
        """
        explanation = {
            "feature_values": X_row.to_dict(),
            "shap_values": {},
            "top_contributors": [],
            "status": "success"
        }

        if not SHAP_AVAILABLE or self.explainer is None:
            explanation["status"] = "SHAP not available"
            return explanation

        try:
            # Convert single Series row to 2D shape for SHAP
            X_df = pd.DataFrame([X_row])
            shap_values = self.explainer(X_df)
            
            # Extract values for the first (and only) instance
            row_shap = shap_values.values[0]
            
            # Map features to their SHAP value
            feature_impacts = {}
            for name, val in zip(self.feature_names, row_shap):
                feature_impacts[name] = float(val)
                explanation["shap_values"][name] = float(val)

            # Sort features by absolute impact (highest magnitude first)
            sorted_impacts = sorted(
                feature_impacts.items(),
                key=lambda item: abs(item[1]),
                reverse=True
            )

            # Generate natural language narrative for the top contributors
            narratives = []
            for feat, impact in sorted_impacts[:3]:
                # Only report significant impacts
                if abs(impact) < 0.05:
                    continue
                    
                val_raw = X_row[feat]
                direction = "increased" if impact > 0 else "decreased"
                
                narrative = f"'{feat}' value of {val_raw:.2f} {direction} the anomaly score (impact: {impact:.3f})"
                narratives.append(narrative)
                
            explanation["top_contributors"] = narratives

        except Exception as e:
            logger.error("Error computing SHAP values: %s", e)
            explanation["status"] = f"error: {str(e)}"

        return explanation


if __name__ == "__main__":
    # Test stub with synthetic variables
    if not SHAP_AVAILABLE:
        print("SHAP is not installed, skipping test.")
    else:
        # Create a mock decision class
        class MockModel:
            def decision_function(self, X):
                # Return distance (e.g. sum of columns)
                return -np.sum(X, axis=1)

        feature_cols = ["f1", "f2", "f3"]
        df_train = pd.DataFrame(np.random.normal(size=(100, 3)), columns=feature_cols)
        
        mock_model = MockModel()
        explainer = AnomalyExplainer(mock_model, df_train, feature_cols)
        
        test_row = pd.Series([2.5, -0.1, 0.2], index=feature_cols)
        explanation_result = explainer.explain_instance(test_row)
        
        print("\n--- SHAP Anomaly Explanation Report ---")
        print("Narrative Summary of Flags:")
        for note in explanation_result["top_contributors"]:
            print(f" - {note}")
        print("\nRaw SHAP Impacts:")
        print(explanation_result["shap_values"])
