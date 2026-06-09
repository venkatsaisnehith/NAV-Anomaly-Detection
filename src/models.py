"""
Machine Learning Models Module (models.py)
Implements two anomaly detection models:
1. Isolation Forest (Classical unsupervised tree-based baseline)
2. PyTorch Autoencoder (Deep learning reconstruction-loss model)
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.ensemble import IsolationForest
import logging

# We import PyTorch dynamically inside the Autoencoder class to prevent import failures 
# if the environment is still being initialized, but since it is required, we use standard torch.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Models")


class IsolationForestDetector:
    """
    Wrapper for scikit-learn's Isolation Forest model.
    Good for fast training, non-linear relationships, and tabular baselines.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> None:
        """Fits the Isolation Forest on the training feature matrix X."""
        logger.info("Fitting Isolation Forest model...")
        self.model.fit(X)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomalies.
        Returns:
            Tuple of:
            - binary_predictions: (1 = Anomaly, 0 = Normal)
            - anomaly_scores: Float values (lower means more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
            
        raw_preds = self.model.predict(X)  # Returns -1 for anomaly, 1 for normal
        binary_preds = np.where(raw_preds == -1, 1, 0)
        
        # Decision function returns signed distance to separating hyperplane
        # Lower score = more anomalous
        scores = self.model.decision_function(X)
        return binary_preds, scores


# Define PyTorch Network
if TORCH_AVAILABLE:
    class AutoencoderNet(nn.Module):
        """
        Autoencoder architecture that compresses input features and reconstructs them.
        """
        def __init__(self, input_dim: int, latent_dim: int = 4):
            super(AutoencoderNet, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, latent_dim),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8),
                nn.ReLU(),
                nn.Linear(8, input_dim)
            )

        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed
else:
    # Fallback definition when PyTorch dependencies are not available
    class AutoencoderNet:
        pass


class AutoencoderDetector:
    """
    Deep Learning Autoencoder for Anomaly Detection.
    Trains on 'normal' data to minimize reconstruction loss.
    High reconstruction loss = Anomaly.
    """

    def __init__(self, latent_dim: int = 4, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch is not installed. AutoencoderDetector will not work. Install PyTorch.")
            
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net = None
        self.threshold = None  # Quantile threshold to determine binary anomaly classification
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, contamination: float = 0.05) -> None:
        """
        Fits the Autoencoder.
        Finds reconstruction error threshold based on the specified contamination.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AutoencoderDetector.")

        logger.info("Initializing and training PyTorch Autoencoder...")
        input_dim = X.shape[1]
        self.net = AutoencoderNet(input_dim=input_dim, latent_dim=self.latent_dim)
        
        # Convert DataFrame to PyTorch DataLoader
        x_tensor = torch.tensor(X.values, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                inputs = batch[0]
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss / len(X):.4f}")

        # Compute threshold based on training set reconstruction losses
        self.net.eval()
        with torch.no_grad():
            reconstructed_train = self.net(x_tensor)
            # Row-wise mean squared error (reconstruction loss)
            train_losses = torch.mean((x_tensor - reconstructed_train) ** 2, dim=1).numpy()
            
        # The threshold is set at the (1 - contamination) percentile.
        # Anything higher is classified as an anomaly.
        self.threshold = np.quantile(train_losses, 1 - contamination)
        logger.info(f"Set reconstruction loss anomaly threshold to: {self.threshold:.4f}")
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomalies using reconstruction loss.
        Returns:
            Tuple of:
            - binary_predictions: (1 = Anomaly, 0 = Normal)
            - reconstruction_losses: Float MSE loss per row (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")

        self.net.eval()
        x_tensor = torch.tensor(X.values, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed = self.net(x_tensor)
            losses = torch.mean((x_tensor - reconstructed) ** 2, dim=1).numpy()

        binary_preds = np.where(losses > self.threshold, 1, 0)
        return binary_preds, losses


if __name__ == "__main__":
    # Small verification test with dummy data
    print("PyTorch Available:", TORCH_AVAILABLE)
    
    # Generate mock features
    dummy_data = pd.DataFrame(np.random.normal(size=(100, 10)))
    
    # Test Isolation Forest
    logger.info("Testing Isolation Forest...")
    if_detector = IsolationForestDetector(contamination=0.05)
    if_detector.fit(dummy_data)
    if_preds, if_scores = if_detector.predict(dummy_data)
    print("Isolation Forest anomalies detected:", if_preds.sum())

    # Test Autoencoder
    if TORCH_AVAILABLE:
        logger.info("Testing PyTorch Autoencoder...")
        ae_detector = AutoencoderDetector(epochs=10, batch_size=16)
        ae_detector.fit(dummy_data)
        ae_preds, ae_losses = ae_detector.predict(dummy_data)
        print("Autoencoder anomalies detected:", ae_preds.sum())
    else:
        print("Skipping Autoencoder test (PyTorch not available)")
