"""Baseline models, XGBoost detector, PyTorch Autoencoder, and calibrated Hybrid Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BenignAutoencoder(nn.Module):
    """Symmetric deep autoencoder trained exclusively on benign telemetry."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderAnomalyDetector:
    """Wrapper handling scaling, benign-only PyTorch training, and anomaly scoring."""

    def __init__(self, input_dim: int, epochs: int = 25, lr: float = 1e-3, seed: int = 42) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.scaler = StandardScaler()
        self.model = BenignAutoencoder(input_dim=input_dim)
        self.epochs = epochs
        self.lr = lr
        self._train_p99: float = 1.0

    def fit(self, X_train: np.ndarray, y_binary_train: np.ndarray) -> "AutoencoderAnomalyDetector":
        """Fit scaler and autoencoder strictly on benign training samples."""
        X_benign = X_train[y_binary_train == 0]
        X_scaled = self.scaler.fit_transform(X_benign)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        self.model.train()
        batch_size = 256
        n = tensor_x.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                batch = tensor_x[perm[i : i + batch_size]]
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()

        raw_errors = self.reconstruction_errors(X_benign)
        self._train_p99 = max(float(np.percentile(raw_errors, 99)), 1e-6)
        return self

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            tx = torch.tensor(X_scaled, dtype=torch.float32)
            recon = self.model(tx)
            mse = torch.mean((tx - recon) ** 2, dim=1).numpy()
        return mse

    def per_feature_errors(self, X: np.ndarray) -> np.ndarray:
        """Return squared error per feature for anomaly explainability."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            tx = torch.tensor(X_scaled, dtype=torch.float32)
            recon = self.model(tx)
            sq_err = ((tx - recon) ** 2).numpy()
        return sq_err

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated [0, 1) anomaly score using monotonic sigmoid scaling."""
        mse = self.reconstruction_errors(X)
        ratio = mse / self._train_p99
        # Map 1.0x p99 benign error -> ~0.50; higher ratios approach 1.0
        return 1.0 / (1.0 + np.exp(-2.2 * (ratio - 1.0)))


@dataclass
class ModelSuite:
    """Trained baseline and primary detection models."""

    logistic_baseline: Pipeline
    random_forest: RandomForestClassifier
    lightgbm_clf: lgb.LGBMClassifier
    isolation_forest: IsolationForest
    xgb_binary: xgb.XGBClassifier
    xgb_multiclass: xgb.XGBClassifier
    autoencoder: AutoencoderAnomalyDetector
    fusion_calibrator: LogisticRegression

    def predict_hybrid_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute calibrated hybrid fusion threat probability."""
        p_xgb = self.xgb_binary.predict_proba(X)[:, 1]
        p_ae = self.autoencoder.predict_anomaly_score(X)
        meta_features = np.column_stack([p_xgb, p_ae, p_xgb * p_ae])
        return self.fusion_calibrator.predict_proba(meta_features)[:, 1]


def train_model_suite(
    X_train: np.ndarray,
    y_multi_train: np.ndarray,
    X_val: np.ndarray,
    y_multi_val: np.ndarray,
    seed: int = 42,
) -> ModelSuite:
    """Train baselines, XGBoost detectors, Autoencoder, and validation-calibrated fusion."""
    y_bin_train = (y_multi_train > 0).astype(int)
    y_bin_val = (y_multi_val > 0).astype(int)

    pos_weight = float((y_bin_train == 0).sum() / max((y_bin_train == 1).sum(), 1))

    logistic_baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )
    logistic_baseline.fit(X_train, y_bin_train)

    random_forest = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_bin_train)

    lightgbm_clf = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=seed,
        verbose=-1,
    )
    lightgbm_clf.fit(X_train, y_bin_train)

    isolation_forest = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=seed,
        n_jobs=-1,
    )
    isolation_forest.fit(X_train[y_bin_train == 0])

    xgb_binary = xgb.XGBClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_binary.fit(X_train, y_bin_train, eval_set=[(X_val, y_bin_val)], verbose=False)

    xgb_multiclass = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_multiclass.fit(X_train, y_multi_train, eval_set=[(X_val, y_multi_val)], verbose=False)

    autoencoder = AutoencoderAnomalyDetector(input_dim=X_train.shape[1], epochs=25, seed=seed)
    autoencoder.fit(X_train, y_bin_train)

    # Calibrate fusion on validation set strictly
    p_xgb_val = xgb_binary.predict_proba(X_val)[:, 1]
    p_ae_val = autoencoder.predict_anomaly_score(X_val)
    meta_val = np.column_stack([p_xgb_val, p_ae_val, p_xgb_val * p_ae_val])
    fusion_calibrator = LogisticRegression(random_state=seed)
    fusion_calibrator.fit(meta_val, y_bin_val)

    return ModelSuite(
        logistic_baseline=logistic_baseline,
        random_forest=random_forest,
        lightgbm_clf=lightgbm_clf,
        isolation_forest=isolation_forest,
        xgb_binary=xgb_binary,
        xgb_multiclass=xgb_multiclass,
        autoencoder=autoencoder,
        fusion_calibrator=fusion_calibrator,
    )
