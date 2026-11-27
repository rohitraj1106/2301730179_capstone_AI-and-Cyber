"""Adversarial ML evasion simulator and adversarially hardened defense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from sentinelmesh.config import ENGINEERED_FEATURES
from sentinelmesh.models import AutoencoderAnomalyDetector


# Monotonic constraints: +1 enforces that higher risk signal cannot lower predicted risk
MONOTONE_CONSTRAINTS: Dict[str, int] = {
    "auth_failures_5m": 1,
    "mfa_push_count_5m": 1,
    "geo_velocity_kmph": 1,
    "new_asn_flag": 1,
    "new_device_flag": 1,
    "ja3_rarity_score": 1,
    "process_lineage_rarity": 1,
    "admin_cli_burst_15m": 1,
    "internal_port_fanout_15m": 1,
    "file_rename_entropy": 1,
    "bytes_out_mb_1h": 1,
    "egress_ingress_ratio": 1,
    "dest_domain_entropy": 1,
    "beacon_periodicity_score": 1,
    "nxdomain_ratio_1h": 1,
    "cloud_iam_sensitive_calls_15m": 1,
    "cross_account_hops": 1,
    "user_bytes_out_zscore": 1,
    "dept_admin_burst_zscore": 1,
    "identity_risk_compound": 1,
}


def generate_constrained_evasion(
    X_df: pd.DataFrame,
    y_binary: np.ndarray,
    strength: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate realistic functionality-preserving evasion attacks on malicious rows.

    Attackers throttle noisy volume/timing features (slow-drip exfiltration,
    beacon jitter, residential proxy ASN masking, off-hours blending) while
    retaining the core invariant artifacts required for the attack to succeed.
    """
    rng = np.random.default_rng(seed)
    X_adv = X_df.copy()
    mal = y_binary == 1
    n_mal = int(mal.sum())
    if n_mal == 0:
        return X_adv

    s = float(np.clip(strength, 0.1, 1.0))

    # 1. Slow-and-low throttling of auth & exfiltration volume
    X_adv.loc[mal, "auth_failures_5m"] = np.maximum(
        0.0, X_adv.loc[mal, "auth_failures_5m"] * (1.0 - 0.55 * s)
    )
    X_adv.loc[mal, "mfa_push_count_5m"] = np.maximum(
        0.0, X_adv.loc[mal, "mfa_push_count_5m"] * (1.0 - 0.60 * s)
    )
    X_adv.loc[mal, "bytes_out_mb_1h"] = np.maximum(
        1.0, X_adv.loc[mal, "bytes_out_mb_1h"] * (1.0 - 0.50 * s)
    )
    X_adv.loc[mal, "egress_ingress_ratio"] = np.maximum(
        0.15, X_adv.loc[mal, "egress_ingress_ratio"] * (1.0 - 0.45 * s)
    )
    X_adv.loc[mal, "user_bytes_out_zscore"] = np.maximum(
        -0.5, X_adv.loc[mal, "user_bytes_out_zscore"] * (1.0 - 0.50 * s)
    )

    # 2. C2 jitter & business-hour blending
    X_adv.loc[mal, "beacon_periodicity_score"] = np.clip(
        X_adv.loc[mal, "beacon_periodicity_score"] - 0.34 * s, 0.05, 1.0
    )
    X_adv.loc[mal, "off_hours_score"] = np.clip(
        X_adv.loc[mal, "off_hours_score"] * (1.0 - 0.65 * s), 0.0, 1.0
    )

    # 3. Residential proxy / stolen session cookie masking
    mask_asn = rng.random(n_mal) < (0.65 * s)
    asn_vals = X_adv.loc[mal, "new_asn_flag"].to_numpy(copy=True)
    asn_vals[mask_asn] = 0.0
    X_adv.loc[mal, "new_asn_flag"] = asn_vals

    X_adv.loc[mal, "geo_velocity_kmph"] = np.maximum(
        15.0, X_adv.loc[mal, "geo_velocity_kmph"] * (1.0 - 0.72 * s)
    )
    X_adv.loc[mal, "ja3_rarity_score"] = np.clip(
        X_adv.loc[mal, "ja3_rarity_score"] - 0.25 * s, 0.05, 1.0
    )

    # 4. Dampened lateral bursts (invariant lineage/IAM calls still remain partially elevated)
    X_adv.loc[mal, "admin_cli_burst_15m"] = np.maximum(
        0.0, X_adv.loc[mal, "admin_cli_burst_15m"] * (1.0 - 0.40 * s)
    )
    X_adv.loc[mal, "dept_admin_burst_zscore"] = np.maximum(
        -0.5, X_adv.loc[mal, "dept_admin_burst_zscore"] * (1.0 - 0.40 * s)
    )
    X_adv.loc[mal, "internal_port_fanout_15m"] = np.maximum(
        0.0, X_adv.loc[mal, "internal_port_fanout_15m"] * (1.0 - 0.42 * s)
    )
    X_adv.loc[mal, "dest_domain_entropy"] = np.maximum(
        2.4, X_adv.loc[mal, "dest_domain_entropy"] - 0.45 * s
    )

    # Recompute derived compound identity risk consistently
    X_adv.loc[mal, "identity_risk_compound"] = np.round(
        (
            0.4 * X_adv.loc[mal, "new_asn_flag"]
            + 0.35 * X_adv.loc[mal, "new_device_flag"]
            + 0.25 * X_adv.loc[mal, "off_hours_score"]
        )
        * (1.0 + 0.1 * X_adv.loc[mal, "mfa_push_count_5m"]),
        4,
    )
    return X_adv


@dataclass
class HardenedDetector:
    """Adversarially trained + monotonic-constrained XGBoost fused with Benign Autoencoder."""

    xgb_hardened: xgb.XGBClassifier
    autoencoder: AutoencoderAnomalyDetector
    fusion_calibrator: LogisticRegression

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        p_xgb = self.xgb_hardened.predict_proba(arr)[:, 1]
        p_ae = self.autoencoder.predict_anomaly_score(arr)
        # Disagreement-aware fusion: high AE error preserves anomaly alert under XGB evasion
        meta = np.column_stack([p_xgb, p_ae, np.maximum(p_xgb, 0.85 * p_ae)])
        return self.fusion_calibrator.predict_proba(meta)[:, 1]


def train_hardened_detector(
    X_train_df: pd.DataFrame,
    y_bin_train: np.ndarray,
    X_val_df: pd.DataFrame,
    y_bin_val: np.ndarray,
    autoencoder: AutoencoderAnomalyDetector,
    seed: int = 42,
) -> HardenedDetector:
    """Train adversarially augmented + monotonic-constrained detector."""
    adv_mid = generate_constrained_evasion(X_train_df, y_bin_train, strength=0.6, seed=seed + 1)
    adv_full = generate_constrained_evasion(X_train_df, y_bin_train, strength=1.0, seed=seed + 2)

    mal_mask = y_bin_train == 1
    X_aug = pd.concat(
        [X_train_df, adv_mid.loc[mal_mask], adv_full.loc[mal_mask]],
        axis=0,
        ignore_index=True,
    )
    y_aug = np.concatenate([y_bin_train, y_bin_train[mal_mask], y_bin_train[mal_mask]])

    pos_weight = float((y_aug == 0).sum() / max((y_aug == 1).sum(), 1))
    constraints = tuple(MONOTONE_CONSTRAINTS.get(f, 0) for f in ENGINEERED_FEATURES)

    xgb_hardened = xgb.XGBClassifier(
        n_estimators=240,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        monotone_constraints=constraints,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_hardened.fit(X_aug.to_numpy(), y_aug, eval_set=[(X_val_df.to_numpy(), y_bin_val)], verbose=False)

    # Calibrate on mixed clean + evasion validation set
    X_val_adv = generate_constrained_evasion(X_val_df, y_bin_val, strength=0.85, seed=seed + 3)
    X_val_mix = pd.concat([X_val_df, X_val_adv.loc[y_bin_val == 1]], axis=0, ignore_index=True)
    y_val_mix = np.concatenate([y_bin_val, y_bin_val[y_bin_val == 1]])

    p_xgb_val = xgb_hardened.predict_proba(X_val_mix.to_numpy())[:, 1]
    p_ae_val = autoencoder.predict_anomaly_score(X_val_mix.to_numpy())
    meta_val = np.column_stack([p_xgb_val, p_ae_val, np.maximum(p_xgb_val, 0.85 * p_ae_val)])

    cal = LogisticRegression(random_state=seed)
    cal.fit(meta_val, y_val_mix)

    return HardenedDetector(xgb_hardened=xgb_hardened, autoencoder=autoencoder, fusion_calibrator=cal)


def evaluate_adversarial_robustness(
    baseline_xgb: xgb.XGBClassifier,
    hardened: HardenedDetector,
    X_test_df: pd.DataFrame,
    y_bin_test: np.ndarray,
    threshold: float = 0.50,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Measure clean vs. adversarial evasion performance for baseline and hardened models."""
    X_evasion_df = generate_constrained_evasion(X_test_df, y_bin_test, strength=1.0, seed=seed + 9)

    X_clean = X_test_df.to_numpy()
    X_ev = X_evasion_df.to_numpy()

    def _metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        preds = (probs >= threshold).astype(int)
        fpr = float(((preds == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
        return {
            "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
            "recall_tpr": round(float(recall_score(y_true, preds)), 4),
            "fpr": round(fpr, 4),
        }

    return {
        "baseline_xgb_clean": _metrics(baseline_xgb.predict_proba(X_clean)[:, 1], y_bin_test),
        "baseline_xgb_under_evasion": _metrics(baseline_xgb.predict_proba(X_ev)[:, 1], y_bin_test),
        "hardened_hybrid_clean": _metrics(hardened.predict_proba(X_clean), y_bin_test),
        "hardened_hybrid_under_evasion": _metrics(hardened.predict_proba(X_ev), y_bin_test),
    }
