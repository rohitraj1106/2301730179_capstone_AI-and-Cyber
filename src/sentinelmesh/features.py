"""Leak-free feature engineering, missing-value handling, and PII pseudonymization."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sentinelmesh.config import ENGINEERED_FEATURES


def pseudonymize_identifier(raw_id: str, secret_key: str | None = None) -> str:
    """Compute a deterministic HMAC-SHA256 pseudonym for PII-safe analytics."""
    key = (secret_key or os.environ.get("SENTINELMESH_HMAC_KEY") or "meridian-fintech-dev-key").encode(
        "utf-8"
    )
    digest = hmac.new(key, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"usr_{digest[:16]}"


class FeatureTransformer:
    """Fits UEBA entity/peer baselines and missing-value statistics strictly on train data."""

    def __init__(self, hmac_key: str | None = None) -> None:
        self.hmac_key = hmac_key
        self._geo_median: float = 0.0
        self._bytes_out_median: float = 1.0
        self._user_bytes_stats: Dict[str, Tuple[float, float]] = {}
        self._global_bytes_mean: float = 10.0
        self._global_bytes_std: float = 10.0
        self._dept_admin_stats: Dict[str, Tuple[float, float]] = {}
        self._global_admin_mean: float = 1.0
        self._global_admin_std: float = 1.5
        self._fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureTransformer":
        """Learn medians and benign entity/department baselines from training split only."""
        self._geo_median = float(train_df["geo_velocity_kmph"].median(skipna=True))
        self._bytes_out_median = float(train_df["bytes_out_mb_1h"].median(skipna=True))

        benign_train = train_df[train_df["threat_type"] == 0]
        if benign_train.empty:
            benign_train = train_df

        bytes_filled = benign_train["bytes_out_mb_1h"].fillna(self._bytes_out_median)
        self._global_bytes_mean = float(bytes_filled.mean())
        self._global_bytes_std = max(float(bytes_filled.std(ddof=0)), 1.0)

        # Compute per-user baseline over pseudonymized IDs
        pseudo_ids = benign_train["user_id"].map(
            lambda u: pseudonymize_identifier(str(u), self.hmac_key)
        )
        user_grp = (
            pd.DataFrame({"uid": pseudo_ids, "bytes_out": bytes_filled})
            .groupby("uid")["bytes_out"]
            .agg(["mean", "std"])
        )
        self._user_bytes_stats = {
            uid: (float(row["mean"]), max(float(0.0 if np.isnan(row["std"]) else row["std"]), 2.5))
            for uid, row in user_grp.iterrows()
        }

        self._global_admin_mean = float(benign_train["admin_cli_burst_15m"].mean())
        self._global_admin_std = max(float(benign_train["admin_cli_burst_15m"].std(ddof=0)), 0.75)

        dept_grp = benign_train.groupby("department")["admin_cli_burst_15m"].agg(["mean", "std"])
        self._dept_admin_stats = {
            dept: (float(row["mean"]), max(float(0.0 if np.isnan(row["std"]) else row["std"]), 0.75))
            for dept, row in dept_grp.iterrows()
        }

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw telemetry into the engineered ML feature matrix."""
        if not self._fitted:
            raise RuntimeError("FeatureTransformer must be fitted on training data before transform.")

        out = pd.DataFrame(index=df.index)

        # Missingness indicator flags (security-meaningful when VPN/telemetry is suppressed)
        geo_missing = df["geo_velocity_kmph"].isna().astype(float)
        bytes_missing = df["bytes_out_mb_1h"].isna().astype(float)

        geo_filled = df["geo_velocity_kmph"].fillna(self._geo_median)
        bytes_out_filled = df["bytes_out_mb_1h"].fillna(self._bytes_out_median)
        bytes_in_safe = np.maximum(df["bytes_in_mb_1h"].fillna(1.0).to_numpy(), 0.5)

        out["auth_failures_5m"] = df["auth_failures_5m"].astype(float)
        out["mfa_push_count_5m"] = df["mfa_push_count_5m"].astype(float)
        out["geo_velocity_kmph"] = geo_filled.astype(float)
        out["new_asn_flag"] = df["new_asn_flag"].astype(float)
        out["new_device_flag"] = df["new_device_flag"].astype(float)
        out["ja3_rarity_score"] = df["ja3_rarity_score"].astype(float)
        out["off_hours_score"] = df["off_hours_score"].astype(float)
        out["process_lineage_rarity"] = df["process_lineage_rarity"].astype(float)
        out["admin_cli_burst_15m"] = df["admin_cli_burst_15m"].astype(float)
        out["internal_port_fanout_15m"] = df["internal_port_fanout_15m"].astype(float)
        out["file_rename_entropy"] = df["file_rename_entropy"].astype(float)
        out["bytes_out_mb_1h"] = bytes_out_filled.astype(float)
        out["egress_ingress_ratio"] = np.round(bytes_out_filled.to_numpy() / bytes_in_safe, 4)
        out["dest_domain_entropy"] = df["dest_domain_entropy"].astype(float)
        out["beacon_periodicity_score"] = df["beacon_periodicity_score"].astype(float)
        out["nxdomain_ratio_1h"] = df["nxdomain_ratio_1h"].astype(float)
        out["cloud_iam_sensitive_calls_15m"] = df["cloud_iam_sensitive_calls_15m"].astype(float)
        out["cross_account_hops"] = df["cross_account_hops"].astype(float)

        # UEBA entity & peer-group Z-scores computed against train-split baselines
        pseudo_ids = df["user_id"].map(lambda u: pseudonymize_identifier(str(u), self.hmac_key))
        user_means = np.array(
            [self._user_bytes_stats.get(uid, (self._global_bytes_mean, self._global_bytes_std))[0] for uid in pseudo_ids]
        )
        user_stds = np.array(
            [self._user_bytes_stats.get(uid, (self._global_bytes_mean, self._global_bytes_std))[1] for uid in pseudo_ids]
        )
        out["user_bytes_out_zscore"] = np.clip(
            (bytes_out_filled.to_numpy() - user_means) / user_stds, -3.0, 25.0
        )

        dept_means = np.array(
            [self._dept_admin_stats.get(d, (self._global_admin_mean, self._global_admin_std))[0] for d in df["department"]]
        )
        dept_stds = np.array(
            [self._dept_admin_stats.get(d, (self._global_admin_mean, self._global_admin_std))[1] for d in df["department"]]
        )
        out["dept_admin_burst_zscore"] = np.clip(
            (df["admin_cli_burst_15m"].to_numpy() - dept_means) / dept_stds, -3.0, 25.0
        )

        out["asset_criticality"] = df["asset_criticality"].astype(float)
        out["identity_risk_compound"] = np.round(
            (0.4 * out["new_asn_flag"] + 0.35 * out["new_device_flag"] + 0.25 * out["off_hours_score"])
            * (1.0 + 0.1 * out["mfa_push_count_5m"]),
            4,
        )
        out["geo_velocity_missing_flag"] = geo_missing
        out["bytes_out_missing_flag"] = bytes_missing

        return out[ENGINEERED_FEATURES]

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed training features."""
        return self.fit(train_df).transform(train_df)
