"""Synthetic hybrid-cloud FinTech enterprise telemetry generator.

Generates 60 days of realistic event-window telemetry for ~500 employees across
five departments, including benign behavioral variations (travel, on-call spikes,
missing telemetry) and five MITRE-mapped threat families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sentinelmesh.config import ENTERPRISE_PROFILE, RAW_TELEMETRY_COLUMNS


@dataclass(frozen=True)
class ChronologicalSplits:
    """Chronological train/validation/test splits preventing temporal leakage."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def generate_enterprise_telemetry(
    n_events: int = 18000,
    n_users: int = 500,
    n_days: int = 60,
    attack_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic hybrid-cloud enterprise telemetry dataset."""
    rng = np.random.default_rng(seed)

    departments = ENTERPRISE_PROFILE["departments"]
    dept_weights = ENTERPRISE_PROFILE["department_weights"]

    # Build persistent employee directory with realistic role profiles
    user_ids = np.array([f"emp_{i:04d}" for i in range(n_users)])
    user_depts = rng.choice(departments, size=n_users, p=dept_weights)
    user_role_tier = rng.choice([1, 2, 3], size=n_users, p=[0.65, 0.25, 0.10])
    user_travel_eligible = np.where(
        np.isin(user_depts, ["Sales_CS", "Operations"]) | (user_role_tier == 3),
        rng.random(n_users) < 0.55,
        rng.random(n_users) < 0.08,
    )
    user_on_call_eligible = np.where(
        np.isin(user_depts, ["Engineering", "Security_IT"]),
        rng.random(n_users) < 0.45,
        rng.random(n_users) < 0.05,
    )
    user_base_bytes = np.where(
        user_depts == "Engineering",
        rng.lognormal(mean=2.3, sigma=0.55, size=n_users),
        np.where(
            user_depts == "Finance",
            rng.lognormal(mean=1.8, sigma=0.45, size=n_users),
            rng.lognormal(mean=1.5, sigma=0.50, size=n_users),
        ),
    )

    # Sample chronological timestamps across n_days
    day_index = np.sort(rng.integers(0, n_days, size=n_events))
    hour_of_day = rng.integers(0, 24, size=n_events)
    timestamp_hour = day_index * 24.0 + hour_of_day + rng.uniform(0, 0.99, size=n_events)

    u_idx = rng.integers(0, n_users, size=n_events)
    dept = user_depts[u_idx]
    role_tier = user_role_tier[u_idx]
    travel_eligible = user_travel_eligible[u_idx].astype(int)
    on_call_eligible = user_on_call_eligible[u_idx].astype(int)
    base_bytes = user_base_bytes[u_idx]

    # Asset criticality higher for Finance, Security_IT, and privileged roles
    asset_criticality = np.clip(
        np.where(np.isin(dept, ["Finance", "Security_IT"]), 0.65, 0.35)
        + 0.12 * (role_tier - 1)
        + rng.normal(0.0, 0.10, size=n_events),
        0.1,
        1.0,
    )

    # Assign labels: 0 = benign, 1..5 = 5 threat families
    is_attack = rng.random(n_events) < attack_rate
    threat_type = np.zeros(n_events, dtype=int)
    n_attacks = int(is_attack.sum())
    # Realistic prevalence across the top-5 FinTech threats
    threat_type[is_attack] = rng.choice(
        [1, 2, 3, 4, 5],
        size=n_attacks,
        p=[0.28, 0.18, 0.22, 0.16, 0.16],
    )

    # --- Benign baseline distributions (with realistic hard edge cases) ---
    auth_failures_5m = rng.poisson(0.25, size=n_events).astype(float)
    mfa_push_count_5m = rng.binomial(1, 0.25, size=n_events).astype(float)

    # Traveling users occasionally have legitimate high geo velocity & new ASNs
    benign_travel_spike = (travel_eligible == 1) & (rng.random(n_events) < 0.09)
    geo_velocity_kmph = np.where(
        benign_travel_spike,
        rng.uniform(250, 880, size=n_events),
        rng.exponential(18.0, size=n_events),
    )
    new_asn_flag = (
        (rng.random(n_events) < np.where(benign_travel_spike, 0.55, 0.04))
    ).astype(float)
    new_device_flag = (rng.random(n_events) < 0.03).astype(float)
    ja3_rarity_score = np.clip(rng.beta(1.3, 9.0, size=n_events), 0.0, 1.0)

    # On-call engineers legitimately work off-hours and run admin CLIs
    off_hours_raw = np.where((hour_of_day < 7) | (hour_of_day > 19), 1.0, 0.0)
    off_hours_score = np.clip(
        off_hours_raw * np.where(on_call_eligible == 1, 0.55, 0.90)
        + rng.uniform(0.0, 0.15, size=n_events),
        0.0,
        1.0,
    )

    process_lineage_rarity = np.clip(rng.beta(1.2, 8.5, size=n_events), 0.0, 1.0)
    admin_cli_mean = np.where(
        np.isin(dept, ["Engineering", "Security_IT"]),
        np.where(on_call_eligible == 1, 3.2, 1.8),
        0.25,
    )
    admin_cli_burst_15m = rng.poisson(admin_cli_mean).astype(float)
    internal_port_fanout_15m = rng.poisson(
        np.where(dept == "Security_IT", 2.4, 0.9), size=n_events
    ).astype(float)
    file_rename_entropy = np.clip(rng.beta(1.4, 7.0, size=n_events), 0.0, 1.0)

    bytes_out_mb_1h = np.clip(base_bytes * rng.lognormal(0.0, 0.45, size=n_events), 0.1, 2000.0)
    bytes_in_mb_1h = np.clip(
        bytes_out_mb_1h * rng.uniform(1.4, 6.5, size=n_events), 0.5, 4000.0
    )
    dest_domain_entropy = np.clip(rng.normal(2.55, 0.42, size=n_events), 1.0, 4.9)
    cloud_iam_sensitive_calls_15m = rng.poisson(
        np.where(np.isin(dept, ["Engineering", "Security_IT"]), 0.45, 0.05), size=n_events
    ).astype(float)
    cross_account_hops = rng.poisson(
        np.where(dept == "Engineering", 0.35, 0.04), size=n_events
    ).astype(float)
    beacon_periodicity_score = np.clip(rng.beta(1.3, 7.5, size=n_events), 0.0, 1.0)
    nxdomain_ratio_1h = np.clip(rng.beta(1.1, 18.0, size=n_events), 0.0, 1.0)

    # Inject realistic "hard benign" enterprise operational spikes (creates realistic decision boundaries)
    benign_mask = threat_type == 0
    # 1) Legitimate quarterly BigQuery/Finance exports & CI/CD artifact uploads
    benign_export_spike = benign_mask & (rng.random(n_events) < 0.045)
    bytes_out_mb_1h[benign_export_spike] *= rng.uniform(4.5, 11.0, size=int(benign_export_spike.sum()))
    bytes_in_mb_1h[benign_export_spike] *= rng.uniform(0.35, 0.85, size=int(benign_export_spike.sum()))
    off_hours_score[benign_export_spike] = np.clip(
        off_hours_score[benign_export_spike] + rng.uniform(0.25, 0.55, size=int(benign_export_spike.sum())),
        0.0,
        1.0,
    )

    # 2) Legitimate DevOps / Security_IT incident response & patch deployment bursts
    benign_ops_burst = benign_mask & np.isin(dept, ["Engineering", "Security_IT"]) & (rng.random(n_events) < 0.05)
    n_ops = int(benign_ops_burst.sum())
    if n_ops > 0:
        admin_cli_burst_15m[benign_ops_burst] += rng.poisson(7.5, size=n_ops)
        internal_port_fanout_15m[benign_ops_burst] += rng.poisson(8.0, size=n_ops)
        cloud_iam_sensitive_calls_15m[benign_ops_burst] += rng.poisson(3.2, size=n_ops)
        cross_account_hops[benign_ops_burst] += rng.poisson(2.0, size=n_ops)
        process_lineage_rarity[benign_ops_burst] = np.clip(
            process_lineage_rarity[benign_ops_burst] + rng.uniform(0.25, 0.45, size=n_ops), 0.0, 1.0
        )

    # 3) Legitimate mobile / airport VPN MFA retries
    benign_mfa_jitter = benign_mask & (rng.random(n_events) < 0.035)
    n_mj = int(benign_mfa_jitter.sum())
    if n_mj > 0:
        auth_failures_5m[benign_mfa_jitter] += rng.poisson(3.5, size=n_mj)
        mfa_push_count_5m[benign_mfa_jitter] += rng.poisson(2.2, size=n_mj)
        new_asn_flag[benign_mfa_jitter] = 1.0
        ja3_rarity_score[benign_mfa_jitter] = np.clip(
            ja3_rarity_score[benign_mfa_jitter] + rng.uniform(0.20, 0.40, size=n_mj), 0.0, 1.0
        )

    # --- Inject Realistic Threat Signatures (with stealthy/low-and-slow overlap) ---
    # T1: Credential Stuffing / AiTM Phishing & Account Takeover
    t1 = threat_type == 1
    n1 = int(t1.sum())
    if n1 > 0:
        auth_failures_5m[t1] = rng.poisson(4.5, size=n1) + 1
        mfa_push_count_5m[t1] = rng.poisson(2.8, size=n1) + 1
        geo_velocity_kmph[t1] = rng.uniform(350, 3200, size=n1)
        new_asn_flag[t1] = (rng.random(n1) < 0.78).astype(float)
        new_device_flag[t1] = (rng.random(n1) < 0.72).astype(float)
        ja3_rarity_score[t1] = np.clip(rng.normal(0.64, 0.19, size=n1), 0.15, 1.0)
        off_hours_score[t1] = np.clip(rng.normal(0.68, 0.22, size=n1), 0.05, 1.0)

    # T2: LotL Lateral Movement & Pre-Ransomware
    t2 = threat_type == 2
    n2 = int(t2.sum())
    if n2 > 0:
        process_lineage_rarity[t2] = np.clip(rng.normal(0.75, 0.16, size=n2), 0.25, 1.0)
        admin_cli_burst_15m[t2] = rng.poisson(10.0, size=n2) + 3
        internal_port_fanout_15m[t2] = rng.poisson(13.0, size=n2) + 3
        file_rename_entropy[t2] = np.clip(rng.normal(0.77, 0.15, size=n2), 0.25, 1.0)
        off_hours_score[t2] = np.clip(rng.normal(0.65, 0.23, size=n2), 0.05, 1.0)

    # T3: Stealthy Data Exfiltration & Staging
    t3 = threat_type == 3
    n3 = int(t3.sum())
    if n3 > 0:
        bytes_out_mb_1h[t3] = np.clip(
            base_bytes[t3] * rng.uniform(5.0, 18.0, size=n3), 25.0, 4000.0
        )
        bytes_in_mb_1h[t3] = np.clip(bytes_out_mb_1h[t3] * rng.uniform(0.12, 0.55, size=n3), 1.0, 900.0)
        dest_domain_entropy[t3] = np.clip(rng.normal(3.62, 0.50, size=n3), 2.1, 5.0)
        off_hours_score[t3] = np.clip(rng.normal(0.66, 0.24, size=n3), 0.05, 1.0)
        new_asn_flag[t3] = (rng.random(n3) < 0.40).astype(float)

    # T4: Cloud Control-Plane Privilege Escalation & IAM Abuse
    t4 = threat_type == 4
    n4 = int(t4.sum())
    if n4 > 0:
        cloud_iam_sensitive_calls_15m[t4] = rng.poisson(6.0, size=n4) + 2
        cross_account_hops[t4] = rng.poisson(3.2, size=n4) + 1
        ja3_rarity_score[t4] = np.clip(rng.normal(0.62, 0.19, size=n4), 0.15, 1.0)
        new_asn_flag[t4] = (rng.random(n4) < 0.65).astype(float)
        admin_cli_burst_15m[t4] = rng.poisson(5.5, size=n4) + 1

    # T5: Encrypted C2 Beaconing & Malware Callback
    t5 = threat_type == 5
    n5 = int(t5.sum())
    if n5 > 0:
        beacon_periodicity_score[t5] = np.clip(rng.normal(0.79, 0.14, size=n5), 0.30, 1.0)
        nxdomain_ratio_1h[t5] = np.clip(rng.normal(0.36, 0.16, size=n5), 0.05, 0.90)
        dest_domain_entropy[t5] = np.clip(rng.normal(3.85, 0.44, size=n5), 2.4, 5.0)
        ja3_rarity_score[t5] = np.clip(rng.normal(0.68, 0.17, size=n5), 0.20, 1.0)
        process_lineage_rarity[t5] = np.clip(rng.normal(0.56, 0.20, size=n5), 0.10, 1.0)

    # Inject realistic missingness (VPN suppressed geo, dropped UDP flow byte counters)
    geo_missing_mask = rng.random(n_events) < 0.03
    bytes_missing_mask = rng.random(n_events) < 0.02
    geo_velocity_kmph = geo_velocity_kmph.astype(float)
    bytes_out_mb_1h = bytes_out_mb_1h.astype(float)
    geo_velocity_kmph[geo_missing_mask] = np.nan
    bytes_out_mb_1h[bytes_missing_mask] = np.nan

    df = pd.DataFrame(
        {
            "event_id": [f"evt_{i:06d}" for i in range(n_events)],
            "timestamp_hour": np.round(timestamp_hour, 3),
            "day_index": day_index,
            "user_id": user_ids[u_idx],
            "department": dept,
            "role_tier": role_tier,
            "travel_eligible": travel_eligible,
            "on_call_eligible": on_call_eligible,
            "asset_criticality": np.round(asset_criticality, 4),
            "auth_failures_5m": auth_failures_5m,
            "mfa_push_count_5m": mfa_push_count_5m,
            "geo_velocity_kmph": np.round(geo_velocity_kmph, 2),
            "new_asn_flag": new_asn_flag,
            "new_device_flag": new_device_flag,
            "ja3_rarity_score": np.round(ja3_rarity_score, 4),
            "off_hours_score": np.round(off_hours_score, 4),
            "process_lineage_rarity": np.round(process_lineage_rarity, 4),
            "admin_cli_burst_15m": admin_cli_burst_15m,
            "internal_port_fanout_15m": internal_port_fanout_15m,
            "file_rename_entropy": np.round(file_rename_entropy, 4),
            "bytes_out_mb_1h": np.round(bytes_out_mb_1h, 3),
            "bytes_in_mb_1h": np.round(bytes_in_mb_1h, 3),
            "dest_domain_entropy": np.round(dest_domain_entropy, 4),
            "cloud_iam_sensitive_calls_15m": cloud_iam_sensitive_calls_15m,
            "cross_account_hops": cross_account_hops,
            "beacon_periodicity_score": np.round(beacon_periodicity_score, 4),
            "nxdomain_ratio_1h": np.round(nxdomain_ratio_1h, 4),
            "threat_type": threat_type,
        }
    )
    return df[RAW_TELEMETRY_COLUMNS]


def chronological_split(
    df: pd.DataFrame, train_days: int = 36, val_days: int = 12
) -> ChronologicalSplits:
    """Split telemetry chronologically by day_index to prevent future leakage."""
    ordered = df.sort_values("timestamp_hour").reset_index(drop=True)
    train_df = ordered[ordered["day_index"] < train_days].reset_index(drop=True)
    val_df = ordered[
        (ordered["day_index"] >= train_days) & (ordered["day_index"] < train_days + val_days)
    ].reset_index(drop=True)
    test_df = ordered[ordered["day_index"] >= train_days + val_days].reset_index(drop=True)
    return ChronologicalSplits(train=train_df, val=val_df, test=test_df)
