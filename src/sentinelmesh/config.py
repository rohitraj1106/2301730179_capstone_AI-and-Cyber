"""Configuration, enterprise profile, threat taxonomy, and feature specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


ENTERPRISE_PROFILE = {
    "name": "Meridian FinTech Inc.",
    "employees": 500,
    "endpoints": 650,
    "departments": ["Engineering", "Finance", "Sales_CS", "Operations", "Security_IT"],
    "department_weights": [0.46, 0.14, 0.20, 0.14, 0.06],
    "cloud_stack": ["AWS EKS & IAM", "GCP BigQuery", "Okta SSO", "Microsoft 365", "GitHub Enterprise"],
    "daily_log_volume_gb": 6.4,
    "avg_events_per_sec": 1850,
    "peak_events_per_sec": 7800,
}

THREAT_TAXONOMY: Dict[int, Dict[str, str]] = {
    0: {
        "code": "BENIGN",
        "name": "Benign Enterprise Activity",
        "mitre": "N/A",
        "severity": "None",
    },
    1: {
        "code": "T1_ATO",
        "name": "Credential Stuffing / AiTM Phishing & Account Takeover",
        "mitre": "T1078, T1110, T1557",
        "severity": "High",
    },
    2: {
        "code": "T2_LOTL_RANSOM",
        "name": "Living-off-the-Land Lateral Movement & Pre-Ransomware",
        "mitre": "T1059, T1021, T1486",
        "severity": "Critical",
    },
    3: {
        "code": "T3_EXFIL",
        "name": "Stealthy Data Exfiltration & Staging",
        "mitre": "T1041, T1048, T1567",
        "severity": "Critical",
    },
    4: {
        "code": "T4_CLOUD_PRIVESC",
        "name": "Cloud Control-Plane Privilege Escalation & IAM Abuse",
        "mitre": "T1098, T1548, T1078.004",
        "severity": "Critical",
    },
    5: {
        "code": "T5_C2_BEACON",
        "name": "Encrypted C2 Beaconing & Malware Callback",
        "mitre": "T1071, T1573",
        "severity": "High",
    },
}

RAW_TELEMETRY_COLUMNS: List[str] = [
    "event_id",
    "timestamp_hour",
    "day_index",
    "user_id",
    "department",
    "role_tier",
    "travel_eligible",
    "on_call_eligible",
    "asset_criticality",
    "auth_failures_5m",
    "mfa_push_count_5m",
    "geo_velocity_kmph",
    "new_asn_flag",
    "new_device_flag",
    "ja3_rarity_score",
    "off_hours_score",
    "process_lineage_rarity",
    "admin_cli_burst_15m",
    "internal_port_fanout_15m",
    "file_rename_entropy",
    "bytes_out_mb_1h",
    "bytes_in_mb_1h",
    "dest_domain_entropy",
    "cloud_iam_sensitive_calls_15m",
    "cross_account_hops",
    "beacon_periodicity_score",
    "nxdomain_ratio_1h",
    "threat_type",
]

ENGINEERED_FEATURES: List[str] = [
    # Identity & Auth Velocity
    "auth_failures_5m",
    "mfa_push_count_5m",
    "geo_velocity_kmph",
    "new_asn_flag",
    "new_device_flag",
    "ja3_rarity_score",
    "off_hours_score",
    # Endpoint & Lateral Movement
    "process_lineage_rarity",
    "admin_cli_burst_15m",
    "internal_port_fanout_15m",
    "file_rename_entropy",
    # Exfiltration & Network Metadata
    "bytes_out_mb_1h",
    "egress_ingress_ratio",
    "dest_domain_entropy",
    "beacon_periodicity_score",
    "nxdomain_ratio_1h",
    # Cloud Control Plane
    "cloud_iam_sensitive_calls_15m",
    "cross_account_hops",
    # Leak-Free UEBA Peer-Group & Entity Baselines (fitted strictly on train split)
    "user_bytes_out_zscore",
    "dept_admin_burst_zscore",
    "asset_criticality",
    # Composite Context Flags & Missingness Indicators
    "identity_risk_compound",
    "geo_velocity_missing_flag",
    "bytes_out_missing_flag",
]

# Non-technical stakeholder readable descriptions for every feature
FEATURE_BUSINESS_LABELS: Dict[str, str] = {
    "auth_failures_5m": "Rapid repeated login failures within 5 minutes",
    "mfa_push_count_5m": "Burst of MFA approval prompts (MFA fatigue indicator)",
    "geo_velocity_kmph": "Impossible travel speed between consecutive logins",
    "new_asn_flag": "Sign-in from an unfamiliar internet provider / cloud proxy",
    "new_device_flag": "Access attempted from an unrecognized device",
    "ja3_rarity_score": "Unusual encrypted browser/client TLS fingerprint",
    "off_hours_score": "Activity occurring outside the employee's normal working hours",
    "process_lineage_rarity": "Unusual parent-to-child system command execution chain",
    "admin_cli_burst_15m": "Spike in administrative scripting commands (PowerShell/CLI)",
    "internal_port_fanout_15m": "Device scanning or probing multiple internal servers",
    "file_rename_entropy": "High-rate encrypted/randomized file renaming (ransomware sign)",
    "bytes_out_mb_1h": "Large outbound data transfer volume in the past hour",
    "egress_ingress_ratio": "Disproportionately high outbound vs. inbound data ratio",
    "dest_domain_entropy": "Data sent to an algorithmically generated / suspicious domain",
    "beacon_periodicity_score": "Clockwork-like automated heartbeat callbacks to external host",
    "nxdomain_ratio_1h": "High rate of failed DNS lookups (malware domain probing)",
    "cloud_iam_sensitive_calls_15m": "Spike in sensitive cloud permission/role changes",
    "cross_account_hops": "Unusual role-switching across multiple cloud accounts",
    "user_bytes_out_zscore": "Outbound transfer far above this employee's 30-day norm",
    "dept_admin_burst_zscore": "Admin command volume far above department peers",
    "asset_criticality": "Sensitivity tier of the targeted FinTech asset",
    "identity_risk_compound": "Combined unusual network + unrecognized device + off-hours login",
    "geo_velocity_missing_flag": "Location telemetry suppressed or unavailable (VPN/anonymizer)",
    "bytes_out_missing_flag": "Network flow byte telemetry incomplete/dropped",
}


@dataclass(frozen=True)
class SOARConfig:
    """Confidence-gated automated response configuration."""

    medium_threshold: float = 0.40
    high_threshold: float = 0.70
    critical_threshold: float = 0.85
    circuit_breaker_max_isolations_per_window: int = 5
    hmac_secret_env: str = "SENTINELMESH_HMAC_KEY"
