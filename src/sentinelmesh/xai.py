"""Explainable AI (TreeSHAP + Autoencoder Attribution) and Stakeholder Alert Translator."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from sentinelmesh.config import ENGINEERED_FEATURES, FEATURE_BUSINESS_LABELS, THREAT_TAXONOMY
from sentinelmesh.models import AutoencoderAnomalyDetector
from sentinelmesh.soar import SOARDecision


@dataclass
class StakeholderDriver:
    """Plain-language risk driver derived from exact TreeSHAP values."""

    feature_name: str
    business_label: str
    observed_value: float
    shap_contribution: float
    relative_impact_pct: float


@dataclass
class StakeholderAlertBrief:
    """Non-technical stakeholder incident briefing."""

    incident_id: str
    department: str
    pseudonymized_user: str
    threat_category: str
    mitre_techniques: str
    risk_score: float
    confidence_label: str
    headline: str
    executive_summary: str
    top_drivers: List[StakeholderDriver]
    automated_actions_taken: List[str]
    business_risk_mitigated: str
    stakeholder_action_needed: str

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return d


def compute_treeshap_contributions(
    xgb_model: xgb.XGBClassifier, X: np.ndarray | pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact TreeSHAP contributions via XGBoost pred_contribs=True.

    Returns:
        shap_values: array of shape (n_samples, n_features)
        base_values: array of shape (n_samples,)
    """
    arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    dmat = xgb.DMatrix(arr, feature_names=ENGINEERED_FEATURES)
    contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)
    return contribs[:, :-1], contribs[:, -1]


def global_shap_importance(
    xgb_model: xgb.XGBClassifier, X_sample: np.ndarray | pd.DataFrame
) -> pd.DataFrame:
    """Return features ranked by mean absolute TreeSHAP contribution."""
    shap_vals, _ = compute_treeshap_contributions(xgb_model, X_sample)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    df = pd.DataFrame(
        {
            "feature": ENGINEERED_FEATURES,
            "business_label": [FEATURE_BUSINESS_LABELS[f] for f in ENGINEERED_FEATURES],
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return df


def _format_business_risk(threat_id: int, department: str) -> str:
    risk_map = {
        1: (
            f"Prevented unauthorized takeover of a {department} employee account, protecting "
            "corporate SSO, customer financial dashboards, and internal payment workflows."
        ),
        2: (
            f"Stopped potential lateral movement and pre-ransomware file encryption within {department} "
            "systems before production payment services could be disrupted."
        ),
        3: (
            f"Blocked suspected unauthorized bulk export of sensitive {department} / customer records, "
            "mitigating PCI-DSS and GDPR data-breach exposure."
        ),
        4: (
            "Halted anomalous cloud permission escalation that could have granted unauthorized "
            "administrative access to production AWS/GCP environments."
        ),
        5: (
            "Disrupted suspected covert command-and-control malware communication attempting to "
            "maintain persistent remote access inside the corporate network."
        ),
    }
    return risk_map.get(
        threat_id,
        "Mitigated anomalous activity deviating significantly from established enterprise norms.",
    )


def translate_alert_for_stakeholder(
    incident_id: str,
    raw_row: pd.Series,
    feature_row: pd.Series,
    shap_row: np.ndarray,
    ae_detector: AutoencoderAnomalyDetector,
    soar_decision: SOARDecision,
    predicted_threat_id: int,
    top_k: int = 4,
) -> StakeholderAlertBrief:
    """Translate technical SHAP + Autoencoder signals into a plain-English executive brief."""
    threat_meta = THREAT_TAXONOMY.get(predicted_threat_id, THREAT_TAXONOMY[0])
    department = str(raw_row["department"])

    # Focus on positive risk-increasing TreeSHAP drivers for clear explanation
    pos_indices = np.argsort(shap_row)[::-1]
    pos_total = float(np.sum(np.maximum(shap_row, 0.0))) or 1e-6

    drivers: List[StakeholderDriver] = []
    for idx in pos_indices:
        val = float(shap_row[idx])
        if val <= 0 and len(drivers) >= 1:
            continue
        feat = ENGINEERED_FEATURES[idx]
        drivers.append(
            StakeholderDriver(
                feature_name=feat,
                business_label=FEATURE_BUSINESS_LABELS[feat],
                observed_value=round(float(feature_row[feat]), 3),
                shap_contribution=round(val, 4),
                relative_impact_pct=round(max(val, 0.0) / pos_total * 100.0, 1),
            )
        )
        if len(drivers) >= top_k:
            break

    ae_score = float(
        ae_detector.predict_anomaly_score(feature_row.to_numpy().reshape(1, -1))[0]
    )
    confidence_label = (
        "Very High Confidence"
        if soar_decision.risk_score >= 85
        else "High Confidence"
        if soar_decision.risk_score >= 70
        else "Moderate Confidence"
    )

    top_driver_text = "; ".join(
        f"{d.business_label.lower()} ({d.relative_impact_pct}% of risk)" for d in drivers[:2]
    )
    headline = (
        f"[{soar_decision.tier.replace('_', ' ')}] {threat_meta['name']} "
        f"in {department} (Risk Score: {soar_decision.risk_score}/100)"
    )
    executive_summary = (
        f"SentinelMesh AI flagged unusual activity associated with pseudonymized employee account "
        f"{soar_decision.pseudonymized_user} in the {department} department. Compared to this employee's "
        f"30-day normal baseline and department peers (anomaly score {ae_score:.0%}), the system "
        f"detected a strong pattern matching {threat_meta['name']}. Primary drivers: {top_driver_text}."
    )

    action_needed = (
        "Please confirm with the employee's manager whether legitimate urgent business travel or "
        "approved maintenance is underway; Security Operations has temporarily isolated access and "
        "can restore it within 2 minutes once verified."
        if soar_decision.risk_score >= 85
        else "Review the verification prompt status with Security Operations within 15 minutes."
    )

    return StakeholderAlertBrief(
        incident_id=incident_id,
        department=department,
        pseudonymized_user=soar_decision.pseudonymized_user,
        threat_category=threat_meta["name"],
        mitre_techniques=threat_meta["mitre"],
        risk_score=soar_decision.risk_score,
        confidence_label=confidence_label,
        headline=headline,
        executive_summary=executive_summary,
        top_drivers=drivers,
        automated_actions_taken=soar_decision.automated_actions,
        business_risk_mitigated=_format_business_risk(predicted_threat_id, department),
        stakeholder_action_needed=action_needed,
    )
