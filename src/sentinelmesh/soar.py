"""Confidence-gated SOAR automated response engine with blast-radius circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from sentinelmesh.config import SOARConfig, THREAT_TAXONOMY
from sentinelmesh.features import pseudonymize_identifier


@dataclass
class SOARDecision:
    """Structured SOAR response record for an analyzed telemetry event."""

    event_id: str
    pseudonymized_user: str
    risk_score: float
    predicted_threat_code: str
    predicted_threat_name: str
    tier: str
    automated_actions: List[str]
    human_approval_required: bool
    circuit_breaker_tripped: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SOAREngine:
    """Applies confidence-gated containment with circuit-breaker safeguards."""

    def __init__(self, config: SOARConfig | None = None, hmac_key: str | None = None) -> None:
        self.config = config or SOARConfig()
        self.hmac_key = hmac_key
        self._recent_isolations: int = 0
        self.audit_log: List[Dict[str, object]] = []

    def reset_window(self) -> None:
        """Reset the rolling isolation counter for a new 10-minute window."""
        self._recent_isolations = 0

    def evaluate(
        self,
        event_id: str,
        raw_user_id: str,
        risk_prob: float,
        predicted_threat_id: int,
        asset_criticality: float,
    ) -> SOARDecision:
        """Determine confidence-gated SOAR response and record an audit entry."""
        # Criticality-adjusted risk score on [0, 100]
        adjusted_prob = min(1.0, float(risk_prob) * (0.92 + 0.16 * float(asset_criticality)))
        risk_score = round(adjusted_prob * 100.0, 2)

        threat_info = THREAT_TAXONOMY.get(predicted_threat_id, THREAT_TAXONOMY[0])
        pseudo_user = pseudonymize_identifier(raw_user_id, self.hmac_key)
        breaker_tripped = False

        if adjusted_prob < self.config.medium_threshold:
            tier = "LOW_MONITOR"
            actions = ["Log to cold parquet lake", "Update UEBA baseline only if benign-confirmed"]
            human_required = False
        elif adjusted_prob < self.config.high_threshold:
            tier = "MEDIUM_ENRICH_VERIFY"
            actions = [
                "Open Tier-1 SOC triage ticket",
                "Send out-of-band push verification to user",
            ]
            human_required = True
        elif adjusted_prob < self.config.critical_threshold:
            tier = "HIGH_SOFT_CONTAINMENT"
            actions = [
                "Enforce step-up FIDO2 MFA challenge",
                "Apply temporary IAM read-only / egress rate-limit",
                "Escalate to Tier-2 SOC (15-min SLA)",
            ]
            human_required = True
        else:
            tier = "CRITICAL_AUTO_CONTAIN"
            if self._recent_isolations < self.config.circuit_breaker_max_isolations_per_window:
                self._recent_isolations += 1
                actions = [
                    "Revoke active Okta SSO & OAuth refresh tokens",
                    "Attach AWS STS explicit-deny quarantine policy",
                    "Network-isolate endpoint via EDR (reversible)",
                    "Page Incident Commander & generate executive XAI brief",
                ]
                human_required = False  # Reversible containment is automatic; permanent changes need human
            else:
                breaker_tripped = True
                tier = "CRITICAL_CIRCUIT_BREAKER_HOLD"
                actions = [
                    "Circuit breaker engaged: halt bulk auto-isolation to prevent enterprise self-DoS",
                    "Page CISO & Tier-2 SOC Lead for manual bulk containment approval",
                ]
                human_required = True

        decision = SOARDecision(
            event_id=event_id,
            pseudonymized_user=pseudo_user,
            risk_score=risk_score,
            predicted_threat_code=threat_info["code"],
            predicted_threat_name=threat_info["name"],
            tier=tier,
            automated_actions=actions,
            human_approval_required=human_required,
            circuit_breaker_tripped=breaker_tripped,
        )
        self.audit_log.append(decision.to_dict())
        return decision
