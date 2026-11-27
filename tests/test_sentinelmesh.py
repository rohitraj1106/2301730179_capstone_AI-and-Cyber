"""Unit and integration tests for SentinelMesh AI."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from sentinelmesh.adversarial import (
    evaluate_adversarial_robustness,
    generate_constrained_evasion,
    train_hardened_detector,
)
from sentinelmesh.config import ENGINEERED_FEATURES
from sentinelmesh.data_generator import chronological_split, generate_enterprise_telemetry
from sentinelmesh.features import FeatureTransformer, pseudonymize_identifier
from sentinelmesh.models import train_model_suite
from sentinelmesh.soar import SOARConfig, SOAREngine
from sentinelmesh.xai import compute_treeshap_contributions, translate_alert_for_stakeholder


def test_pseudonymization_is_deterministic_and_opaque() -> None:
    p1 = pseudonymize_identifier("emp_0042", secret_key="unit-test-key")
    p2 = pseudonymize_identifier("emp_0042", secret_key="unit-test-key")
    p3 = pseudonymize_identifier("emp_0043", secret_key="unit-test-key")
    assert p1 == p2
    assert p1 != p3
    assert "emp_0042" not in p1
    assert p1.startswith("usr_")


def test_chronological_split_and_leak_free_features() -> None:
    df = generate_enterprise_telemetry(n_events=2500, n_users=80, n_days=60, seed=7)
    splits = chronological_split(df, train_days=36, val_days=12)
    assert splits.train["day_index"].max() < splits.val["day_index"].min()
    assert splits.val["day_index"].max() < splits.test["day_index"].min()

    tf = FeatureTransformer()
    X_train = tf.fit_transform(splits.train)
    X_test = tf.transform(splits.test)
    assert list(X_train.columns) == ENGINEERED_FEATURES
    assert list(X_test.columns) == ENGINEERED_FEATURES
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()


def test_models_xai_and_adversarial_recovery() -> None:
    df = generate_enterprise_telemetry(n_events=4500, n_users=120, n_days=60, seed=11)
    splits = chronological_split(df, train_days=36, val_days=12)
    tf = FeatureTransformer()
    X_train = tf.fit_transform(splits.train)
    X_val = tf.transform(splits.val)
    X_test = tf.transform(splits.test)

    y_m_train = splits.train["threat_type"].to_numpy()
    y_m_val = splits.val["threat_type"].to_numpy()
    y_m_test = splits.test["threat_type"].to_numpy()
    y_b_train = (y_m_train > 0).astype(int)
    y_b_val = (y_m_val > 0).astype(int)
    y_b_test = (y_m_test > 0).astype(int)

    suite = train_model_suite(X_train.to_numpy(), y_m_train, X_val.to_numpy(), y_m_val, seed=11)
    probs = suite.predict_hybrid_proba(X_test.to_numpy())
    assert probs.shape == (len(X_test),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Verify exact TreeSHAP additivity (sum of contributions + bias == raw margin)
    shap_vals, base_vals = compute_treeshap_contributions(suite.xgb_binary, X_test.iloc[:5])
    raw_margins = suite.xgb_binary.get_booster().predict(
        xgb.DMatrix(X_test.iloc[:5].to_numpy(), feature_names=ENGINEERED_FEATURES),
        output_margin=True,
    )
    assert np.allclose(shap_vals.sum(axis=1) + base_vals, raw_margins, atol=1e-4)

    # Verify adversarial hardening recovers recall under evasion compared to baseline
    hardened = train_hardened_detector(
        X_train, y_b_train, X_val, y_b_val, autoencoder=suite.autoencoder, seed=11
    )
    adv_res = evaluate_adversarial_robustness(suite.xgb_binary, hardened, X_test, y_b_test, seed=11)
    assert (
        adv_res["hardened_hybrid_under_evasion"]["recall_tpr"]
        >= adv_res["baseline_xgb_under_evasion"]["recall_tpr"]
    )

    # Verify Stakeholder Brief translation produces plain-English drivers
    ev_df = generate_constrained_evasion(X_test, y_b_test, strength=1.0, seed=11)
    assert ev_df.shape == X_test.shape

    soar = SOAREngine()
    mal_idx = int(np.where(y_b_test == 1)[0][0])
    dec = soar.evaluate("evt_test", "emp_0001", 0.94, int(y_m_test[mal_idx]), 0.85)
    brief = translate_alert_for_stakeholder(
        "INC-TEST",
        splits.test.iloc[mal_idx],
        X_test.iloc[mal_idx],
        shap_vals[0],
        suite.autoencoder,
        dec,
        int(y_m_test[mal_idx]),
    )
    assert len(brief.top_drivers) >= 1
    assert "emp_0001" not in brief.executive_summary


def test_soar_circuit_breaker_prevents_self_dos() -> None:
    soar = SOAREngine(config=SOARConfig(circuit_breaker_max_isolations_per_window=2))
    d1 = soar.evaluate("e1", "emp_1", 0.95, 2, 0.9)
    d2 = soar.evaluate("e2", "emp_2", 0.95, 2, 0.9)
    d3 = soar.evaluate("e3", "emp_3", 0.95, 2, 0.9)
    assert not d1.circuit_breaker_tripped
    assert not d2.circuit_breaker_tripped
    assert d3.circuit_breaker_tripped
    assert d3.tier == "CRITICAL_CIRCUIT_BREAKER_HOLD"
