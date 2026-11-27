"""End-to-end SentinelMesh AI benchmark pipeline, evaluation, and visual generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sentinelmesh.adversarial import (
    evaluate_adversarial_robustness,
    generate_constrained_evasion,
    train_hardened_detector,
)
from sentinelmesh.config import ENTERPRISE_PROFILE, ENGINEERED_FEATURES, THREAT_TAXONOMY
from sentinelmesh.data_generator import chronological_split, generate_enterprise_telemetry
from sentinelmesh.features import FeatureTransformer
from sentinelmesh.models import train_model_suite
from sentinelmesh.soar import SOAREngine
from sentinelmesh.xai import (
    compute_treeshap_contributions,
    global_shap_importance,
    translate_alert_for_stakeholder,
)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 300,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for a scoring metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(float(metric_fn(y_true[idx], y_score[idx])))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def _evaluate_binary_model(
    name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.50
) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs))
    pr_lo, pr_hi = _bootstrap_ci(y_true, probs, average_precision_score)
    fpr = float(((preds == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
    return {
        "model": name,
        "pr_auc": round(pr_auc, 4),
        "pr_auc_ci95": [pr_lo, pr_hi],
        "roc_auc": round(roc_auc, 4),
        "f1": round(float(f1_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds)), 4),
        "fpr": round(fpr, 4),
    }


def _generate_figures(
    fig_dir: Path,
    y_bin_test: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    y_multi_test: np.ndarray,
    pred_multi_test: np.ndarray,
    shap_df: pd.DataFrame,
    adv_metrics: Dict[str, Dict[str, float]],
    sample_brief_dict: Dict[str, Any],
    dept_slices: List[Dict[str, Any]],
) -> None:
    """Render high-DPI publication charts for the design report and README."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    # 1. ROC & Precision-Recall Curves Comparison
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=220)
    palette = {
        "Logistic Regression (Baseline)": "#64748b",
        "Random Forest": "#0ea5e9",
        "LightGBM": "#8b5cf6",
        "PyTorch Autoencoder (Unsupervised)": "#f59e0b",
        "XGBoost": "#10b981",
        "SentinelMesh Hybrid Fusion": "#dc2626",
    }
    for name, probs in model_probs.items():
        fpr_c, tpr_c, _ = roc_curve(y_bin_test, probs)
        prec_c, rec_c, _ = precision_recall_curve(y_bin_test, probs)
        axes[0].plot(
            fpr_c,
            tpr_c,
            label=f"{name} ({roc_auc_score(y_bin_test, probs):.3f})",
            color=palette.get(name, "#334155"),
            linewidth=2.3 if "SentinelMesh" in name else 1.6,
        )
        axes[1].plot(
            rec_c,
            prec_c,
            label=f"{name} ({average_precision_score(y_bin_test, probs):.3f})",
            color=palette.get(name, "#334155"),
            linewidth=2.3 if "SentinelMesh" in name else 1.6,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="#cbd5e1", linewidth=1)
    axes[0].set_title("ROC Curves (Test Split: Days 48–59)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=8.2)

    axes[1].set_title("Precision-Recall Curves (Test Split)", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8.2)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_model_roc_pr.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Multi-Threat Confusion Matrix (T0..T5)
    labels = [THREAT_TAXONOMY[i]["code"] for i in range(6)]
    cm = confusion_matrix(y_multi_test, pred_multi_test, labels=list(range(6)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=220)
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Row-Normalized Recall"},
        ax=ax,
    )
    ax.set_title("Multi-Threat Attribution Confusion Matrix (Counts & Row Recall)", fontweight="bold")
    ax.set_xlabel("Predicted Threat Class")
    ax.set_ylabel("True Threat Class")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_threat_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Global TreeSHAP Importance + Single-Alert Stakeholder Driver Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)
    top_global = shap_df.head(10).iloc[::-1]
    axes[0].barh(top_global["feature"], top_global["mean_abs_shap"], color="#1d4ed8")
    axes[0].set_title("Top-10 Global Features (Mean |TreeSHAP|)", fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP Contribution| (log-odds)")

    drivers = sample_brief_dict["top_drivers"]
    d_labels = [d["business_label"][:38] + ("..." if len(d["business_label"]) > 38 else "") for d in drivers][::-1]
    d_pcts = [d["relative_impact_pct"] for d in drivers][::-1]
    axes[1].barh(d_labels, d_pcts, color="#dc2626")
    axes[1].set_title(f"Stakeholder Alert Drivers ({sample_brief_dict['incident_id']})", fontweight="bold")
    axes[1].set_xlabel("Relative Positive Risk Contribution (%)")
    for i, v in enumerate(d_pcts):
        axes[1].text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[1].set_xlim(0, max(d_pcts) * 1.22 if d_pcts else 100)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_xai_shap_drivers.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Adversarial Evasion Impact & Mitigation Recovery + Department Slice FPR
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=220)
    scenarios = [
        ("Baseline XGB\n(Clean)", adv_metrics["baseline_xgb_clean"]["recall_tpr"], "#0ea5e9"),
        ("Baseline XGB\n(Evasion Attack)", adv_metrics["baseline_xgb_under_evasion"]["recall_tpr"], "#ef4444"),
        ("Hardened Hybrid\n(Clean)", adv_metrics["hardened_hybrid_clean"]["recall_tpr"], "#10b981"),
        ("Hardened Hybrid\n(Evasion Attack)", adv_metrics["hardened_hybrid_under_evasion"]["recall_tpr"], "#059669"),
    ]
    bars = axes[0].bar(
        [s[0] for s in scenarios],
        [s[1] * 100 for s in scenarios],
        color=[s[2] for s in scenarios],
        width=0.55,
    )
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Threat Detection Recall (%)")
    axes[0].set_title("Adversarial Evasion Impact & Mitigation Recovery", fontweight="bold")
    for b in bars:
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.5,
            f"{b.get_height():.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    depts = [s["department"] for s in dept_slices]
    fprs = [s["fpr"] * 100 for s in dept_slices]
    recalls = [s["recall"] * 100 for s in dept_slices]
    x_pos = np.arange(len(depts))
    axes[1].bar(x_pos - 0.18, fprs, width=0.36, label="False Positive Rate (%)", color="#f59e0b")
    axes[1].bar(x_pos + 0.18, recalls, width=0.36, label="Recall (%)", color="#2563eb")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(depts, rotation=18, ha="right")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Fairness / Slice Audit Across Enterprise Departments", fontweight="bold")
    axes[1].legend(fontsize=8.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_adversarial_and_fairness.png", bbox_inches="tight")
    plt.close(fig)


def run_full_pipeline(output_dir: Path | str = "outputs", seed: int = 42) -> Dict[str, Any]:
    """Execute the full SentinelMesh AI pipeline and export metrics and figures."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_dir = out_path / "figures"

    # 1. Generate & chronologically split telemetry (Days 0-35 train, 36-47 val, 48-59 test)
    df = generate_enterprise_telemetry(n_events=18000, n_users=500, n_days=60, attack_rate=0.08, seed=seed)
    splits = chronological_split(df, train_days=36, val_days=12)

    missing_stats = {
        "geo_velocity_missing_pct": round(float(df["geo_velocity_kmph"].isna().mean() * 100), 2),
        "bytes_out_missing_pct": round(float(df["bytes_out_mb_1h"].isna().mean() * 100), 2),
    }

    # 2. Strict leak-free featurization: fit on train ONLY, transform all splits independently
    transformer = FeatureTransformer()
    X_train_df = transformer.fit_transform(splits.train)
    X_val_df = transformer.transform(splits.val)
    X_test_df = transformer.transform(splits.test)

    y_multi_train = splits.train["threat_type"].to_numpy()
    y_multi_val = splits.val["threat_type"].to_numpy()
    y_multi_test = splits.test["threat_type"].to_numpy()

    y_bin_train = (y_multi_train > 0).astype(int)
    y_bin_val = (y_multi_val > 0).astype(int)
    y_bin_test = (y_multi_test > 0).astype(int)

    # 3. Train baselines & primary models
    suite = train_model_suite(
        X_train=X_train_df.to_numpy(),
        y_multi_train=y_multi_train,
        X_val=X_val_df.to_numpy(),
        y_multi_val=y_multi_val,
        seed=seed,
    )

    X_test_np = X_test_df.to_numpy()
    # Benchmark inference latency per 1,000 events
    t0 = time.perf_counter()
    p_hybrid = suite.predict_hybrid_proba(X_test_np)
    latency_ms_total = (time.perf_counter() - t0) * 1000.0
    latency_ms_per_1k = round(latency_ms_total / max(len(X_test_np) / 1000.0, 1e-6), 2)

    model_probs = {
        "Logistic Regression (Baseline)": suite.logistic_baseline.predict_proba(X_test_np)[:, 1],
        "Random Forest": suite.random_forest.predict_proba(X_test_np)[:, 1],
        "LightGBM": suite.lightgbm_clf.predict_proba(X_test_np)[:, 1],
        "PyTorch Autoencoder (Unsupervised)": suite.autoencoder.predict_anomaly_score(X_test_np),
        "XGBoost": suite.xgb_binary.predict_proba(X_test_np)[:, 1],
        "SentinelMesh Hybrid Fusion": p_hybrid,
    }
    model_comparison = [
        _evaluate_binary_model(name, y_bin_test, probs) for name, probs in model_probs.items()
    ]

    # Multi-threat attribution metrics
    pred_multi_test = suite.xgb_multiclass.predict(X_test_np)
    per_threat_recall: Dict[str, float] = {}
    for tid, meta in THREAT_TAXONOMY.items():
        mask = y_multi_test == tid
        if mask.sum() > 0:
            per_threat_recall[meta["code"]] = round(
                float((pred_multi_test[mask] == tid).mean()), 4
            )

    # 4. Department Slice Audit (Fairness / False Positive Equity)
    hybrid_preds = (p_hybrid >= 0.50).astype(int)
    dept_slices: List[Dict[str, Any]] = []
    for dept in ENTERPRISE_PROFILE["departments"]:
        d_mask = (splits.test["department"] == dept).to_numpy()
        y_d = y_bin_test[d_mask]
        p_d = hybrid_preds[d_mask]
        fpr_d = float(((p_d == 1) & (y_d == 0)).sum() / max((y_d == 0).sum(), 1))
        rec_d = float(((p_d == 1) & (y_d == 1)).sum() / max((y_d == 1).sum(), 1))
        dept_slices.append(
            {
                "department": dept,
                "n_events": int(d_mask.sum()),
                "fpr": round(fpr_d, 4),
                "recall": round(rec_d, 4),
            }
        )

    # 5. Adversarial Evasion Evaluation & Hardened Hybrid Mitigation
    hardened = train_hardened_detector(
        X_train_df=X_train_df,
        y_bin_train=y_bin_train,
        X_val_df=X_val_df,
        y_bin_val=y_bin_val,
        autoencoder=suite.autoencoder,
        seed=seed,
    )
    adv_metrics = evaluate_adversarial_robustness(
        baseline_xgb=suite.xgb_binary,
        hardened=hardened,
        X_test_df=X_test_df,
        y_bin_test=y_bin_test,
        seed=seed,
    )

    # 6. XAI (Exact TreeSHAP) & Non-Technical Stakeholder Demonstration
    shap_vals, _ = compute_treeshap_contributions(suite.xgb_binary, X_test_df)
    shap_df = global_shap_importance(suite.xgb_binary, X_test_df.iloc[:1000])

    # Pick a high-confidence Finance / Exfiltration or ATO incident from the test set
    cand_mask = (y_multi_test == 3) | (y_multi_test == 1)
    cand_indices = np.where(cand_mask & (p_hybrid >= 0.85))[0]
    chosen_idx = int(cand_indices[0]) if len(cand_indices) > 0 else int(np.argmax(p_hybrid))

    soar = SOAREngine()
    raw_row = splits.test.iloc[chosen_idx]
    feat_row = X_test_df.iloc[chosen_idx]
    pred_tid = int(pred_multi_test[chosen_idx]) or int(y_multi_test[chosen_idx]) or 1
    soar_decision = soar.evaluate(
        event_id=str(raw_row["event_id"]),
        raw_user_id=str(raw_row["user_id"]),
        risk_prob=float(p_hybrid[chosen_idx]),
        predicted_threat_id=pred_tid,
        asset_criticality=float(raw_row["asset_criticality"]),
    )
    brief = translate_alert_for_stakeholder(
        incident_id="INC-2026-0418",
        raw_row=raw_row,
        feature_row=feat_row,
        shap_row=shap_vals[chosen_idx],
        ae_detector=suite.autoencoder,
        soar_decision=soar_decision,
        predicted_threat_id=pred_tid,
    )
    sample_brief_dict = brief.to_dict()

    # 7. Generate publication figures
    _generate_figures(
        fig_dir=fig_dir,
        y_bin_test=y_bin_test,
        model_probs=model_probs,
        y_multi_test=y_multi_test,
        pred_multi_test=pred_multi_test,
        shap_df=shap_df,
        adv_metrics=adv_metrics,
        sample_brief_dict=sample_brief_dict,
        dept_slices=dept_slices,
    )

    # Assemble structured metrics output
    summary = {
        "enterprise_profile": ENTERPRISE_PROFILE,
        "dataset_summary": {
            "total_events": int(len(df)),
            "train_events": int(len(splits.train)),
            "val_events": int(len(splits.val)),
            "test_events": int(len(splits.test)),
            "attack_rate_pct": round(float((df["threat_type"] > 0).mean() * 100), 2),
            "n_engineered_features": len(ENGINEERED_FEATURES),
            "missingness": missing_stats,
        },
        "model_comparison": model_comparison,
        "per_threat_attribution_recall": per_threat_recall,
        "inference_latency_ms_per_1k_events": latency_ms_per_1k,
        "department_slice_audit": dept_slices,
        "adversarial_robustness": adv_metrics,
        "top_10_global_shap": shap_df.head(10).to_dict(orient="records"),
        "stakeholder_xai_demo": sample_brief_dict,
    }

    with open(out_path / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary
