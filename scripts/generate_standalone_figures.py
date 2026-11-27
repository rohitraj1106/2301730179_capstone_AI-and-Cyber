"""Generate large, spacious standalone high-DPI figures for the new academic report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinelmesh.adversarial import evaluate_adversarial_robustness, train_hardened_detector
from sentinelmesh.config import ENTERPRISE_PROFILE, THREAT_TAXONOMY
from sentinelmesh.data_generator import chronological_split, generate_enterprise_telemetry
from sentinelmesh.features import FeatureTransformer
from sentinelmesh.models import train_model_suite
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def main() -> None:
    out_dir = ROOT / "outputs" / "figures_academic"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ROOT / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Re-run deterministic seed=42 curves/confusion matrix for full-resolution standalone charts
    df = generate_enterprise_telemetry(n_events=18000, n_users=500, n_days=60, attack_rate=0.08, seed=42)
    splits = chronological_split(df, train_days=36, val_days=12)
    tf = FeatureTransformer()
    X_train = tf.fit_transform(splits.train)
    X_val = tf.transform(splits.val)
    X_test = tf.transform(splits.test)

    y_m_train = splits.train["threat_type"].to_numpy()
    y_m_val = splits.val["threat_type"].to_numpy()
    y_m_test = splits.test["threat_type"].to_numpy()
    y_b_test = (y_m_test > 0).astype(int)

    suite = train_model_suite(X_train.to_numpy(), y_m_train, X_val.to_numpy(), y_m_val, seed=42)
    X_test_np = X_test.to_numpy()

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.08)

    # 1. Threat Distribution Across 5 Attack Types (18,000 events)
    counts = df[df["threat_type"] > 0]["threat_type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=240)
    names = [
        f"{THREAT_TAXONOMY[k]['code'].replace('_', ' ')}\n({THREAT_TAXONOMY[k]['mitre'].split(',')[0]})"
        for k in counts.index
    ]
    colors = ["#1d4ed8", "#dc2626", "#d97706", "#7c3aed", "#059669"]
    bars = ax.bar(names, counts.to_numpy(), color=colors, width=0.52, edgecolor="#1e293b", linewidth=0.7)
    ax.tick_params(axis="x", labelsize=9.2)
    ax.set_title(
        f"Attack Class Distribution Across 60-Day Enterprise Telemetry (n = {int(counts.sum()):,} Attacks / {len(df):,} Events)",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Event Windows")
    ax.set_ylim(0, max(counts.to_numpy()) * 1.18)
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 8,
            f"{int(b.get_height())}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_threat_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Confusion Matrix (Standalone large)
    pred_multi = suite.xgb_multiclass.predict(X_test_np)
    labels = [THREAT_TAXONOMY[i]["code"] for i in range(6)]
    cm = confusion_matrix(y_m_test, pred_multi, labels=list(range(6)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=240)
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Row-Normalized Recall"},
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Multi-Threat Attribution Confusion Matrix (Held-Out Test Split, n = 3,490)", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Threat Class", fontsize=11)
    ax.set_ylabel("Actual Threat Class", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 3. ROC & Precision-Recall Benchmark (Full width)
    model_probs = {
        "Logistic Regression (Baseline)": suite.logistic_baseline.predict_proba(X_test_np)[:, 1],
        "PyTorch Autoencoder (Unsupervised)": suite.autoencoder.predict_anomaly_score(X_test_np),
        "LightGBM": suite.lightgbm_clf.predict_proba(X_test_np)[:, 1],
        "Random Forest": suite.random_forest.predict_proba(X_test_np)[:, 1],
        "XGBoost (Supervised)": suite.xgb_binary.predict_proba(X_test_np)[:, 1],
        "SentinelMesh Hybrid Fusion": suite.predict_hybrid_proba(X_test_np),
    }
    palette = {
        "Logistic Regression (Baseline)": "#64748b",
        "PyTorch Autoencoder (Unsupervised)": "#f59e0b",
        "LightGBM": "#8b5cf6",
        "Random Forest": "#0284c7",
        "XGBoost (Supervised)": "#10b981",
        "SentinelMesh Hybrid Fusion": "#dc2626",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), dpi=240)
    for name, probs in model_probs.items():
        fpr_c, tpr_c, _ = roc_curve(y_b_test, probs)
        prec_c, rec_c, _ = precision_recall_curve(y_b_test, probs)
        lw = 2.4 if "SentinelMesh" in name or "XGBoost" in name else 1.7
        axes[0].plot(
            fpr_c, tpr_c,
            label=f"{name} ({roc_auc_score(y_b_test, probs):.3f})",
            color=palette[name], linewidth=lw,
        )
        axes[1].plot(
            rec_c, prec_c,
            label=f"{name} ({average_precision_score(y_b_test, probs):.3f})",
            color=palette[name], linewidth=lw,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="#cbd5e1", linewidth=1)
    axes[0].set_title("ROC Curves (Test Days 48–59)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", fontsize=8.3)

    axes[1].set_title("Precision-Recall Curves (Test Days 48–59)", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8.3)
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Adversarial Evasion Impact & Recovery (Standalone large)
    adv = metrics["adversarial_robustness"]
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=240)
    scenarios = [
        ("Baseline XGBoost\n(Clean Test)", adv["baseline_xgb_clean"]["recall_tpr"] * 100, "#0284c7"),
        ("Baseline XGBoost\n(Evasion Attack)", adv["baseline_xgb_under_evasion"]["recall_tpr"] * 100, "#dc2626"),
        ("Hardened Hybrid\n(Clean Test)", adv["hardened_hybrid_clean"]["recall_tpr"] * 100, "#10b981"),
        ("Hardened Hybrid\n(Evasion Attack)", adv["hardened_hybrid_under_evasion"]["recall_tpr"] * 100, "#047857"),
    ]
    bars = ax.bar([s[0] for s in scenarios], [s[1] for s in scenarios], color=[s[2] for s in scenarios], width=0.5, edgecolor="#1e293b", linewidth=0.7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Threat Detection Recall (%)", fontsize=11)
    ax.set_title("Adversarial Evasion Recall Collapse vs. Hardened Hybrid Recovery", fontweight="bold", pad=12)
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.8,
            f"{b.get_height():.2f}%",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
        )
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_adversarial_recovery.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Global TreeSHAP Importance (Standalone large)
    top_shap = metrics["top_10_global_shap"][::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=240)
    ax.barh(
        [r["feature"] for r in top_shap],
        [r["mean_abs_shap"] for r in top_shap],
        color="#1d4ed8",
        edgecolor="#0f172a",
        linewidth=0.6,
    )
    ax.set_title("Global Explainable AI: Top-10 Features by Mean |TreeSHAP| Contribution", fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Contribution| (log-odds impact)", fontsize=10.5)
    for i, r in enumerate(top_shap):
        ax.text(r["mean_abs_shap"] + 0.025, i, f"{r['mean_abs_shap']:.3f}", va="center", fontsize=9.5)
    ax.set_xlim(0, max(r["mean_abs_shap"] for r in top_shap) * 1.16)
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_global_shap.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Department Fairness / Slice Audit (Standalone large with data labels)
    slices = metrics["department_slice_audit"]
    depts = [s["department"] for s in slices]
    fprs = [s["fpr"] * 100 for s in slices]
    recalls = [s["recall"] * 100 for s in slices]
    x = np.arange(len(depts))
    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=240)
    b1 = ax.bar(x - 0.19, fprs, width=0.38, label="False Positive Rate (%)", color="#d97706", edgecolor="#1e293b", linewidth=0.6)
    b2 = ax.bar(x + 0.19, recalls, width=0.38, label="Threat Recall (%)", color="#1d4ed8", edgecolor="#1e293b", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(depts, fontsize=10.5)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Departmental Fairness & Slice Audit (SentinelMesh Hybrid Fusion)", fontweight="bold", pad=12)
    ax.legend(loc="upper center", ncol=2, fontsize=9.5, frameon=True)
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.2f}%", ha="center", fontsize=8.8, fontweight="bold", color="#92400e")
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%", ha="center", fontsize=8.8, fontweight="bold", color="#1e3a8a")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_department_fairness.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated 6 standalone figures in:", out_dir)


if __name__ == "__main__":
    main()
