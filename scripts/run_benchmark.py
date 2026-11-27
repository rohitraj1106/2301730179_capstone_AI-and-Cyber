"""CLI script to execute the full SentinelMesh AI benchmark and export metrics/charts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinelmesh.pipeline import run_full_pipeline


def main() -> None:
    outputs_dir = ROOT / "outputs"
    summary = run_full_pipeline(output_dir=outputs_dir, seed=42)
    print("=== SentinelMesh AI Benchmark Complete ===")
    for row in summary["model_comparison"]:
        print(
            f"  {row['model']:36s} | PR-AUC={row['pr_auc']:.4f} {row['pr_auc_ci95']} "
            f"| ROC-AUC={row['roc_auc']:.4f} | F1={row['f1']:.4f} | FPR={row['fpr']:.4f}"
        )
    print("\n=== Adversarial Robustness ===")
    print(json.dumps(summary["adversarial_robustness"], indent=2))
    print(f"\nInference latency: {summary['inference_latency_ms_per_1k_events']} ms / 1,000 events")
    print(f"Stakeholder Brief Headline: {summary['stakeholder_xai_demo']['headline']}")


if __name__ == "__main__":
    main()
