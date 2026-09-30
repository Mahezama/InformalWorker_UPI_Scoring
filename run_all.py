"""
Main orchestration script for the UPI credit scoring simulation study.
Run: python run_all.py
All data is SYNTHETIC. See results/report.pdf for full details.
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

t_total_start = time.time()

print("=" * 70)
print("UPI Credit Scoring Simulation Study")
print("ALL DATA IS SYNTHETIC — Methodological feasibility only")
print("=" * 70)

# ── Step 1: Synthetic Data Generation ───────────────────────────────────────
print("\n[STEP 1] Generating synthetic data (seed=42)...")
t0 = time.time()

from src.data_generator import generate_dataset

Path("data").mkdir(exist_ok=True)

data_files_exist = (
    Path("data/upi_events.csv").exists() and
    Path("data/recharge_events.csv").exists() and
    Path("data/worker_meta.csv").exists()
)

if data_files_exist:
    print("  Loading existing data files (seed=42 already ran)...")
    upi_df = pd.read_csv("data/upi_events.csv")
    recharge_df = pd.read_csv("data/recharge_events.csv")
    worker_df = pd.read_csv("data/worker_meta.csv")
else:
    upi_df, recharge_df, worker_df = generate_dataset(seed=42, n_workers=10_000)
    upi_df.to_csv("data/upi_events.csv", index=False)
    recharge_df.to_csv("data/recharge_events.csv", index=False)
    worker_df.to_csv("data/worker_meta.csv", index=False)

print(f"  UPI events      : {len(upi_df):>10,}")
print(f"  Recharge events : {len(recharge_df):>10,}")
print(f"  Workers         : {len(worker_df):>10,}")
print(f"  Time: {time.time()-t0:.1f}s")

# ── Step 2: Feature Engineering ──────────────────────────────────────────────
print("\n[STEP 2] Computing features from raw event logs...")
t0 = time.time()

from src.feature_engineering import compute_features

if Path("data/features.csv").exists():
    print("  Loading existing features.csv...")
    feat_df = pd.read_csv("data/features.csv")
else:
    feat_df = compute_features(upi_df, recharge_df, worker_df)
    feat_df.to_csv("data/features.csv", index=False)

print(f"  Feature matrix shape: {feat_df.shape}")
print(f"  Time: {time.time()-t0:.1f}s")

# ── Step 3 & 4: Experiments and Metrics ──────────────────────────────────────
print("\n[STEP 3+4] Running experiments...")
t0 = time.time()

from src.experiments import run_experiments

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
(results_dir / "figs").mkdir(exist_ok=True)

exp_outputs = run_experiments(feat_df, results_dir)
print(f"  Experiments done. Time: {time.time()-t0:.1f}s")

# ── Step 5a: Figures ─────────────────────────────────────────────────────────
print("\n[STEP 5a] Generating figures...")
t0 = time.time()

from src.plotting import generate_all_figures

fig_paths = generate_all_figures(exp_outputs, results_dir)
print(f"  Figures written to results/figs/. Time: {time.time()-t0:.1f}s")

# ── Step 5b: PDF Report ──────────────────────────────────────────────────────
print("\n[STEP 5b] Building PDF report...")
t0 = time.time()

from src.report_builder import build_pdf

pdf_path = build_pdf(results_dir, fig_paths, exp_outputs)
print(f"  Time: {time.time()-t0:.1f}s")

# ── Step 5c: Summary Markdown ────────────────────────────────────────────────
print("\n[STEP 5c] Writing summary.md...")

res = exp_outputs["results"]
ms  = res["multi_seed"]
cs  = res["cold_start"]
lat = res["latency_ms"]
cal = res["calibration"]
fair = res["fairness"]
abl  = res["ablation"]
imb  = res["imbalance_experiment"]

metrics_main = {m["model"]: m for m in res["main_metrics"]}

summary_lines = [
    "# Key Numbers — UPI Credit Scoring Simulation Study (SYNTHETIC DATA)",
    "",
    "## Main Model Metrics (Table 2, held-out test set)",
    f"- XGBoost AUC       : {metrics_main['XGBoost']['AUC']}   (Fig 1, Table 2)",
    f"- XGBoost Brier     : {metrics_main['XGBoost']['Brier']}   (Table 2, 3)",
    f"- XGBoost F1        : {metrics_main['XGBoost']['F1']}   (Table 2)",
    f"- XGBoost Accuracy  : {metrics_main['XGBoost']['Accuracy']}   (Table 2)",
    f"- LR AUC            : {metrics_main['LogisticRegression']['AUC']}   (Table 2)",
    f"- RF AUC            : {metrics_main['RandomForest']['AUC']}   (Table 2)",
    "",
    "## Calibration (Table 3)",
    f"- Brier before isotonic : {cal['brier_before_calibration']}   (Table 3, Fig 3)",
    f"- Brier after isotonic  : {cal['brier_after_calibration']}   (Table 3, Fig 3)",
    "",
    "## Multi-seed Stability (Table 6)",
    f"- XGBoost AUC: {ms['xgb_mean_auc']} ± {ms['xgb_std_auc']}   (Table 6, Fig 6)",
    f"- LR AUC:      {ms['lr_mean_auc']} ± {ms['lr_std_auc']}   (Table 6, Fig 6)",
    "",
    "## Tier Analysis (Table 4, Fig 5)",
]
for t in res["tier_stats"]:
    summary_lines.append(f"- {t['tier']:10s}: count={t['count']}, default_rate={t['default_rate']:.4f}, mean_score={t['mean_score']:.2f}")

summary_lines += [
    f"- Tier monotonicity: {'YES' if res['tier_monotonic'] else 'NO — reported as-is'}",
    "",
    "## Ablation (Table 9, Fig 4)",
    f"- UPI-only   AUC: {abl['UPI_only']['AUC']}",
    f"- Recharge-only AUC: {abl['Recharge_only']['AUC']}",
    f"- Combined   AUC: {abl['Combined']['AUC']}",
    "",
    "## Cold-start (Table 10)",
    f"- Cold-start workers (<15 UPI txns): {cs['n_cold_workers']}",
    f"- UPI-only AUC (cold): {cs.get('UPI_only_AUC', 'N/A')}",
    f"- Combined AUC (cold): {cs.get('Combined_AUC', 'N/A')}",
    "",
    "## Fairness (Table 11, Fig 7)",
    f"- Disparate Impact Ratio (income tercile): {fair['disparate_impact_ratio_income']}",
    f"- Disparate Impact Ratio (worker type):    {fair['disparate_impact_ratio_workertype']}",
    f"- Acceptable range: 0.80–1.25",
    "",
    "## Latency (Section 8)",
    f"- Mean latency (predict + SHAP): {lat['mean']} ms",
    f"- P95 latency  (predict + SHAP): {lat['p95']} ms",
    "",
    "## Imbalanced-data Experiment (80/20)",
    f"- AUC: {imb['AUC']}   Brier: {imb['Brier']}",
    "",
    "---",
    "NOTE: ALL DATA IS SYNTHETIC. All results indicate methodological feasibility only.",
    "See results/report.pdf for full details and limitations.",
]

with open(results_dir / "summary.md", "w") as f:
    f.write("\n".join(summary_lines))

# ── Final output ──────────────────────────────────────────────────────────────
total_time = time.time() - t_total_start

print("\n" + "=" * 70)
print(f"DONE  |  Total runtime: {total_time:.1f}s")
print("=" * 70)
print(f"\nPDF report: {pdf_path.absolute()}")
print("\n10-line Headline Summary:")
print(f"  [1] XGBoost Test AUC       : {metrics_main['XGBoost']['AUC']}  (Table 2, Fig 1)")
print(f"  [2] XGBoost Brier Score    : {metrics_main['XGBoost']['Brier']}  (Table 2/3, Fig 3)")
print(f"  [3] XGBoost F1 Score       : {metrics_main['XGBoost']['F1']}  (Table 2)")
print(f"  [4] XGB vs LR AUC (5 seeds): {ms['xgb_mean_auc']}±{ms['xgb_std_auc']} vs {ms['lr_mean_auc']}±{ms['lr_std_auc']}  (Table 6)")
print(f"  [5] Ablation (UPI/Rch/Comb): {abl['UPI_only']['AUC']} / {abl['Recharge_only']['AUC']} / {abl['Combined']['AUC']} AUC  (Table 9)")
print(f"  [6] Cold-start UPI/Combined: {cs.get('UPI_only_AUC','N/A')} / {cs.get('Combined_AUC','N/A')} AUC  (Table 10)")
print(f"  [7] Brier after calibration: {cal['brier_after_calibration']}  (Table 3)")
print(f"  [8] DIR income/worker-type : {fair['disparate_impact_ratio_income']} / {fair['disparate_impact_ratio_workertype']}  (Table 11)")
print(f"  [9] Latency mean/P95 (ms)  : {lat['mean']} / {lat['p95']}  (Section 8)")
print(f" [10] Data: SYNTHETIC — 10k workers — ALL results for feasibility only")
