"""
Plotting module for the UPI credit scoring simulation study.
All data is SYNTHETIC.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import shap
from pathlib import Path

FIG_DPI = 300
PALETTE = {
    "XGBoost":           "#2563EB",
    "LogisticRegression":"#DC2626",
    "RandomForest":      "#16A34A",
    "Eligible":          "#15803D",
    "Review":            "#D97706",
    "High_risk":         "#DC2626",
    "class0":            "#3B82F6",
    "class1":            "#EF4444",
}

CAPTION_SUFFIX = "(Synthetic data — methodological feasibility only)"


def save_fig(fig, path, dpi=FIG_DPI):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_data, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, color_key in [("xgb", "XGBoost"), ("lr", "LogisticRegression"), ("rf", "RandomForest")]:
        d = roc_data[name]
        label = f"{color_key} (AUC={d['auc']:.4f})"
        ax.plot(d["fpr"], d["tpr"], color=PALETTE[color_key], lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — All Models\n{CAPTION_SUFFIX}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)
    return out_path


def plot_score_distribution(score_by_class, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 100, 41)
    ax.hist(score_by_class["class0_scores"], bins=bins, alpha=0.6,
            color=PALETTE["class0"], label="Repaid (class 0)", density=True)
    ax.hist(score_by_class["class1_scores"], bins=bins, alpha=0.6,
            color=PALETTE["class1"], label="Defaulted (class 1)", density=True)
    ax.axvline(70, color=PALETTE["Eligible"], lw=1.5, linestyle="--", label="Eligible threshold (70)")
    ax.axvline(40, color=PALETTE["Review"],   lw=1.5, linestyle=":",  label="Review threshold (40)")
    ax.set_xlabel("Credit Score (0–100)")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution by True Class\n{CAPTION_SUFFIX}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)
    return out_path


def plot_calibration(calibration_data, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(calibration_data["mean_pred_before"], calibration_data["frac_pos_before"],
            "s-", color=PALETTE["XGBoost"], label=f"XGBoost (Brier={calibration_data['brier_before_calibration']:.4f})")
    ax.plot(calibration_data["mean_pred_after"], calibration_data["frac_pos_after"],
            "o--", color="#7C3AED", label=f"+ Isotonic (Brier={calibration_data['brier_after_calibration']:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve (XGBoost)\n{CAPTION_SUFFIX}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)
    return out_path


def plot_tier_analysis(tier_stats, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    tier_order = ["Eligible", "Review", "High_risk"]
    ts = tier_stats.set_index("tier").reindex(tier_order).reset_index()

    colors = [PALETTE.get(t, "#6B7280") for t in tier_order]

    # Bar: count
    axes[0].bar(ts["tier"], ts["count"], color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title(f"Workers per Tier\n{CAPTION_SUFFIX}", fontsize=9)
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    # Bar: default rate
    bars = axes[1].bar(ts["tier"], ts["default_rate"], color=colors, edgecolor="white", linewidth=0.5)
    axes[1].axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.6)
    axes[1].set_title(f"Default Rate per Tier\n{CAPTION_SUFFIX}", fontsize=9)
    axes[1].set_ylabel("Default Rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, ts["default_rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.2%}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_fig(fig, out_path)
    return out_path


def plot_ablation(ablation_results, out_path):
    names = list(ablation_results.keys())
    aucs   = [ablation_results[n]["AUC"] for n in names]
    briers = [ablation_results[n]["Brier"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    x = np.arange(len(names))
    colors = ["#7C3AED", "#D97706", "#2563EB"]

    axes[0].bar(x, aucs, color=colors, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("AUC")
    axes[0].set_title(f"Ablation — AUC\n{CAPTION_SUFFIX}", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    for xi, v in zip(x, aucs):
        axes[0].text(xi, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, briers, color=colors, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylabel("Brier Score")
    axes[1].set_title(f"Ablation — Brier Score\n{CAPTION_SUFFIX}", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    for xi, v in zip(x, briers):
        axes[1].text(xi, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_fig(fig, out_path)
    return out_path


def plot_fairness(fairness, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Income tercile approval rates
    rates_i = fairness["income_tercile_approval_rates"]
    keys_i = list(rates_i.keys())
    vals_i = [rates_i[k] for k in keys_i]
    axes[0].bar(keys_i, vals_i, color=["#1D4ED8", "#3B82F6", "#93C5FD"], edgecolor="white")
    if max(vals_i) > 0:
        axes[0].axhline(min(vals_i) / max(vals_i) * max(vals_i), color="gray", linestyle="--", lw=0.8)
    axes[0].set_title(f"Approval Rate by Income Tercile\n{CAPTION_SUFFIX}", fontsize=9)
    axes[0].set_ylabel("Approval Rate (Eligible Tier)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)
    dir_i = fairness.get("disparate_impact_ratio_income", "N/A")
    dir_i_str = f"{dir_i:.3f}" if isinstance(dir_i, (int, float)) else str(dir_i)
    axes[0].set_xlabel(f"DIR = {dir_i_str} (acceptable: 0.80–1.25)")

    # Worker type approval rates
    rates_w = fairness["worker_type_approval_rates"]
    keys_w = list(rates_w.keys())
    vals_w = [rates_w[k] for k in keys_w]
    colors_w = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]
    axes[1].bar(keys_w, vals_w, color=colors_w[:len(keys_w)], edgecolor="white")
    axes[1].set_title(f"Approval Rate by Worker Category\n{CAPTION_SUFFIX}", fontsize=9)
    axes[1].set_ylabel("Approval Rate (Eligible Tier)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)
    dir_w = fairness.get("disparate_impact_ratio_workertype", "N/A")
    dir_w_str = f"{dir_w:.3f}" if isinstance(dir_w, (int, float)) else str(dir_w)
    axes[1].set_xlabel(f"DIR = {dir_w_str} (acceptable: 0.80–1.25)")

    plt.tight_layout()
    save_fig(fig, out_path)
    return out_path


def plot_shap_global(shap_values, feature_names, out_dir):
    """Global SHAP bar plot and beeswarm."""
    shap_arr = shap_values.values
    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:15]

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_names_top = [feature_names[i] for i in order]
    vals_top = mean_abs_shap[order]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(vals_top)))
    ax.barh(range(len(vals_top)), vals_top[::-1], color=colors)
    ax.set_yticks(range(len(vals_top)))
    ax.set_yticklabels(feat_names_top[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Global Feature Importance\n{CAPTION_SUFFIX}", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = out_dir / "shap_global_bar.png"
    save_fig(fig, bar_path)

    # Beeswarm
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Beeswarm Summary\n{CAPTION_SUFFIX}", fontsize=9)
    plt.tight_layout()
    bee_path = out_dir / "shap_beeswarm.png"
    plt.savefig(bee_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")

    return bar_path, bee_path


def plot_multi_seed(multi_seed, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    seeds = multi_seed["seeds"]
    xgb_aucs = multi_seed["xgb_aucs"]
    lr_aucs  = multi_seed["lr_aucs"]
    x = np.arange(len(seeds))

    ax.bar(x - 0.2, xgb_aucs, 0.35, label="XGBoost", color=PALETTE["XGBoost"])
    ax.bar(x + 0.2, lr_aucs,  0.35, label="LR",       color=PALETTE["LogisticRegression"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds], fontsize=8)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(max(0.5, min(xgb_aucs + lr_aucs) - 0.05), 1.0)
    ax.set_title(f"Multi-seed Stability (5 seeds)\n{CAPTION_SUFFIX}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path)
    return out_path


def generate_all_figures(exp_outputs: dict, results_dir: Path) -> dict:
    figs_dir = results_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    res = exp_outputs["results"]
    paths = {}

    print("Plotting ROC curves...")
    paths["roc"] = plot_roc_curves(res["roc_data"], figs_dir / "roc_curves.png")

    print("Plotting score distribution...")
    paths["score_dist"] = plot_score_distribution(res["score_by_class"], figs_dir / "score_distribution.png")

    print("Plotting calibration...")
    paths["calibration"] = plot_calibration(res["calibration"], figs_dir / "calibration.png")

    print("Plotting tier analysis...")
    paths["tier"] = plot_tier_analysis(
        pd.DataFrame(res["tier_stats"]), figs_dir / "tier_analysis.png"
    )

    print("Plotting ablation...")
    paths["ablation"] = plot_ablation(res["ablation"], figs_dir / "ablation.png")

    print("Plotting fairness...")
    paths["fairness"] = plot_fairness(res["fairness"], figs_dir / "fairness.png")

    print("Plotting SHAP global...")
    bar_p, bee_p = plot_shap_global(
        exp_outputs["shap_values"], exp_outputs["explain_sample"].columns.tolist(), figs_dir
    )
    paths["shap_bar"] = bar_p
    paths["shap_beeswarm"] = bee_p

    print("Plotting multi-seed stability...")
    paths["multi_seed"] = plot_multi_seed(res["multi_seed"], figs_dir / "multi_seed_stability.png")

    return paths
