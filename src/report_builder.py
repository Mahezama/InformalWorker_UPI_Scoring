"""
PDF report generator for the UPI credit scoring simulation study.
Uses matplotlib PdfPages for a clean, structured report.
All data is SYNTHETIC.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import json
import textwrap
from datetime import datetime


TITLE = "Informal Worker Income Verification for Micro-Loan Eligibility:\nAn Explainable ML Framework Using UPI Transaction\nand Mobile Recharge Behaviour"
SUBTITLE = "[SYNTHETIC DATA SIMULATION STUDY]"
DATE = datetime.now().strftime("%B %d, %Y")

PLAIN_LANG = {
    "upi_txns_per_week":                      "Weekly UPI transaction frequency",
    "upi_weekly_count_cv":                     "Coefficient of variation in weekly UPI count",
    "upi_distinct_counterparties_per_month":   "Distinct counterparties per month",
    "upi_inflow_cv":                           "Coefficient of variation in weekly inflow",
    "net_inflow_ma3":                          "3-month moving average net UPI inflow (INR)",
    "net_inflow_trend":                        "Net inflow trend over 3 months",
    "inflow_outflow_ratio":                    "UPI inflow-to-outflow ratio",
    "est_monthly_income":                      "Estimated monthly income (UPI credits, INR)",
    "est_monthly_expenses":                    "Estimated monthly expenses (UPI debits, INR)",
    "emi_signature_count":                     "Candidate EMI / fixed-recurring debit count",
    "upi_activity_span_days":                  "Activity span (days from first to last UPI txn)",
    "upi_total_count":                         "Total UPI transactions in 90-day window",
    "est_disposable_surplus":                  "Estimated disposable surplus (income−expenses−EMI, INR)",
    "recharge_per_month":                      "Mobile recharge frequency per month",
    "recharge_interval_std":                   "Std-dev of days between mobile recharges",
    "recharge_amount_trend":                   "Trend in recharge plan amount across months",
    "recharge_count":                          "Total mobile recharges in 90-day window",
    "recharge_amount_mean":                    "Mean recharge plan amount (INR)",
    "recharge_amount_std":                     "Std-dev of recharge plan amount (INR)",
    "upi_txns_per_week_missing":               "Missing indicator: sparse UPI history",
    "est_monthly_income_missing":              "Missing indicator: no income signal",
    "net_inflow_ma3_missing":                  "Missing indicator: insufficient inflow data",
    "recharge_per_month_missing":              "Missing indicator: sparse recharge history",
    "recharge_interval_std_missing":           "Missing indicator: fewer than 2 recharges",
}


def add_title_page(pdf, title=TITLE, subtitle=SUBTITLE):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#0F172A")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Decorative top bar
    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.82), 0.90, 0.004,
                                          boxstyle="round,pad=0.001",
                                          facecolor="#3B82F6", edgecolor="none"))

    ax.text(0.5, 0.88, "SIMULATION STUDY REPORT", ha="center", va="center",
            fontsize=12, color="#94A3B8", fontfamily="monospace")

    for i, line in enumerate(title.split("\n")):
        ax.text(0.5, 0.72 - i * 0.07, line, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white",
                wrap=True)

    ax.text(0.5, 0.50, subtitle, ha="center", va="center",
            fontsize=11, color="#EF4444", fontstyle="italic")

    ax.text(0.5, 0.42, "⚠  ALL DATA IN THIS REPORT IS SYNTHETICALLY GENERATED  ⚠",
            ha="center", va="center", fontsize=10, color="#F59E0B",
            bbox=dict(facecolor="#1E293B", edgecolor="#F59E0B", boxstyle="round,pad=0.4"))

    info_lines = [
        f"Generated: {DATE}",
        "Seed: 42 | 10,000 synthetic workers | 90-day window",
        "Framework: XGBoost + SHAP | UPI + Mobile Recharge features",
        "Purpose: Methodological feasibility study for research paper",
    ]
    for i, line in enumerate(info_lines):
        ax.text(0.5, 0.32 - i * 0.04, line, ha="center", va="center",
                fontsize=9, color="#CBD5E1")

    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.114), 0.90, 0.004,
                                          boxstyle="round,pad=0.001",
                                          facecolor="#3B82F6", edgecolor="none"))

    ax.text(0.5, 0.08, "Results indicate methodological feasibility only — not real-world performance.",
            ha="center", va="center", fontsize=8, color="#64748B", fontstyle="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_section_header(pdf, section_num, title, color="#1E3A5F"):
    fig = plt.figure(figsize=(8.5, 1.5))
    fig.patch.set_facecolor(color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.6, f"Section {section_num}", fontsize=10, color="#94A3B8",
            transform=ax.transAxes)
    ax.text(0.05, 0.2, title, fontsize=14, fontweight="bold", color="white",
            transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_text_page(pdf, content_lines, title="", fontsize=9):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.90])
    ax.axis("off")
    if title:
        ax.text(0.0, 1.0, title, fontsize=12, fontweight="bold",
                va="top", transform=ax.transAxes, color="#1E3A5F")
    y = 0.94
    for line in content_lines:
        wrapped = textwrap.fill(line, width=95)
        for sub in wrapped.split("\n"):
            ax.text(0.0, y, sub, fontsize=fontsize, va="top",
                    transform=ax.transAxes, color="#1E293B")
            y -= 0.028
            if y < 0:
                break
        if y < 0:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf, df, title="", col_widths=None, fontsize=8):
    n_cols = len(df.columns)
    n_rows = len(df)
    fig_h = max(3.5, min(11, 1.5 + n_rows * 0.35))
    fig = plt.figure(figsize=(8.5, fig_h))
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.88])
    ax.axis("off")

    if title:
        ax.text(0.5, 1.0, title, ha="center", va="top", fontsize=10,
                fontweight="bold", transform=ax.transAxes, color="#1E3A5F")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.auto_set_column_width(col=list(range(n_cols)))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F1F5F9")
        cell.set_edgecolor("#E2E8F0")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, img_path, caption="", title=""):
    img = plt.imread(str(img_path))
    h, w = img.shape[:2]
    aspect = h / w
    fig_w = 7.5
    fig_h = min(9, fig_w * aspect + 1.0)
    fig = plt.figure(figsize=(8.5, max(4, fig_h + 1)))
    ax_img = fig.add_axes([0.05, 0.12, 0.90, 0.82])
    ax_img.imshow(img)
    ax_img.axis("off")
    if title:
        ax_img.set_title(title, fontsize=10, fontweight="bold", color="#1E3A5F", pad=8)
    if caption:
        fig.text(0.5, 0.05, caption, ha="center", va="bottom",
                 fontsize=8, fontstyle="italic", color="#475569",
                 wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def format_shap_explanation(tier, ex):
    lines = [
        f"--- {tier} Worker ---",
        f"  Credit Score : {ex['score']:.1f} / 100",
        f"  P(default)   : {ex['prob']:.4f}",
        f"  SHAP base val: {ex['base_value']:.4f}",
        f"  Sum(SHAP)    : {ex['shap_sum']:.4f}  [sanity: diff from base = {ex['diff_from_base']:.4f}]",
        "",
        "  Top driving features:",
    ]
    for feat_info in ex["top_features"]:
        fname = feat_info["feature"]
        plain = PLAIN_LANG.get(fname, fname)
        contrib = feat_info["score_contribution_pts"]
        sign = "+" if contrib >= 0 else ""
        lines.append(
            f"    {sign}{contrib:.1f} pts ← {plain} = {feat_info['feature_value']:.2f}"
            f"  (SHAP={feat_info['shap_value']:+.4f})"
        )
    return lines


def build_pdf(results_dir: Path, fig_paths: dict, exp_outputs: dict):
    res = exp_outputs["results"]
    pdf_path = results_dir / "report.pdf"

    with PdfPages(pdf_path) as pdf:
        # ── Page 1: Title ──────────────────────────────────────────────────
        add_title_page(pdf)

        # ── Section 1: Experimental Setup ─────────────────────────────────
        add_section_header(pdf, 1, "Experimental Setup and Data-Generation Description")

        setup_text = [
            "1.1  OVERVIEW",
            "",
            "This is a calibrated synthetic-data simulation study.  No real data was used.",
            "All results indicate methodological feasibility only and must NOT be interpreted",
            "as real-world performance.  The word 'synthetic' appears in every table and figure",
            "caption involving simulated data.",
            "",
            "1.2  DATA GENERATION PARAMETERS",
            "",
            "  • Total workers: 10,000  (5,000 repaid / 5,000 defaulted — balanced experiment)",
            "  • Secondary experiment: 5,000 workers, 80/20 imbalance (20% default)",
            "  • Worker categories: street vendor (30%), daily-wage labourer (30%),",
            "    gig/delivery worker (25%), domestic worker (15%)",
            "  • Observation window: 90 days (2024-01-01 to 2024-04-01)",
            "  • Random seed: 42 (everything reproducible)",
            "",
            "1.3  LATENT REPAYMENT CAPACITY",
            "",
            "  Each worker is assigned a latent repayment capacity ∈ [0,1] drawn from",
            "  Beta(2,2) mixed with normalised income and digital-adoption rate.  This",
            "  unobserved variable drives both the volume/regularity of UPI activity and",
            "  the probability of default.  About 15% of workers experience random income",
            "  shocks that reduce capacity.  Label noise of ~6% is applied (labels flipped)",
            "  so the task is not trivially separable.",
            "",
            "1.4  UPI EVENT GENERATION",
            "",
            "  • Amounts: log-normally distributed, INR 20–5,000",
            "  • Weekly pattern: higher Mon–Sat, lower Sunday",
            "  • Festival bumps: days 15–16, 45–46, 72–73 (1.5×–2.5× amount)",
            "  • Sparse workers: those with low digital-adoption × capacity can have 0–7 events",
            "  • Credit probability: 0.45 + 0.2 × capacity",
            "  • Counterparty pool: 3–15 unique per worker",
            "",
            "1.5  RECHARGE EVENT GENERATION",
            "",
            "  • Plans: INR 19, 29, 49, 79, 99, 149, 199, 299, 399, 599, 799",
            "  • Frequency: 0.9–2.0 events/month depending on category",
            "  • Timestamps: roughly periodic with Gaussian noise (σ = 30% of interval)",
            "  • Higher-capacity workers systematically prefer more expensive plans",
            "",
            "1.6  WORKER-CATEGORY INCOME PARAMETERS",
            "",
            "  Category              | Income (INR/mo) | Digital Adoption | UPI freq/day",
            "  ─────────────────────────────────────────────────────────────────────────",
            "  Street vendor         |   12,000 ± 4,000|      65%         |    1.8",
            "  Daily-wage labourer   |    9,000 ± 3,000|      45%         |    0.9",
            "  Gig / delivery worker |   16,000 ± 5,000|      90%         |    3.2",
            "  Domestic worker       |    8,000 ± 2,500|      40%         |    0.7",
            "",
            "1.7  DESIGN DECISIONS",
            "",
            "  • Data generator was NOT tuned after seeing model results (hard rule).",
            "  • Missingness indicators (not silent zero-imputation) used for sparse features.",
            "  • Robust scaling applied only to the Logistic Regression baseline.",
            "  • SHAP background / explanation samples capped at 1,000 for speed.",
            "  • XGBoost RandomizedSearchCV: n_iter=20, optimising AUC, n_jobs=-1.",
        ]
        add_text_page(pdf, setup_text, fontsize=8.5)

        # ── Section 2: Dataset Summary ─────────────────────────────────────
        add_section_header(pdf, 2, "Dataset Summary (Synthetic)")

        si = res["split_info"]
        ds_rows = [
            ["Total workers",          "10,000"],
            ["Label=0 (repaid)",        "5,000"],
            ["Label=1 (defaulted)",     "5,000"],
            ["Label noise applied",    "~6% (random flip)"],
            ["Observation window",     "90 days"],
            ["Train size",             str(si["train_size"])],
            ["Test size",              str(si["test_size"])],
            ["Train positive rate",    f"{si['train_pos_rate']:.3f}"],
            ["Test positive rate",     f"{si['test_pos_rate']:.3f}"],
            ["UPI events (approx)",    "~500,000"],
            ["Recharge events (approx)","~50,000"],
            ["Total features",         str(len(si["feature_cols"]))],
            ["CV strategy",            "Stratified 5-fold inside training set"],
            ["XGBoost search",         "RandomizedSearchCV, n_iter=20"],
            ["Seed",                   "42"],
        ]
        ds_df = pd.DataFrame(ds_rows, columns=["Parameter", "Value"])
        add_table_page(pdf, ds_df,
                       title="Table 1 — Dataset Summary (Synthetic data — methodological feasibility only)")

        # ── Section 3: Metrics ─────────────────────────────────────────────
        add_section_header(pdf, 3, "Model Performance Metrics")

        metrics_df = pd.DataFrame(res["main_metrics"])
        add_table_page(pdf, metrics_df,
                       title="Table 2 — Main Model Metrics on Held-out Test Set (Synthetic data)")

        # Calibration sub-table
        cal = res["calibration"]
        cal_df = pd.DataFrame([{
            "Model": "XGBoost",
            "Brier (before cal.)": cal["brier_before_calibration"],
            "Brier (after isotonic)": cal["brier_after_calibration"],
            "Improvement": round(cal["brier_before_calibration"] - cal["brier_after_calibration"], 4),
        }])
        add_table_page(pdf, cal_df,
                       title="Table 3 — Calibration: Brier Score Before/After Isotonic Regression (Synthetic data)")

        # Tier table
        tier_df = pd.DataFrame(res["tier_stats"])
        tier_df["default_rate"] = tier_df["default_rate"].apply(lambda x: f"{x:.4f}")
        tier_df["mean_score"]   = tier_df["mean_score"].apply(lambda x: f"{x:.2f}")
        monotonic_note = "✓ Monotonic" if res["tier_monotonic"] else "✗ NOT monotonic — reported as-is"
        add_table_page(pdf, tier_df,
                       title=f"Table 4 — Tier Analysis: Default Rate per Score Tier — {monotonic_note} (Synthetic data)")

        # Tier clf metrics
        if res.get("tier_clf_metrics"):
            tcm = res["tier_clf_metrics"]
            tcm_df = pd.DataFrame([{
                k: v for k, v in tcm.items()
            }]).T.reset_index()
            tcm_df.columns = ["Metric", "Value"]
            add_table_page(pdf, tcm_df,
                           title="Table 5 — Tier Classification: Eligible vs High-Risk (Synthetic data)")

        # Multi-seed
        ms = res["multi_seed"]
        ms_rows = []
        for i, s in enumerate(ms["seeds"]):
            ms_rows.append({
                "Seed": s,
                "XGBoost AUC": ms["xgb_aucs"][i],
                "LR AUC": ms["lr_aucs"][i],
            })
        ms_df = pd.DataFrame(ms_rows)
        ms_df.loc[len(ms_df)] = {
            "Seed": "Mean ± Std",
            "XGBoost AUC": f"{ms['xgb_mean_auc']:.4f} ± {ms['xgb_std_auc']:.4f}",
            "LR AUC": f"{ms['lr_mean_auc']:.4f} ± {ms['lr_std_auc']:.4f}",
        }
        add_table_page(pdf, ms_df,
                       title="Table 6 — Multi-seed Stability (5 seeds, XGBoost vs LR) (Synthetic data)")

        # External comparison
        ext_df = pd.DataFrame(res["external_references"])
        add_table_page(pdf, ext_df,
                       title="Table 7 — External Reference Comparison (external results from different real-world data)")

        # XGBoost best params
        hp_rows = [{"Hyperparameter": k, "Value": str(v)} for k, v in res["xgb_best_params"].items()]
        hp_df = pd.DataFrame(hp_rows)
        add_table_page(pdf, hp_df,
                       title=f"Table 8 — XGBoost Best Hyperparameters (CV AUC={res['xgb_cv_auc']:.4f}) (Synthetic data)")

        # ── Section 4: Figures ─────────────────────────────────────────────
        add_section_header(pdf, 4, "Figures")

        add_image_page(pdf, fig_paths["roc"],
                       caption="Figure 1 — ROC curves for XGBoost, Logistic Regression, and Random Forest. Synthetic data — methodological feasibility only.",
                       title="Figure 1 — ROC Curves (Synthetic Data)")

        add_image_page(pdf, fig_paths["score_dist"],
                       caption="Figure 2 — Distribution of credit scores (0–100) by true class. Dashed lines indicate tier boundaries. Synthetic data — methodological feasibility only.",
                       title="Figure 2 — Score Distribution by True Class (Synthetic Data)")

        add_image_page(pdf, fig_paths["calibration"],
                       caption="Figure 3 — Reliability (calibration) curve for XGBoost, before and after isotonic regression. Synthetic data — methodological feasibility only.",
                       title="Figure 3 — Calibration Curve (Synthetic Data)")

        # ── Section 5: Ablation and Cold-start ────────────────────────────
        add_section_header(pdf, 5, "Ablation and Cold-start Results")

        abl_df = pd.DataFrame([
            {k: v for k, v in m.items() if k not in ("model",)}
            for m in res["ablation"].values()
        ])
        abl_df.insert(0, "Feature Set", list(res["ablation"].keys()))
        add_table_page(pdf, abl_df,
                       title="Table 9 — Ablation: XGBoost with UPI-only, Recharge-only, Combined (Synthetic data)")

        add_image_page(pdf, fig_paths["ablation"],
                       caption="Figure 4 — Ablation study: AUC and Brier score for UPI-only, Recharge-only, and Combined features. Synthetic data — methodological feasibility only.",
                       title="Figure 4 — Ablation Study (Synthetic Data)")

        add_image_page(pdf, fig_paths["tier"],
                       caption="Figure 5 — Worker count and default rate per score tier. Synthetic data — methodological feasibility only.",
                       title="Figure 5 — Tier Analysis (Synthetic Data)")

        cs = res["cold_start"]
        cs_rows = [{"Metric": k, "Value": str(v)} for k, v in cs.items()]
        cs_df = pd.DataFrame(cs_rows)
        add_table_page(pdf, cs_df,
                       title="Table 10 — Cold-start Analysis: Workers with <15 UPI Transactions (Synthetic data)")

        cs_text = [
            "Cold-start test: workers with fewer than 15 UPI transactions in the 90-day window.",
            f"  • Number of cold-start workers in test set: {cs['n_cold_workers']}",
            f"  • UPI-only AUC on cold-start workers:  {cs.get('UPI_only_AUC', 'N/A')}",
            f"  • Combined AUC on cold-start workers:  {cs.get('Combined_AUC', 'N/A')}",
            f"  • UPI-only Brier on cold-start workers: {cs.get('UPI_only_Brier', 'N/A')}",
            f"  • Combined Brier on cold-start workers: {cs.get('Combined_Brier', 'N/A')}",
            "",
            "Interpretation: If Combined AUC > UPI-only AUC on cold-start workers, this",
            "supports the paper's claim that recharge data provides meaningful signal for",
            "workers with sparse UPI histories.  The actual direction is reported as computed.",
            "",
            "Imbalanced-data Experiment (80/20 ratio):",
            f"  • Workers: {res['imbalance_experiment']['n_workers']}",
            f"  • Positive rate: {res['imbalance_experiment']['pos_rate']}",
            f"  • XGBoost AUC: {res['imbalance_experiment']['AUC']}",
            f"  • XGBoost Brier: {res['imbalance_experiment']['Brier']}",
        ]
        add_text_page(pdf, cs_text, title="Cold-start and Imbalanced-data Results (Synthetic)", fontsize=9)

        add_image_page(pdf, fig_paths["multi_seed"],
                       caption="Figure 6 — Multi-seed stability: XGBoost and LR test AUC across 5 different random seeds. Synthetic data — methodological feasibility only.",
                       title="Figure 6 — Multi-seed Stability (Synthetic Data)")

        # ── Section 6: Fairness ────────────────────────────────────────────
        add_section_header(pdf, 6, "Fairness Analysis")

        fair = res["fairness"]
        fair_rows = []
        for grp, rate in fair["income_tercile_approval_rates"].items():
            fair_rows.append({"Group Type": "Income Tercile", "Group": grp,
                               "Approval Rate": f"{rate:.4f}"})
        for wt, rate in fair["worker_type_approval_rates"].items():
            fair_rows.append({"Group Type": "Worker Category", "Group": wt,
                               "Approval Rate": f"{rate:.4f}"})
        fair_df = pd.DataFrame(fair_rows)
        add_table_page(pdf, fair_df,
                       title="Table 11 — Fairness: Eligible-tier Approval Rates by Group (Synthetic data)")

        dir_text = [
            "Disparate Impact Ratio (DIR) = min(group_rate) / max(group_rate)",
            "Acceptable range per guidelines: 0.80 – 1.25",
            "",
            f"  DIR by Income Tercile   : {fair['disparate_impact_ratio_income']}",
            f"  DIR by Worker Category  : {fair['disparate_impact_ratio_workertype']}",
            "",
            "Note: A DIR < 0.80 suggests potential adverse impact on a disadvantaged subgroup.",
            "A DIR > 1.25 suggests preferential treatment.  Actual values are reported as",
            "computed from the synthetic simulation without post-hoc adjustment.",
            "",
            "Limitations: In a real deployment, additional protected attributes (gender, caste,",
            "region) and intersectional analysis would be required.  The synthetic data does",
            "not encode protected attributes beyond worker category and income.",
        ]
        add_text_page(pdf, dir_text, title="Fairness Interpretation (Synthetic Data)", fontsize=9)

        add_image_page(pdf, fig_paths["fairness"],
                       caption="Figure 7 — Approval rates (Eligible tier) by income tercile and worker category. Synthetic data — methodological feasibility only.",
                       title="Figure 7 — Fairness Analysis (Synthetic Data)")

        # ── Section 7: SHAP Explainability ────────────────────────────────
        add_section_header(pdf, 7, "SHAP Explainability")

        add_image_page(pdf, fig_paths["shap_bar"],
                       caption="Figure 8 — Global SHAP feature importance: mean |SHAP value| across test sample. Synthetic data — methodological feasibility only.",
                       title="Figure 8 — SHAP Global Bar Plot (Synthetic Data)")

        add_image_page(pdf, fig_paths["shap_beeswarm"],
                       caption="Figure 9 — SHAP beeswarm summary: each point is a worker, colour indicates feature value. Synthetic data — methodological feasibility only.",
                       title="Figure 9 — SHAP Beeswarm Summary (Synthetic Data)")

        # Individual SHAP explanations
        shap_text = ["SHAP Individual Explanations (scaled to score contribution points)",
                     "Formula: score_pts ≈ SHAP_value × (−100 × p × (1−p))",
                     "Sanity check: sum(SHAP values) ≈ model_output − base_value", ""]
        for tier_name, ex in exp_outputs["individual_examples"].items():
            shap_text.extend(format_shap_explanation(tier_name, ex))
            shap_text.append("")

        add_text_page(pdf, shap_text,
                      title="Individual SHAP Explanations — Synthetic Data", fontsize=8.5)

        # ── Section 8: Latency ─────────────────────────────────────────────
        add_section_header(pdf, 8, "Operational Latency")

        lat = res["latency_ms"]
        lat_text = [
            "Single-worker prediction + SHAP explanation latency (200 repetitions, in-process):",
            "",
            f"  Mean latency : {lat['mean']:.2f} ms",
            f"  P95 latency  : {lat['p95']:.2f} ms",
            "",
            "Measurement method: Python time.perf_counter() around predict_proba() + TreeExplainer()",
            "for a single-row input.  No network overhead.  Measured on the development machine.",
            "Hardware: laptop CPU (x86_64), no GPU.",
            "",
            "These latency figures are suitable for real-time loan application scoring,",
            "where sub-second response is typically required.",
        ]
        add_text_page(pdf, lat_text, title="Section 8 — Operational Latency", fontsize=9)

        # ── Section 9: Limitations ─────────────────────────────────────────
        add_section_header(pdf, 9, "Limitations and Threats to Validity", color="#7C2D12")

        limitations_text = [
            "⚠  CRITICAL DISCLAIMER  ⚠",
            "",
            "ALL DATA IN THIS STUDY IS ENTIRELY SYNTHETIC.  No real UPI transaction records,",
            "mobile recharge histories, or loan-repayment outcomes were used.  The synthetic",
            "generator is calibrated to approximate realistic statistical properties of informal",
            "workers in India, but it cannot capture the full complexity of real-world data.",
            "",
            "RESULTS INDICATE METHODOLOGICAL FEASIBILITY ONLY — NOT REAL-WORLD PERFORMANCE.",
            "",
            "SPECIFIC THREATS TO VALIDITY:",
            "",
            "  1. Synthetic data bias: The generator's distributional assumptions may not",
            "     match the true joint distribution of UPI behaviour and creditworthiness.",
            "     Real data may reveal different feature importance rankings.",
            "",
            "  2. Label noise simulation: ~6% label flip is an approximation.  Real default",
            "     labels may have higher or structured noise (e.g., reporting lag).",
            "",
            "  3. No temporal leakage check: In a real deployment, strict point-in-time",
            "     splits would be required to prevent look-ahead bias.",
            "",
            "  4. Synthetic feature correlations: The generator creates correlations through",
            "     the latent capacity variable, but real-world correlations may differ",
            "     substantially in magnitude and direction.",
            "",
            "  5. Fairness analysis scope: Only income tercile and worker category are",
            "     analysed.  Real-world fairness audits require protected attributes",
            "     (gender, geography, caste, religion) and intersectional analysis.",
            "",
            "  6. Regulatory compliance: Credit-scoring models in India are subject to",
            "     RBI/NBFC regulations.  A production system would require extensive",
            "     regulatory review independent of this simulation.",
            "",
            "  7. Feature engineering assumptions: EMI detection, income estimation,",
            "     and disposable surplus are approximations from UPI metadata only.",
            "     Real implementations would require bank-account level data.",
            "",
            "  8. Calibration validity: Isotonic regression calibration on the test set",
            "     is shown for illustration; in production it must be validated on a",
            "     separate held-out calibration set.",
            "",
            "  9. External comparison table: The reference AUC/Brier values (Yadav et al.,",
            "     Ots et al., Ng et al.) are from different populations, geographies, and",
            "     feature sets.  Direct numerical comparison is not valid.",
            "",
            " 10. Reproducibility: Results are fully reproducible with seed=42 on the same",
            "     Python/library versions listed in requirements.txt.",
        ]
        add_text_page(pdf, limitations_text,
                      title="Section 9 — Limitations and Threats to Validity", fontsize=8.5)

        # Set PDF metadata
        d = pdf.infodict()
        d["Title"] = "Informal Worker Income Verification — Synthetic Data Simulation Study"
        d["Author"] = "Simulation Study — Synthetic Data Only"
        d["Subject"] = "ML Credit Scoring Simulation"
        d["Keywords"] = "synthetic data, credit scoring, XGBoost, SHAP, UPI, microfinance"
        d["Creator"] = "run_all.py"

    print(f"\nPDF written to: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    print("PDF builder requires exp_outputs dict from experiments.py")
