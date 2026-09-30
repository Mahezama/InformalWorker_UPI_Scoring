# UPI Credit Scoring — Simulation Study

> **All data is SYNTHETIC. Results indicate methodological feasibility only.**

## Single command to reproduce everything

```bash
python run_all.py
```

This will:
1. Generate 10,000 synthetic workers with UPI + recharge event logs (seed=42)
2. Compute 28 features per worker from raw events
3. Train XGBoost (RandomizedSearchCV), Logistic Regression, and Random Forest
4. Run ablation (UPI-only / Recharge-only / Combined), cold-start, multi-seed stability
5. Compute SHAP explanations, fairness metrics, and operational latency
6. Save all results to `results/`

## Launch Interactive Web App & Mentor Demo

To showcase the live simulation, real-time credit score gauge, SHAP waterfall explanations, and cohort explorer to your mentor:

```bash
python serve.py
```

Or visit: **[http://localhost:8080](http://localhost:8080)** in any browser.

Features included in the Web App:
- **Interactive Credit Scoring Simulator**: Real-time 0–100 score gauge, tier classification (`Eligible`, `Review`, `High Risk`), suggested loan limits, and plain-language SHAP attribution factors.
- **Quick Archetype Personas**: 🛵 Gig Worker, 🥘 Street Vendor, 🧹 Domestic Worker, 🔨 Daily-Wage Labourer.
- **Empirical Research Dashboard**: Full benchmark metric tables (Table 2, 4, 7, 9) and KPI cards.
- **300 DPI Publication Figure Gallery**: Interactive lightbox for all 9 manuscript figures.
- **Worker Cohort Explorer**: Filterable, searchable inspector for 100 sample workers with one-click transfer into the live scoring engine.
- **Direct PDF Report Access**: One-click download/view of `results/report.pdf`.

## Outputs

| File | Description |
|------|-------------|
| `results/report.pdf` | Full PDF report (title page + 9 sections) |
| `results/metrics.json` | All computed metrics in JSON |
| `results/main_metrics.csv` | AUC / Brier / F1 table |
| `results/tier_stats.csv` | Default rate per score tier |
| `results/ablation.csv` | Ablation experiment results |
| `results/external_comparison.csv` | Our results vs. external references |
| `results/summary.md` | Key numbers for copy-paste into paper |
| `results/figs/*.png` | All figures at 300 dpi |
| `data/upi_events.csv` | Raw synthetic UPI event log |
| `data/recharge_events.csv` | Raw synthetic recharge event log |
| `data/features.csv` | Computed feature matrix (one row per worker) |

## Requirements

```
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
```

Install: `pip install numpy pandas scikit-learn xgboost shap matplotlib`

## Estimated runtime

~5–8 minutes on a modern laptop CPU (most time: RandomizedSearchCV + SHAP).

## Reproducibility

- Random seed: **42** everywhere
- Python >= 3.9, XGBoost >= 1.7, scikit-learn >= 1.1, SHAP >= 0.41
- `n_jobs=-1` (uses all CPU cores)

## Key design decisions (noted in report Section 1)

1. Balanced 50/50 train split + secondary 80/20 imbalance experiment
2. Label noise ~6% applied to prevent trivial separability
3. Latent capacity variable drives both behaviour and default probability
4. Missingness indicators instead of silent zero-imputation
5. Robust scaling for Logistic Regression only
6. SHAP background sample capped at 500, explanation sample at 1,000

## ⚠ Disclaimer

ALL DATA IS SYNTHETICALLY GENERATED. No real UPI records or loan data were used.
Results indicate methodological feasibility only — not real-world performance.
See `results/report.pdf` Section 9 (Limitations) for full threat-to-validity analysis.
