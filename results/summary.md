# Key Numbers — UPI Credit Scoring Simulation Study (SYNTHETIC DATA)

## Main Model Metrics (Table 2, held-out test set)
- XGBoost AUC       : 0.7214   (Fig 1, Table 2)
- XGBoost Brier     : 0.2072   (Table 2, 3)
- XGBoost F1        : 0.6621   (Table 2)
- XGBoost Accuracy  : 0.6525   (Table 2)
- LR AUC            : 0.6977   (Table 2)
- RF AUC            : 0.6993   (Table 2)

## Calibration (Table 3)
- Brier before isotonic : 0.20724889636039734   (Table 3, Fig 3)
- Brier after isotonic  : 0.2023797333240509   (Table 3, Fig 3)

## Multi-seed Stability (Table 6)
- XGBoost AUC: 0.7263 ± 0.0087   (Table 6, Fig 6)
- LR AUC:      0.6934 ± 0.0131   (Table 6, Fig 6)

## Tier Analysis (Table 4, Fig 5)
- Eligible  : count=236, default_rate=0.1483, mean_score=84.68
- Review    : count=1362, default_rate=0.4670, mean_score=51.65
- High_risk : count=402, default_rate=0.8259, mean_score=23.58
- Tier monotonicity: YES

## Ablation (Table 9, Fig 4)
- UPI-only   AUC: 0.7124
- Recharge-only AUC: 0.5447
- Combined   AUC: 0.7234

## Cold-start (Table 10)
- Cold-start workers (<15 UPI txns): 1042
- UPI-only AUC (cold): 0.5126
- Combined AUC (cold): 0.5468

## Fairness (Table 11, Fig 7)
- Disparate Impact Ratio (income tercile): 0.0
- Disparate Impact Ratio (worker type):    0.0
- Acceptable range: 0.80–1.25

## Latency (Section 8)
- Mean latency (predict + SHAP): 11.21 ms
- P95 latency  (predict + SHAP): 11.73 ms

## Imbalanced-data Experiment (80/20)
- AUC: 0.677   Brier: 0.1519

---
NOTE: ALL DATA IS SYNTHETIC. All results indicate methodological feasibility only.
See results/report.pdf for full details and limitations.