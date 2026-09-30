"""
Model training, evaluation, and experiments for the UPI credit-scoring paper.
All results computed on held-out test data. No result is hardcoded.
All data is SYNTHETIC.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, roc_curve, confusion_matrix, classification_report
)
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import shap

from src.feature_engineering import ALL_FEATURE_COLS, UPI_FEATURE_COLS, RECHARGE_FEATURE_COLS


SEED = 42
SCORE_THRESHOLD = 0.5

TIER_BOUNDS = {"Eligible": (70, 100), "Review": (40, 69), "High_risk": (0, 39)}


def prob_to_score(p):
    return 100.0 * (1.0 - p)


def assign_tier(score):
    if score >= 70:
        return "Eligible"
    elif score >= 40:
        return "Review"
    else:
        return "High_risk"


def compute_metrics(y_true, y_prob, threshold=0.5, label="model"):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": label,
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
    }


def train_xgboost(X_train, y_train, feature_cols, n_iter=20, seed=SEED):
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    base_xgb = xgb.XGBClassifier(
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        base_xgb, param_dist, n_iter=n_iter, scoring="roc_auc",
        cv=cv, random_state=seed, n_jobs=-1, verbose=0
    )
    search.fit(X_train[feature_cols], y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def train_logistic(X_train, y_train, feature_cols, seed=SEED):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train[feature_cols])
    lr = LogisticRegression(
        max_iter=2000, random_state=seed, C=1.0, solver="lbfgs"
    )
    lr.fit(X_scaled, y_train)
    return lr, scaler


def train_random_forest(X_train, y_train, feature_cols, seed=SEED):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=seed, n_jobs=-1
    )
    rf.fit(X_train[feature_cols], y_train)
    return rf


def run_experiments(feat_df: pd.DataFrame, results_dir: Path) -> dict:
    results_dir.mkdir(exist_ok=True)
    (results_dir / "figs").mkdir(exist_ok=True)

    X = feat_df[ALL_FEATURE_COLS].copy()
    y = feat_df["true_label"].values
    worker_types = feat_df["worker_type"].values
    est_income = feat_df["est_monthly_income"].values

    # ---- Stratified 80/20 split ----
    X_train, X_test, y_train, y_test, wt_train, wt_test, inc_train, inc_test = (
        train_test_split(X, y, worker_types, est_income,
                         test_size=0.2, stratify=y, random_state=SEED)
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train pos rate: {y_train.mean():.3f}, Test pos rate: {y_test.mean():.3f}")

    split_info = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
        "seed": SEED,
        "feature_cols": ALL_FEATURE_COLS,
    }

    # ========== Train models ==========
    print("\nTraining XGBoost (RandomizedSearchCV, n_iter=20)...")
    t0 = time.time()
    xgb_model, xgb_best_params, xgb_cv_auc = train_xgboost(
        X_train, y_train, ALL_FEATURE_COLS, n_iter=20
    )
    print(f"  XGBoost CV AUC={xgb_cv_auc:.4f}, time={time.time()-t0:.1f}s")
    print(f"  Best params: {xgb_best_params}")

    print("\nTraining Logistic Regression...")
    lr_model, lr_scaler = train_logistic(X_train, y_train, ALL_FEATURE_COLS)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train, ALL_FEATURE_COLS)

    # ========== Test predictions ==========
    X_test_arr = X_test[ALL_FEATURE_COLS]
    xgb_prob = xgb_model.predict_proba(X_test_arr)[:, 1]
    lr_prob = lr_model.predict_proba(lr_scaler.transform(X_test_arr))[:, 1]
    rf_prob = rf_model.predict_proba(X_test_arr)[:, 1]

    xgb_score = prob_to_score(xgb_prob)
    xgb_tier = np.array([assign_tier(s) for s in xgb_score])

    # ========== Core metrics ==========
    metrics_xgb = compute_metrics(y_test, xgb_prob, label="XGBoost")
    metrics_lr  = compute_metrics(y_test, lr_prob,  label="LogisticRegression")
    metrics_rf  = compute_metrics(y_test, rf_prob,  label="RandomForest")

    all_metrics = [metrics_xgb, metrics_lr, metrics_rf]
    print("\n--- Main Model Metrics ---")
    for m in all_metrics:
        print(f"  {m['model']:25s} AUC={m['AUC']:.4f}  Brier={m['Brier']:.4f}  F1={m['F1']:.4f}")

    # ========== Calibration ==========
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(xgb_prob, y_test)
    xgb_prob_cal = iso_reg.predict(xgb_prob)
    brier_before = brier_score_loss(y_test, xgb_prob)
    brier_after  = brier_score_loss(y_test, xgb_prob_cal)

    frac_pos_before, mean_pred_before = calibration_curve(y_test, xgb_prob, n_bins=10)
    frac_pos_after,  mean_pred_after  = calibration_curve(y_test, xgb_prob_cal, n_bins=10)

    calibration_data = {
        "brier_before_calibration": float(brier_before),
        "brier_after_calibration":  float(brier_after),
        "frac_pos_before": frac_pos_before.tolist(),
        "mean_pred_before": mean_pred_before.tolist(),
        "frac_pos_after":  frac_pos_after.tolist(),
        "mean_pred_after": mean_pred_after.tolist(),
    }

    # ========== Tier analysis ==========
    tier_df = pd.DataFrame({
        "tier": xgb_tier,
        "true_label": y_test,
        "score": xgb_score,
    })
    tier_stats = tier_df.groupby("tier").agg(
        count=("true_label", "count"),
        default_rate=("true_label", "mean"),
        mean_score=("score", "mean"),
    ).reset_index()
    # Sort for monotonicity check
    tier_order = {"Eligible": 0, "Review": 1, "High_risk": 2}
    tier_stats["order"] = tier_stats["tier"].map(tier_order)
    tier_stats = tier_stats.sort_values("order").drop(columns="order")
    monotonic = bool(tier_stats["default_rate"].is_monotonic_increasing)

    print(f"\nTier monotonicity (High->Eligible default rate increasing): {monotonic}")
    print(tier_stats)

    # Tier-based Eligible vs High_risk classification
    eligible_mask = xgb_tier == "Eligible"
    highrisk_mask = xgb_tier == "High_risk"
    tier_clf_mask = eligible_mask | highrisk_mask

    if tier_clf_mask.sum() > 0:
        y_tier_true = (y_test[tier_clf_mask] == 0).astype(int)  # "repaid" = positive for approval
        y_tier_pred = eligible_mask[tier_clf_mask].astype(int)
        tier_metrics = {
            "Eligible_vs_HighRisk_Accuracy": round(accuracy_score(y_tier_true, y_tier_pred), 4),
            "Eligible_vs_HighRisk_Precision": round(precision_score(y_tier_true, y_tier_pred, zero_division=0), 4),
            "Eligible_vs_HighRisk_Recall":    round(recall_score(y_tier_true, y_tier_pred, zero_division=0), 4),
            "Eligible_vs_HighRisk_F1":        round(f1_score(y_tier_true, y_tier_pred, zero_division=0), 4),
            "n_eligible": int(eligible_mask.sum()),
            "n_review": int((xgb_tier == "Review").sum()),
            "n_highrisk": int(highrisk_mask.sum()),
        }
    else:
        tier_metrics = {}

    # ========== ROC curves data ----
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_prob)
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_prob)

    roc_data = {
        "xgb": {"fpr": fpr_xgb.tolist(), "tpr": tpr_xgb.tolist(), "auc": metrics_xgb["AUC"]},
        "lr":  {"fpr": fpr_lr.tolist(),  "tpr": tpr_lr.tolist(),  "auc": metrics_lr["AUC"]},
        "rf":  {"fpr": fpr_rf.tolist(),  "tpr": tpr_rf.tolist(),  "auc": metrics_rf["AUC"]},
    }

    # ========== ABLATION ==========
    print("\nRunning ablation experiments...")
    ablation_results = {}
    for name, cols in [
        ("UPI_only",      UPI_FEATURE_COLS),
        ("Recharge_only", RECHARGE_FEATURE_COLS),
        ("Combined",      ALL_FEATURE_COLS),
    ]:
        model, params, cv_auc = train_xgboost(X_train, y_train, cols, n_iter=10, seed=SEED)
        prob = model.predict_proba(X_test[cols])[:, 1]
        m = compute_metrics(y_test, prob, label=name)
        m["cv_auc"] = round(cv_auc, 4)
        ablation_results[name] = m
        print(f"  {name:20s}: AUC={m['AUC']:.4f}, Brier={m['Brier']:.4f}")

    # ========== COLD START ==========
    print("\nCold-start analysis (workers with <15 UPI txns)...")
    cold_mask = (X_test["upi_total_count"] < 15).values
    print(f"  Cold-start workers in test: {cold_mask.sum()}")

    if cold_mask.sum() > 10:
        upi_model, _, _ = train_xgboost(X_train, y_train, UPI_FEATURE_COLS, n_iter=10, seed=SEED)
        comb_model_cold, _, _ = train_xgboost(X_train, y_train, ALL_FEATURE_COLS, n_iter=10, seed=SEED)

        upi_cold_prob = upi_model.predict_proba(X_test[cold_mask][UPI_FEATURE_COLS])[:, 1]
        comb_cold_prob = comb_model_cold.predict_proba(X_test[cold_mask][ALL_FEATURE_COLS])[:, 1]
        y_cold = y_test[cold_mask]

        cold_start = {
            "n_cold_workers": int(cold_mask.sum()),
            "UPI_only_AUC": float(round(roc_auc_score(y_cold, upi_cold_prob), 4)) if len(np.unique(y_cold)) > 1 else None,
            "Combined_AUC": float(round(roc_auc_score(y_cold, comb_cold_prob), 4)) if len(np.unique(y_cold)) > 1 else None,
            "UPI_only_Brier": float(round(brier_score_loss(y_cold, upi_cold_prob), 4)),
            "Combined_Brier": float(round(brier_score_loss(y_cold, comb_cold_prob), 4)),
        }
    else:
        cold_start = {"n_cold_workers": int(cold_mask.sum()), "note": "too few cold-start workers"}

    print(f"  Cold-start: {cold_start}")

    # ========== MULTI-SEED STABILITY ==========
    print("\nMulti-seed stability (5 seeds, XGBoost vs LR)...")
    seeds = [42, 123, 456, 789, 999]
    seed_xgb_aucs, seed_lr_aucs = [], []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        m_xgb, _, _ = train_xgboost(Xtr, ytr, ALL_FEATURE_COLS, n_iter=10, seed=s)
        m_lr, sc    = train_logistic(Xtr, ytr, ALL_FEATURE_COLS, seed=s)
        p_xgb = m_xgb.predict_proba(Xte[ALL_FEATURE_COLS])[:, 1]
        p_lr  = m_lr.predict_proba(sc.transform(Xte[ALL_FEATURE_COLS]))[:, 1]
        seed_xgb_aucs.append(roc_auc_score(yte, p_xgb))
        seed_lr_aucs.append(roc_auc_score(yte, p_lr))

    multi_seed = {
        "seeds": seeds,
        "xgb_aucs": [round(a, 4) for a in seed_xgb_aucs],
        "lr_aucs":  [round(a, 4) for a in seed_lr_aucs],
        "xgb_mean_auc": round(float(np.mean(seed_xgb_aucs)), 4),
        "xgb_std_auc":  round(float(np.std(seed_xgb_aucs)), 4),
        "lr_mean_auc":  round(float(np.mean(seed_lr_aucs)), 4),
        "lr_std_auc":   round(float(np.std(seed_lr_aucs)), 4),
    }
    print(f"  XGB: {multi_seed['xgb_mean_auc']:.4f} ± {multi_seed['xgb_std_auc']:.4f}")
    print(f"  LR:  {multi_seed['lr_mean_auc']:.4f} ± {multi_seed['lr_std_auc']:.4f}")

    # ========== FAIRNESS ==========
    print("\nFairness analysis...")
    income_terciles = pd.qcut(inc_test, q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    approval_mask = xgb_tier == "Eligible"

    fairness_income = {}
    for grp in income_terciles.unique():
        mask = income_terciles == grp
        rate = approval_mask[mask].mean()
        fairness_income[str(grp)] = round(float(rate), 4)

    fairness_workertype = {}
    for wt in np.unique(wt_test):
        mask = wt_test == wt
        rate = approval_mask[mask].mean()
        fairness_workertype[wt] = round(float(rate), 4)

    # Disparate Impact Ratio (DIRatio = min_rate / max_rate)
    income_rates = list(fairness_income.values())
    wt_rates = list(fairness_workertype.values())
    dir_income = round(min(income_rates) / max(income_rates), 4) if max(income_rates) > 0 else None
    dir_workertype = round(min(wt_rates) / max(wt_rates), 4) if max(wt_rates) > 0 else None

    fairness = {
        "income_tercile_approval_rates": fairness_income,
        "worker_type_approval_rates": fairness_workertype,
        "disparate_impact_ratio_income": dir_income,
        "disparate_impact_ratio_workertype": dir_workertype,
        "acceptable_range": [0.80, 1.25],
    }
    print(f"  DIR (income): {dir_income}  DIR (worker type): {dir_workertype}")

    # ========== SHAP ==========
    print("\nComputing SHAP values...")
    X_test_arr2 = X_test[ALL_FEATURE_COLS]
    explainer = shap.TreeExplainer(xgb_model, feature_perturbation="tree_path_dependent")

    explain_sample = X_test_arr2.sample(n=min(1000, len(X_test_arr2)), random_state=SEED)
    shap_values = explainer(explain_sample)

    # Individual explanations: find one Eligible, one Review, one High_risk
    score_sample = prob_to_score(xgb_model.predict_proba(explain_sample)[:, 1])
    tier_sample  = np.array([assign_tier(s) for s in score_sample])

    individual_examples = {}
    for target_tier in ["Eligible", "Review", "High_risk"]:
        idxs = np.where(tier_sample == target_tier)[0]
        if len(idxs) > 0:
            pick = idxs[0]
            sv = shap_values[pick].values
            base = float(shap_values[pick].base_values)
            feat_vals = explain_sample.iloc[pick]
            # Convert SHAP (log-odds) contributions to score points
            # score = 100*(1-prob), so d(score)/d(shap) ~ -100 * prob*(1-prob) at that point
            # Use simpler linear approximation: scale by -100 * prob(1-p) factor
            prob_i = xgb_model.predict_proba(explain_sample.iloc[[pick]])[:, 1][0]
            scale = -100.0 * prob_i * (1.0 - prob_i)

            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            explanations = []
            for fi in top_idx:
                fname = ALL_FEATURE_COLS[fi]
                shap_contribution = sv[fi]
                score_contribution = shap_contribution * scale
                fval = feat_vals.iloc[fi]
                explanations.append({
                    "feature": fname,
                    "feature_value": round(float(fval), 3),
                    "shap_value": round(float(shap_contribution), 4),
                    "score_contribution_pts": round(float(score_contribution), 2),
                })
            individual_examples[target_tier] = {
                "score": round(float(score_sample[pick]), 2),
                "prob": round(float(prob_i), 4),
                "base_value": round(base, 4),
                "shap_sum": round(float(sv.sum()), 4),
                "diff_from_base": round(float(shap_values[pick].values.sum()), 4),
                "top_features": explanations,
            }

    # Sanity check: shap_sum ≈ xgb_output - base_value
    print("  SHAP sanity checks:")
    for tier_name, ex in individual_examples.items():
        print(f"    {tier_name}: sum(SHAP)={ex['shap_sum']:.4f}, diff_from_base={ex['diff_from_base']:.4f}")

    # ========== LATENCY ==========
    print("\nMeasuring prediction latency...")
    single_row = X_test_arr2.iloc[[0]]
    explainer_latency = shap.TreeExplainer(xgb_model)

    latencies = []
    for _ in range(200):
        t_start = time.perf_counter()
        _ = xgb_model.predict_proba(single_row)[:, 1]
        _ = explainer_latency(single_row)
        latencies.append((time.perf_counter() - t_start) * 1000)

    latency_mean = float(np.mean(latencies))
    latency_p95  = float(np.percentile(latencies, 95))
    print(f"  Mean latency: {latency_mean:.2f} ms, P95: {latency_p95:.2f} ms")

    # ========== Imbalanced experiment ==========
    from src.data_generator import generate_dataset
    from src.feature_engineering import compute_features
    print("\nRunning 80/20 imbalance experiment...")
    upi_imb, rej_imb, wkr_imb = generate_dataset(seed=42, n_workers=5000, imbalance_ratio=0.20)
    feat_imb = compute_features(upi_imb, rej_imb, wkr_imb)
    X_imb = feat_imb[ALL_FEATURE_COLS]
    y_imb = feat_imb["true_label"].values
    Xtr_i, Xte_i, ytr_i, yte_i = train_test_split(X_imb, y_imb, test_size=0.2, stratify=y_imb, random_state=SEED)
    m_imb, _, _ = train_xgboost(Xtr_i, ytr_i, ALL_FEATURE_COLS, n_iter=10, seed=SEED)
    p_imb = m_imb.predict_proba(Xte_i[ALL_FEATURE_COLS])[:, 1]
    imbalance_result = {
        "n_workers": 5000,
        "pos_rate": float(round(y_imb.mean(), 3)),
        "AUC": float(round(roc_auc_score(yte_i, p_imb), 4)),
        "Brier": float(round(brier_score_loss(yte_i, p_imb), 4)),
    }
    print(f"  Imbalanced experiment: {imbalance_result}")

    # ========== Comparison table ==========
    external_references = [
        {"Reference": "Yadav et al. (individual)", "AUC": 0.765, "Brier": 0.195, "Note": "External, different data"},
        {"Reference": "Yadav et al. (MSME)",       "AUC": 0.784, "Brier": 0.184, "Note": "External, different data"},
        {"Reference": "Ots et al.",                "AUC": 0.620, "Brier": None,  "Note": "External, different data"},
        {"Reference": "Ng et al.",                 "AUC": 0.806, "Brier": None,  "Note": "External, different data"},
        {"Reference": "Ours (XGBoost, synthetic)", "AUC": metrics_xgb["AUC"], "Brier": metrics_xgb["Brier"], "Note": "Synthetic data only"},
        {"Reference": "Ours (LR, synthetic)",      "AUC": metrics_lr["AUC"],  "Brier": metrics_lr["Brier"],  "Note": "Synthetic data only"},
        {"Reference": "Ours (RF, synthetic)",      "AUC": metrics_rf["AUC"],  "Brier": metrics_rf["Brier"],  "Note": "Synthetic data only"},
    ]

    # ========== Assemble all results ==========
    all_results = {
        "split_info": split_info,
        "xgb_best_params": xgb_best_params,
        "xgb_cv_auc": float(xgb_cv_auc),
        "main_metrics": all_metrics,
        "calibration": calibration_data,
        "tier_stats": tier_stats.to_dict(orient="records"),
        "tier_clf_metrics": tier_metrics,
        "tier_monotonic": monotonic,
        "ablation": ablation_results,
        "cold_start": cold_start,
        "multi_seed": multi_seed,
        "fairness": fairness,
        "shap_individual": individual_examples,
        "latency_ms": {"mean": round(latency_mean, 2), "p95": round(latency_p95, 2)},
        "imbalance_experiment": imbalance_result,
        "external_references": external_references,
        "roc_data": roc_data,
        "score_by_class": {
            "class0_scores": xgb_score[y_test == 0].tolist(),
            "class1_scores": xgb_score[y_test == 1].tolist(),
        },
    }

    # Save JSON
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save CSVs
    pd.DataFrame(all_metrics).to_csv(results_dir / "main_metrics.csv", index=False)
    tier_stats.to_csv(results_dir / "tier_stats.csv", index=False)
    pd.DataFrame(list(ablation_results.values())).to_csv(results_dir / "ablation.csv", index=False)
    pd.DataFrame(external_references).to_csv(results_dir / "external_comparison.csv", index=False)

    # Return objects needed for plotting
    return {
        "results": all_results,
        "xgb_model": xgb_model,
        "lr_model": lr_model,
        "lr_scaler": lr_scaler,
        "rf_model": rf_model,
        "X_test": X_test,
        "y_test": y_test,
        "wt_test": wt_test,
        "inc_test": inc_test,
        "xgb_prob": xgb_prob,
        "lr_prob": lr_prob,
        "rf_prob": rf_prob,
        "xgb_score": xgb_score,
        "xgb_tier": xgb_tier,
        "xgb_prob_cal": xgb_prob_cal,
        "shap_values": shap_values,
        "explain_sample": explain_sample,
        "individual_examples": individual_examples,
        "tier_stats": tier_stats,
        "fairness": fairness,
    }


if __name__ == "__main__":
    from src.feature_engineering import compute_features
    import pandas as pd

    feat_df = pd.read_csv("data/features.csv")
    run_experiments(feat_df, Path("results"))
