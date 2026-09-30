"""
Feature engineering from raw UPI and recharge event logs.
Produces features aligned with Table III of the paper.
All data is SYNTHETIC.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_features(upi_df: pd.DataFrame, recharge_df: pd.DataFrame,
                     worker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-worker features from raw event logs.
    Returns a DataFrame with one row per worker.
    """
    START_DATE = pd.Timestamp("2024-01-01")
    END_DATE = START_DATE + pd.Timedelta(days=90)

    # Parse timestamps
    if len(upi_df) > 0:
        upi_df = upi_df.copy()
        upi_df["timestamp"] = pd.to_datetime(upi_df["timestamp"], format="mixed")
        upi_df["week"] = ((upi_df["timestamp"] - START_DATE).dt.days // 7).clip(0, 12)
        upi_df["day"] = (upi_df["timestamp"] - START_DATE).dt.days.clip(0, 89)
        upi_df["month"] = (upi_df["day"] // 30).clip(0, 2)

    if len(recharge_df) > 0:
        recharge_df = recharge_df.copy()
        recharge_df["timestamp"] = pd.to_datetime(recharge_df["timestamp"], format="mixed")
        recharge_df["day"] = (recharge_df["timestamp"] - START_DATE).dt.days.clip(0, 89)
        recharge_df["month"] = (recharge_df["day"] // 30).clip(0, 2)

    worker_ids = worker_df["worker_id"].values
    features_list = []

    # Split UPI by direction
    upi_credit = upi_df[upi_df["direction"] == "credit"] if len(upi_df) > 0 else pd.DataFrame()
    upi_debit = upi_df[upi_df["direction"] == "debit"] if len(upi_df) > 0 else pd.DataFrame()

    # ---- UPI aggregations ----
    # Total txn count per worker
    upi_count = upi_df.groupby("worker_id").size().rename("upi_total_count")

    # Txns per week (mean over 13 weeks)
    upi_weekly = upi_df.groupby(["worker_id", "week"]).size().unstack(fill_value=0)
    upi_txns_per_week = upi_weekly.mean(axis=1).rename("upi_txns_per_week")
    upi_weekly_cv = (upi_weekly.std(axis=1) / (upi_weekly.mean(axis=1) + 1e-6)).rename("upi_weekly_count_cv")

    # Weekly inflow CV
    upi_credit_weekly = (upi_credit.groupby(["worker_id", "week"])["amount"]
                         .sum().unstack(fill_value=0) if len(upi_credit) > 0
                         else pd.DataFrame())
    upi_inflow_cv = (
        (upi_credit_weekly.std(axis=1) / (upi_credit_weekly.mean(axis=1) + 1e-6))
        .rename("upi_inflow_cv")
        if len(upi_credit_weekly) > 0 else pd.Series(dtype=float, name="upi_inflow_cv")
    )

    # Distinct counterparties per month
    upi_cp_monthly = (upi_df.groupby(["worker_id", "month"])["counterparty_id"]
                      .nunique().unstack(fill_value=0))
    upi_distinct_cp = upi_cp_monthly.mean(axis=1).rename("upi_distinct_counterparties_per_month")

    # Monthly credit/debit amounts
    monthly_credit = (upi_credit.groupby(["worker_id", "month"])["amount"]
                      .sum().unstack(fill_value=0) if len(upi_credit) > 0
                      else pd.DataFrame())
    monthly_debit = (upi_debit.groupby(["worker_id", "month"])["amount"]
                     .sum().unstack(fill_value=0) if len(upi_debit) > 0
                     else pd.DataFrame())

    # Estimated monthly income (mean monthly UPI credits)
    est_monthly_income = (monthly_credit.mean(axis=1).rename("est_monthly_income")
                          if len(monthly_credit) > 0
                          else pd.Series(dtype=float, name="est_monthly_income"))
    # Estimated monthly expenses (UPI debits)
    est_monthly_expenses = (monthly_debit.mean(axis=1).rename("est_monthly_expenses")
                            if len(monthly_debit) > 0
                            else pd.Series(dtype=float, name="est_monthly_expenses"))

    # Inflow-to-outflow ratio
    total_credit = (upi_credit.groupby("worker_id")["amount"].sum()
                    if len(upi_credit) > 0 else pd.Series(dtype=float))
    total_debit = (upi_debit.groupby("worker_id")["amount"].sum()
                   if len(upi_debit) > 0 else pd.Series(dtype=float))
    inflow_outflow_ratio = (total_credit / (total_debit + 1)).rename("inflow_outflow_ratio")

    # 3-month moving-average net inflow (we have 3 months; use monthly net)
    if len(monthly_credit) > 0 and len(monthly_debit) > 0:
        monthly_net = monthly_credit.reindex(monthly_debit.index, fill_value=0).subtract(
            monthly_debit.reindex(monthly_credit.index, fill_value=0), fill_value=0)
        net_inflow_ma3 = monthly_net.mean(axis=1).rename("net_inflow_ma3")
        # Trend: slope of net inflow across months (simple linear)
        def inflow_trend(row):
            vals = row.values
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals))
            if np.std(vals) < 1e-6:
                return 0.0
            return np.polyfit(x, vals, 1)[0]
        net_inflow_trend = monthly_net.apply(inflow_trend, axis=1).rename("net_inflow_trend")
    else:
        net_inflow_ma3 = pd.Series(dtype=float, name="net_inflow_ma3")
        net_inflow_trend = pd.Series(dtype=float, name="net_inflow_trend")

    # Activity span in days
    upi_first = upi_df.groupby("worker_id")["day"].min()
    upi_last = upi_df.groupby("worker_id")["day"].max()
    activity_span = (upi_last - upi_first).rename("upi_activity_span_days")

    # Candidate EMI signature: use debit sub-DF grouped by worker_id
    def _detect_emi_for_group(sub_df):
        if len(sub_df) < 2:
            return 0
        sub_df = sub_df.sort_values("day")
        rounded = (sub_df["amount"] / 50).round() * 50
        emi_count = 0
        for amt in rounded.unique():
            rows = sub_df[rounded == amt].sort_values("day")
            if len(rows) >= 2:
                intervals = np.diff(rows["day"].values)
                if any((22 <= iv <= 35) for iv in intervals):
                    emi_count += 1
        return min(emi_count, 3)

    if len(upi_debit) > 0:
        emi_list = []
        for wid, grp in upi_debit.groupby("worker_id"):
            emi_list.append({"worker_id": wid, "emi_signature_count": _detect_emi_for_group(grp)})
        upi_emi = pd.DataFrame(emi_list).set_index("worker_id")["emi_signature_count"]
    else:
        upi_emi = pd.Series(dtype=float, name="emi_signature_count")

    # ---- Recharge aggregations ----
    if len(recharge_df) > 0:
        recharge_count = recharge_df.groupby("worker_id").size().rename("recharge_count")
        recharge_per_month = (recharge_df.groupby(["worker_id", "month"]).size()
                              .unstack(fill_value=0).mean(axis=1)
                              .rename("recharge_per_month"))
        recharge_amt_mean = (recharge_df.groupby("worker_id")["amount"]
                             .mean().rename("recharge_amount_mean"))
        recharge_amt_std = (recharge_df.groupby("worker_id")["amount"]
                            .std().fillna(0).rename("recharge_amount_std"))

        # Recharge interval std (days between recharges)
        if len(recharge_df) > 0:
            rint_list = []
            for wid, grp in recharge_df.groupby("worker_id"):
                days = grp["day"].sort_values().values
                std_val = float(np.std(np.diff(days))) if len(days) >= 2 else np.nan
                rint_list.append({"worker_id": wid, "recharge_interval_std": std_val})
            recharge_interval_sd = (pd.DataFrame(rint_list)
                                    .set_index("worker_id")["recharge_interval_std"])

        # Recharge amount trend (slope across months)
        recharge_monthly_amt = (recharge_df.groupby(["worker_id", "month"])["amount"]
                                .mean().unstack(fill_value=np.nan))

        def amt_trend(row):
            vals = row.dropna().values
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals))
            return float(np.polyfit(x, vals, 1)[0])

        recharge_amt_trend = recharge_monthly_amt.apply(amt_trend, axis=1).rename("recharge_amount_trend")
    else:
        recharge_count = pd.Series(dtype=float, name="recharge_count")
        recharge_per_month = pd.Series(dtype=float, name="recharge_per_month")
        recharge_amt_mean = pd.Series(dtype=float, name="recharge_amount_mean")
        recharge_amt_std = pd.Series(dtype=float, name="recharge_amount_std")
        recharge_interval_sd = pd.Series(dtype=float, name="recharge_interval_std")
        recharge_amt_trend = pd.Series(dtype=float, name="recharge_amount_trend")

    # ---- Assemble feature DataFrame ----
    base_df = pd.DataFrame({"worker_id": worker_ids}).set_index("worker_id")

    feature_series = [
        # Frequency
        upi_txns_per_week,
        upi_weekly_cv,
        upi_distinct_cp,
        recharge_per_month,
        # Regularity
        upi_inflow_cv,
        recharge_interval_sd,
        # Volume/Trend
        net_inflow_ma3,
        net_inflow_trend,
        inflow_outflow_ratio,
        recharge_amt_trend,
        # Affordability
        est_monthly_income,
        est_monthly_expenses,
        upi_emi,
        activity_span,
        # Recharge extra
        recharge_count,
        recharge_amt_mean,
        recharge_amt_std,
        # Raw count
        upi_count,
    ]

    feat_df = base_df.copy()
    for s in feature_series:
        feat_df = feat_df.join(s, how="left")

    feat_df = feat_df.reset_index()

    # ---- Missingness indicators for sparse workers ----
    sparse_cols = ["upi_txns_per_week", "est_monthly_income", "net_inflow_ma3",
                   "recharge_per_month", "recharge_interval_std"]
    for col in sparse_cols:
        feat_df[f"{col}_missing"] = feat_df[col].isna().astype(int)

    # ---- Fill NaN with 0 (after creating missingness indicators) ----
    feat_df = feat_df.fillna(0)

    # ---- Estimated disposable surplus ----
    feat_df["est_disposable_surplus"] = (feat_df["est_monthly_income"]
                                          - feat_df["est_monthly_expenses"]
                                          - feat_df["emi_signature_count"] * 500)

    # ---- Merge label and metadata ----
    feat_df = feat_df.merge(
        worker_df[["worker_id", "worker_type", "true_label", "latent_capacity"]],
        on="worker_id", how="left"
    )

    return feat_df


UPI_FEATURE_COLS = [
    "upi_txns_per_week", "upi_weekly_count_cv", "upi_distinct_counterparties_per_month",
    "upi_inflow_cv", "net_inflow_ma3", "net_inflow_trend", "inflow_outflow_ratio",
    "est_monthly_income", "est_monthly_expenses", "emi_signature_count",
    "upi_activity_span_days", "upi_total_count", "est_disposable_surplus",
    "upi_txns_per_week_missing", "est_monthly_income_missing", "net_inflow_ma3_missing",
]

RECHARGE_FEATURE_COLS = [
    "recharge_per_month", "recharge_interval_std", "recharge_amount_trend",
    "recharge_count", "recharge_amount_mean", "recharge_amount_std",
    "recharge_per_month_missing", "recharge_interval_std_missing",
]

ALL_FEATURE_COLS = UPI_FEATURE_COLS + RECHARGE_FEATURE_COLS


if __name__ == "__main__":
    data_dir = Path("data")
    upi_df = pd.read_csv(data_dir / "upi_events.csv")
    recharge_df = pd.read_csv(data_dir / "recharge_events.csv")
    worker_df = pd.read_csv(data_dir / "worker_meta.csv")

    print("Computing features...")
    feat_df = compute_features(upi_df, recharge_df, worker_df)
    feat_df.to_csv(data_dir / "features.csv", index=False)
    print(f"Features shape: {feat_df.shape}")
    print(feat_df.dtypes)
    print("Done.")
