"""
Synthetic data generator for UPI transaction and mobile recharge event logs.
Produces calibrated data for 10,000 workers over a 90-day window.
All data is SYNTHETIC - for methodological feasibility testing only.
Uses vectorised numpy for speed.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_WORKERS = 10_000
WINDOW_DAYS = 90
WORKER_TYPES = ["street_vendor", "daily_wage_labourer", "gig_delivery", "domestic_worker"]

WORKER_PARAMS = {
    "street_vendor":       {"income_mean": 12000, "income_std": 4000,  "digital_adopt": 0.65, "upi_freq_day": 1.8, "recharge_freq_month": 1.2, "seasonality_amp": 0.25},
    "daily_wage_labourer": {"income_mean":  9000, "income_std": 3000,  "digital_adopt": 0.45, "upi_freq_day": 0.9, "recharge_freq_month": 1.1, "seasonality_amp": 0.30},
    "gig_delivery":        {"income_mean": 16000, "income_std": 5000,  "digital_adopt": 0.90, "upi_freq_day": 3.2, "recharge_freq_month": 2.0, "seasonality_amp": 0.15},
    "domestic_worker":     {"income_mean":  8000, "income_std": 2500,  "digital_adopt": 0.40, "upi_freq_day": 0.7, "recharge_freq_month": 0.9, "seasonality_amp": 0.20},
}

FESTIVAL_DAYS = {15, 16, 45, 46, 72, 73}
RECHARGE_PLANS = np.array([19, 29, 49, 79, 99, 149, 199, 299, 399, 599, 799], dtype=float)

WEEKDAY_WEIGHTS = np.array([1.1, 1.2, 1.15, 1.1, 1.25, 1.3, 0.5])


def assign_worker_types(rng, n):
    probs = [0.30, 0.30, 0.25, 0.15]
    return rng.choice(WORKER_TYPES, size=n, p=probs)


def generate_latent_capacity(rng, worker_types, labels):
    capacity = np.zeros(len(labels))
    for i, wtype in enumerate(worker_types):
        p = WORKER_PARAMS[wtype]
        base = rng.beta(2, 2)
        income_norm = np.clip((p["income_mean"] + rng.normal(0, p["income_std"])) / 20000, 0.05, 1.0)
        capacity[i] = 0.5 * base + 0.3 * income_norm + 0.2 * p["digital_adopt"]

    shock_mask = rng.random(len(labels)) < 0.15
    capacity[shock_mask] -= rng.uniform(0.1, 0.4, shock_mask.sum())
    capacity = np.clip(capacity, 0.01, 0.99)

    noise_mask = rng.random(len(labels)) < 0.06
    noisy_labels = labels.copy()
    noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]

    return capacity, noisy_labels


def _generate_upi_batch(rng, worker_ids, worker_types_arr, capacities, labels):
    """Generate UPI events for a batch of workers using vectorised approach."""
    records = []
    START_DATE = pd.Timestamp("2024-01-01")

    for i, wid in enumerate(worker_ids):
        wtype = worker_types_arr[i]
        cap = capacities[i]
        label = labels[i]
        p = WORKER_PARAMS[wtype]

        adopt_thresh = p["digital_adopt"] * (0.5 + 0.5 * cap)
        if rng.random() > adopt_thresh:
            n_events = int(rng.integers(0, 8))
        else:
            base_rate = p["upi_freq_day"] * (0.5 + cap)
            if label == 1:
                base_rate *= rng.uniform(0.6, 0.95)
            n_events = int(np.clip(rng.poisson(base_rate * WINDOW_DAYS), 0, 500))

        if n_events == 0:
            continue

        # Generate all event timestamps at once
        days = np.sort(rng.uniform(0, WINDOW_DAYS, n_events))
        day_ints = days.astype(int)

        # Amounts: log-normal
        mu_log = np.log(300 * (0.5 + cap))
        amounts = np.clip(rng.lognormal(mu_log, 0.8, n_events), 20, 5000)

        # Festival bump (vectorised)
        fest_mask = np.isin(day_ints, list(FESTIVAL_DAYS))
        amounts[fest_mask] *= rng.uniform(1.5, 2.5, fest_mask.sum())
        amounts = np.minimum(amounts, 5000)

        # Weekday multiplier
        dows = day_ints % 7
        wday_mult = WEEKDAY_WEIGHTS[dows] * rng.uniform(0.8, 1.2, n_events)
        amounts *= wday_mult
        amounts = np.round(np.clip(amounts, 20, 5000), 2)

        # Direction
        credit_prob = 0.45 + 0.2 * cap
        directions = np.where(rng.random(n_events) < credit_prob, "credit", "debit")

        # Counterparty
        pool_size = max(3, int(15 * p["digital_adopt"] * cap))
        cp_ids = rng.integers(0, pool_size, n_events)
        counterparties = [f"cp_{wid}_{c:03d}" for c in cp_ids]

        # Build timestamps (vectorised)
        START_NS = np.datetime64("2024-01-01", "ns")
        day_ns = (days * 86400 * 1e9).astype("int64")
        ts_arr = START_NS + day_ns.astype("timedelta64[ns]")
        timestamps = [str(t) for t in ts_arr.astype("datetime64[ms]")]

        for j in range(n_events):
            records.append((wid, timestamps[j], amounts[j], directions[j], counterparties[j]))

    return records


def _generate_recharge_batch(rng, worker_ids, worker_types_arr, capacities, labels):
    """Generate recharge events for a batch of workers."""
    records = []
    START_DATE = pd.Timestamp("2024-01-01")

    # Plan selection probabilities by capacity
    plan_weights_high = np.array([0.01, 0.02, 0.04, 0.08, 0.12, 0.15, 0.20, 0.18, 0.12, 0.06, 0.02])
    plan_weights_low  = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01])
    plan_weights_high /= plan_weights_high.sum()
    plan_weights_low  /= plan_weights_low.sum()

    for i, wid in enumerate(worker_ids):
        wtype = worker_types_arr[i]
        cap = capacities[i]
        label = labels[i]
        p = WORKER_PARAMS[wtype]

        freq = p["recharge_freq_month"] * (0.6 + 0.4 * cap)
        if label == 1:
            freq *= rng.uniform(0.7, 1.0)

        n_events = int(np.clip(rng.poisson(freq * 3), 0, 20))
        if n_events == 0:
            continue

        plan_weights = plan_weights_high if cap > 0.6 else plan_weights_low
        amounts = rng.choice(RECHARGE_PLANS, size=n_events, p=plan_weights)

        base_interval = WINDOW_DAYS / n_events
        days = np.zeros(n_events)
        days[0] = rng.uniform(0, base_interval)
        for j in range(1, n_events):
            days[j] = days[j-1] + base_interval + rng.normal(0, base_interval * 0.3)
        days = np.clip(days, 0, WINDOW_DAYS - 0.01)

        # Build timestamps (vectorised)
        START_NS = np.datetime64("2024-01-01", "ns")
        day_ns = (days * 86400 * 1e9).astype("int64")
        ts_arr = START_NS + day_ns.astype("timedelta64[ns]")
        timestamps = [str(t) for t in ts_arr.astype("datetime64[ms]")]

        for j in range(n_events):
            records.append((wid, timestamps[j], float(amounts[j])))

    return records


def generate_dataset(seed=RNG_SEED, n_workers=N_WORKERS, imbalance_ratio=None):
    rng = np.random.default_rng(seed)

    if imbalance_ratio is None:
        n_pos = n_workers // 2
    else:
        n_pos = int(n_workers * imbalance_ratio)
    n_neg = n_workers - n_pos

    labels = np.array([0] * n_neg + [1] * n_pos)
    worker_types = assign_worker_types(rng, n_workers)
    capacity, labels_noisy = generate_latent_capacity(rng, worker_types, labels)

    worker_ids = np.arange(n_workers)

    upi_records = _generate_upi_batch(rng, worker_ids, worker_types, capacity, labels_noisy)
    recharge_records = _generate_recharge_batch(rng, worker_ids, worker_types, capacity, labels_noisy)

    upi_df = pd.DataFrame(upi_records,
                          columns=["worker_id", "timestamp", "amount", "direction", "counterparty_id"])
    recharge_df = pd.DataFrame(recharge_records,
                               columns=["worker_id", "timestamp", "amount"])

    worker_df = pd.DataFrame({
        "worker_id": worker_ids,
        "worker_type": worker_types,
        "true_label": labels_noisy,
        "latent_capacity": capacity,
    })

    return upi_df, recharge_df, worker_df


if __name__ == "__main__":
    import time
    out = Path("data")
    out.mkdir(exist_ok=True)

    t0 = time.time()
    print("Generating balanced dataset (50/50)...")
    upi_df, recharge_df, worker_df = generate_dataset(seed=42)

    upi_df.to_csv(out / "upi_events.csv", index=False)
    recharge_df.to_csv(out / "recharge_events.csv", index=False)
    worker_df.to_csv(out / "worker_meta.csv", index=False)

    print(f"UPI events: {len(upi_df):,}")
    print(f"Recharge events: {len(recharge_df):,}")
    print(f"Workers: {len(worker_df):,}")
    print(f"Time: {time.time()-t0:.1f}s")
