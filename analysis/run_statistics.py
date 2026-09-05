import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
# -*- coding: utf-8 -*-
"""
CDACF — Statistical analysis script
====================================
Reproduces the Mann–Whitney U comparisons reported in the manuscript:

  * Per-seed means (seeds 5–9, n = 5 per group) as independent observations
  * CDACF vs each static baseline (PoW, PoS, PBFT, PoA, DPoS)
  * Two-sided, EXACT p-values (scipy method='exact'; valid: no ties expected)
  * Holm–Bonferroni correction within each (domain, metric) family
  * Effect size: rank-biserial correlation  r_rb = 1 - 2U/(n1*n2)

Outputs:
  results_per_seed.csv      — raw per-seed means for CDACF and all baselines
  statistics_summary.csv    — U, p_exact, p_holm, rank-biserial r per comparison

Usage:
  python run_statistics.py
Requires iot_integration_bayesian_v4.py in the same directory.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from iot_integration_bayesian_v4 import (
    IoTDomain,
    run_simulation,
    run_all_static_baselines,
    VALIDATION_SEEDS,   # [5, 6, 7, 8, 9]
)

TIME_STEPS = 1000
METRICS = {
    "latency":    ("latency",    "latency"),      # (CDACF history key, baseline key)
    "throughput": ("throughput", "throughput"),
    "mitigation_response": ("posture_index", "posture_index"),
}
BASELINES = ["PoW", "PoS", "PBFT", "PoA", "DPoS"]


def collect_per_seed(domain: IoTDomain) -> pd.DataFrame:
    """Run CDACF and all five static baselines for each validation seed."""
    rows = []
    for seed in VALIDATION_SEEDS:
        r = run_simulation(domain, TIME_STEPS, seed, verbose=False)
        row = {"domain": domain.name, "seed": seed, "system": "CDACF"}
        for metric, (ck, _) in METRICS.items():
            row[metric] = float(np.mean(r[ck]))
        rows.append(row)

        base = run_all_static_baselines(domain, TIME_STEPS, seed)
        for name in BASELINES:
            b = base[name]
            row = {"domain": domain.name, "seed": seed, "system": name}
            for metric, (_, bk) in METRICS.items():
                row[metric] = float(b[bk])
            rows.append(row)
    return pd.DataFrame(rows)


def holm_correction(pvals):
    """Holm–Bonferroni step-down correction. Returns adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return adjusted


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for domain in df["domain"].unique():
        d = df[df["domain"] == domain]
        cdacf = d[d["system"] == "CDACF"]
        for metric in METRICS:
            family = []
            for name in BASELINES:
                b = d[d["system"] == name]
                x = cdacf[metric].values   # n = 5
                y = b[metric].values       # n = 5
                res = mannwhitneyu(x, y, alternative="two-sided", method="exact")
                u = res.statistic
                r_rb = 1.0 - 2.0 * u / (len(x) * len(y))  # rank-biserial, sign: + = CDACF higher ranks
                family.append({
                    "domain": domain, "metric": metric, "baseline": name,
                    "cdacf_mean": float(np.mean(x)), "cdacf_sd": float(np.std(x, ddof=1)),
                    "baseline_mean": float(np.mean(y)), "baseline_sd": float(np.std(y, ddof=1)),
                    "U": float(u), "p_exact_two_sided": float(res.pvalue),
                    "rank_biserial_r": float(r_rb),
                })
            padj = holm_correction([f["p_exact_two_sided"] for f in family])
            for f, p in zip(family, padj):
                f["p_holm"] = float(p)
            out.extend(family)
    return pd.DataFrame(out)


def main():
    frames = [collect_per_seed(IoTDomain.TRAFFIC),
              collect_per_seed(IoTDomain.ENVIRONMENT)]
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed.to_csv("results_per_seed.csv", index=False)
    print(f"Wrote results_per_seed.csv  ({len(per_seed)} rows)")

    summary = analyse(per_seed)
    summary.to_csv("statistics_summary.csv", index=False)
    print(f"Wrote statistics_summary.csv ({len(summary)} comparisons)")
    print()
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary.round(4).to_string(index=False))
    print("\nNote: with n = 5 per group and complete separation, exact two-sided "
          "p = 0.0079 and |rank-biserial r| = 1.0 (its ceiling). Interpret r "
          "descriptively; it is not evidence of generalisable superiority.")


if __name__ == "__main__":
    main()
