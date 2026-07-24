# -*- coding: utf-8 -*-
"""
CDACF — Sensitivity analysis (Table 9) and stability-gate ablation (Table 10)
==============================================================================
Regenerates both tables under the CORRECTED simulator so that:
  * the baseline row equals the main-configuration multi-seed values
    (fixing the Table 9 inconsistency Reviewer 2 identified), and
  * Table 10 reports actual mean ± SD per cell, not relative changes.

Each configuration is run over the validation seeds (5–9), 1,000 steps,
both domains. Parameters are varied one at a time from the defaults:
  improvement threshold 0.08, dwell {Traffic 10, Env 15}, Bayesian window 20.

Outputs: sensitivity_summary.csv, ablation_summary.csv (+ printed tables).
Usage:   python run_sensitivity.py   (needs iot_integration_bayesian_v2.py alongside)
"""

import contextlib
import numpy as np
import pandas as pd
from tabulate import tabulate

from iot_integration_bayesian_v2 import (
    IoTDomain, run_simulation, VALIDATION_SEEDS,
    StabilityAwareConsensusLayer, BayesianThreatDetector,
)

TIME_STEPS = 1000


@contextlib.contextmanager
def patched(dwell_scale=None, improvement=None, window=None):
    """Temporarily override stability-gate / detector parameters."""
    dwell0  = dict(StabilityAwareConsensusLayer.MIN_DWELL)
    imp0    = StabilityAwareConsensusLayer.IMPROVEMENT_THRESHOLD
    window0 = BayesianThreatDetector.WINDOW
    try:
        if dwell_scale is not None:
            StabilityAwareConsensusLayer.MIN_DWELL = {
                k: max(1, int(round(v * dwell_scale))) for k, v in dwell0.items()}
        if improvement is not None:
            StabilityAwareConsensusLayer.IMPROVEMENT_THRESHOLD = improvement
        if window is not None:
            BayesianThreatDetector.WINDOW = window
        yield
    finally:
        StabilityAwareConsensusLayer.MIN_DWELL = dwell0
        StabilityAwareConsensusLayer.IMPROVEMENT_THRESHOLD = imp0
        BayesianThreatDetector.WINDOW = window0


def run_config(domain: IoTDomain) -> dict:
    """Multi-seed means ± SD for one configuration."""
    rows = []
    for seed in VALIDATION_SEEDS:
        r = run_simulation(domain, TIME_STEPS, seed, verbose=False)
        rows.append({
            'latency':    float(np.mean(r['latency'])),
            'throughput': float(np.mean(r['throughput'])),
            'mitigation': float(np.mean(r['threat_prevention'])),
            'switches':   float(r['n_switches']),
        })
    df = pd.DataFrame(rows)
    out = {}
    for k in df.columns:
        out[f'{k}_mean'] = float(df[k].mean())
        out[f'{k}_sd']   = float(df[k].std(ddof=1))
    return out


# One-at-a-time sensitivity configurations (Table 9)
SENSITIVITY_CONFIGS = [
    ('Baseline (main configuration)',        {}),
    ('Improvement threshold 0.04 (-50%)',    {'improvement': 0.04}),
    ('Improvement threshold 0.12 (+50%)',    {'improvement': 0.12}),
    ('Dwell x0.5 (Traffic 5 / Env 8)',       {'dwell_scale': 0.5}),
    ('Dwell x2 (Traffic 20 / Env 30)',       {'dwell_scale': 2.0}),
    ('Bayesian window 10 (-50%)',            {'window': 10}),
    ('Bayesian window 40 (+100%)',           {'window': 40}),
]

# Stability-gate ablation configurations (Table 10)
ABLATION_CONFIGS = [
    ('Full stability gate (dwell + threshold)',   {}),
    ('No improvement threshold (dwell only)',     {'improvement': 0.0}),
    ('No dwell (threshold only)',                 {'dwell_scale': 0.0}),
    ('Gate disabled (dwell=1, threshold=0)',      {'dwell_scale': 0.0, 'improvement': 0.0}),
]


def sweep(configs, label):
    records = []
    for name, overrides in configs:
        for domain, dlabel in [(IoTDomain.TRAFFIC, 'Traffic'),
                               (IoTDomain.ENVIRONMENT, 'Environmental')]:
            print(f'  {label}: {name} — {dlabel} (seeds {VALIDATION_SEEDS})')
            with patched(**overrides):
                res = run_config(domain)
            records.append({'configuration': name, 'domain': dlabel, **res})
    return pd.DataFrame(records)


def pretty(df):
    view = pd.DataFrame({
        'Configuration': df['configuration'],
        'Domain':        df['domain'],
        'Latency (mean±SD)':    [f"{m:.4f} ± {s:.4f}" for m, s in zip(df.latency_mean, df.latency_sd)],
        'Throughput (mean±SD)': [f"{m:.4f} ± {s:.4f}" for m, s in zip(df.throughput_mean, df.throughput_sd)],
        'Mitigation (mean±SD)': [f"{m*100:.2f}% ± {s*100:.2f}%" for m, s in zip(df.mitigation_mean, df.mitigation_sd)],
        'Switches (mean±SD)':   [f"{m:.1f} ± {s:.1f}" for m, s in zip(df.switches_mean, df.switches_sd)],
    })
    return tabulate(view, headers='keys', tablefmt='fancy_grid', showindex=False)


def main():
    print('=' * 70)
    print('TABLE 9 — SENSITIVITY ANALYSIS (one-at-a-time, seeds 5-9, 1,000 steps)')
    print('=' * 70)
    sens = sweep(SENSITIVITY_CONFIGS, 'sensitivity')
    sens.to_csv('sensitivity_summary.csv', index=False)
    print(pretty(sens))

    print('\n' + '=' * 70)
    print('TABLE 10 — STABILITY-GATE ABLATION (seeds 5-9, 1,000 steps)')
    print('=' * 70)
    abl = sweep(ABLATION_CONFIGS, 'ablation')
    abl.to_csv('ablation_summary.csv', index=False)
    print(pretty(abl))

    print('\nWrote sensitivity_summary.csv and ablation_summary.csv')
    print('Note: baseline row of the sensitivity table MUST match the main '
          'multi-seed results (Traffic latency ~0.1819, Env ~0.8704). If it '
          'does not, the environment differs from the corrected simulator.')


if __name__ == '__main__':
    main()
