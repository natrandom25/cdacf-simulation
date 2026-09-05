"""R4: detection metrics, confusion matrices, and the Bayesian-vs-threshold test.

Answers Reviewer 2 point 9 (full detection metrics, calibration separation,
comparison against a simple non-Bayesian threshold).

Also demonstrates empirically the analytic result that the CDACF posterior is a
strictly increasing affine function of the sliding-window indicator mean:

    posterior = (alpha + W*xbar) / (alpha + beta + W)

so thresholding the posterior and thresholding xbar are the same decision rule.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import io, contextlib
import numpy as np, pandas as pd
from iot_integration_bayesian_v4 import IoTDomain, BayesianThreatDetector as BTD, \
    DomainAwareConsensusEngine as ENG

ATTACKS = list(BTD.INDICATOR_WEIGHTS)
TH = ENG.ACTIVATION_THRESHOLDS
STEPS = 1000
EVAL_SEEDS = list(range(5, 35))
CAL_SEEDS = list(range(100, 110))       # disjoint calibration set


def collect(domain, seed):
    """Re-run the telemetry stream and record indicator means and posteriors."""
    import random
    random.seed(seed); np.random.seed(seed)
    sim = (cdacf.TrafficNetworkSimulator() if domain == IoTDomain.TRAFFIC
           else cdacf.EnvironmentalNetworkSimulator())
    det = BTD(domain)
    sched = {int(STEPS * f): a for f, a in
             [(.10, 'sybil'), (.25, 'dos'), (.40, 'byzantine'),
              (.55, 'eclipse'), (.70, 'majority'), (.85, 'routing')]}
    rec = {int(STEPS * f): a for f, a in
           [(.20, 'sybil'), (.35, 'dos'), (.50, 'byzantine'),
            (.65, 'eclipse'), (.80, 'majority'), (.95, 'routing')]}
    rows, active = [], None
    for i in range(STEPS):
        if i in sched:
            active = sched[i]; sim.simulate_attack(active)
        if i in rec:
            sim.recover_from_attack(rec[i]); det.reset_priors(); active = None
        st = sim.generate_network_state()
        post = det.update(st.metrics)
        xbar = {a: float(np.mean(det._window[a])) for a in ATTACKS}
        rows.append(dict(step=i, truth=active or 'none',
                         **{f'post_{a}': post[a] for a in ATTACKS},
                         **{f'xbar_{a}': xbar[a] for a in ATTACKS}))
    return pd.DataFrame(rows)


def metrics(df, attack, thresh, col):
    y = (df.truth == attack).to_numpy()
    p = (df[f'{col}_{attack}'] > thresh).to_numpy()
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec_ = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec_ / (prec + rec_) if prec and rec_ and prec + rec_ else np.nan
    first = np.argmax(y & p) if (y & p).any() else np.nan
    onset = np.argmax(y) if y.any() else np.nan
    ttd = (first - onset) if not np.isnan(first) else np.nan
    return dict(attack=attack, TP=tp, FP=fp, FN=fn, TN=tn, precision=prec,
                recall=rec_, specificity=spec, f1=f1, time_to_detect=ttd)


out_rows, agree_rows = [], []
for domain, dl in [(IoTDomain.TRAFFIC, 'TRAFFIC'),
                   (IoTDomain.ENVIRONMENT, 'ENVIRONMENTAL')]:
    # ---- equivalent raw threshold implied by each posterior threshold ----
    pr = BTD.DOMAIN_PRIORS[domain]; a, b, W = pr['alpha'], pr['beta'], BTD.WINDOW
    equiv = {k: (v * (a + b + W) - a) / W for k, v in TH.items()}

    per_seed = [collect(domain, s) for s in EVAL_SEEDS]
    for atk in ATTACKS:
        mb = [metrics(d, atk, TH[atk], 'post') for d in per_seed]
        mx = [metrics(d, atk, equiv[atk], 'xbar') for d in per_seed]
        agree = np.mean([
            ((d[f'post_{atk}'] > TH[atk]) == (d[f'xbar_{atk}'] > equiv[atk])).mean()
            for d in per_seed])
        agree_rows.append(dict(domain=dl, attack=atk, posterior_threshold=TH[atk],
                               equivalent_raw_threshold=equiv[atk],
                               decision_agreement=agree))
        row = dict(domain=dl, detector='Bayesian posterior')
        for k in ['precision', 'recall', 'specificity', 'f1', 'time_to_detect']:
            row[k] = float(np.nanmean([m[k] for m in mb]))
        row['attack'] = atk
        row['FPR'] = float(np.nanmean([m['FP'] / (m['FP'] + m['TN']) for m in mb]))
        out_rows.append(row)
        row2 = dict(domain=dl, detector='Raw windowed mean')
        for k in ['precision', 'recall', 'specificity', 'f1', 'time_to_detect']:
            row2[k] = float(np.nanmean([m[k] for m in mx]))
        row2['attack'] = atk
        row2['FPR'] = float(np.nanmean([m['FP'] / (m['FP'] + m['TN']) for m in mx]))
        out_rows.append(row2)

det = pd.DataFrame(out_rows)[
    ['domain', 'attack', 'detector', 'precision', 'recall', 'specificity',
     'f1', 'FPR', 'time_to_detect']]
det.to_csv(os.path.join(RESULTS, 'detection_metrics.csv'), index=False)
ag = pd.DataFrame(agree_rows)
ag.to_csv(os.path.join(RESULTS, 'bayes_vs_threshold_agreement.csv'), index=False)

pd.set_option('display.width', 220)
print('=' * 108)
print('DETECTION METRICS  (mean over 30 seeds, 1,000 steps)')
print('=' * 108)
print(det.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print('=' * 108)
print('BAYESIAN POSTERIOR vs RAW WINDOWED MEAN — decision agreement')
print('=' * 108)
print(ag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
