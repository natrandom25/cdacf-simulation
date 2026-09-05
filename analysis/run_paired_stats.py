"""R4: seed-paired comparisons of CDACF against each static baseline.

Replaces the R3 Mann-Whitney design (which treated paired seeds as independent
groups) with exact paired permutation tests and Wilcoxon signed-rank, plus
bootstrap CIs on the mean paired difference.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import itertools
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv(os.path.join(RESULTS, 'results_30seed.csv'))
BASE = ['PoW', 'PoS', 'PBFT', 'PoA', 'DPoS']
METRICS = ['latency', 'throughput', 'posture']
RNG = np.random.default_rng(0)


def exact_perm_p(d):
    """Exact two-sided sign-flip permutation test on paired differences."""
    n = len(d)
    obs = abs(d.mean())
    if n > 22:                                   # 2^n too large: sample
        signs = RNG.choice([-1.0, 1.0], size=(200_000, n))
        null = np.abs((signs * d).mean(axis=1))
        return float((null >= obs - 1e-15).mean()), 'monte-carlo (200k)'
    cnt = 0
    for combo in itertools.product([-1, 1], repeat=n):
        if abs(np.mean(np.array(combo) * d)) >= obs - 1e-15:
            cnt += 1
    return cnt / 2 ** n, 'exact'


def boot_ci(d, n=20000):
    m = RNG.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps))
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (len(ps) - rank) * ps[i])
        prev = max(prev, val)
        out[i] = prev
    return out


rows = []
for domain in ['TRAFFIC', 'ENVIRONMENTAL']:
    for metric in METRICS:
        fam = []
        for b in BASE:
            piv = (df[df.domain == domain]
                   .pivot(index='seed', columns='system', values=metric)
                   .sort_index())
            d = (piv['CDACF'] - piv[b]).to_numpy()
            p_perm, kind = exact_perm_p(d)
            try:
                p_wil = float(stats.wilcoxon(d, alternative='two-sided',
                                             zero_method='wilcox').pvalue)
            except ValueError:
                p_wil = np.nan
            lo, hi = boot_ci(d)
            fam.append(dict(domain=domain, metric=metric, baseline=b,
                            n=len(d),
                            mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                            ci_lo=lo, ci_hi=hi,
                            cohen_dz=d.mean() / d.std(ddof=1),
                            pct_seeds_cdacf_higher=float((d > 0).mean()),
                            p_perm=p_perm, perm_kind=kind, p_wilcoxon=p_wil))
        ps = holm(np.array([f['p_perm'] for f in fam]))
        for f, p in zip(fam, ps):
            f['p_perm_holm'] = p
        rows += fam

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, 'paired_stats_30seed.csv'), index=False)

pd.set_option('display.width', 200)
for domain in ['TRAFFIC', 'ENVIRONMENTAL']:
    print('\n' + '=' * 110)
    print(domain)
    print('=' * 110)
    sub = out[out.domain == domain].copy()
    sub['95% CI of diff'] = sub.apply(
        lambda r: f"({r.ci_lo:+.4f}, {r.ci_hi:+.4f})", axis=1)
    print(sub[['metric', 'baseline', 'mean_diff', '95% CI of diff', 'cohen_dz',
               'pct_seeds_cdacf_higher', 'p_perm', 'p_perm_holm', 'p_wilcoxon']]
          .to_string(index=False,
                     float_format=lambda v: f"{v:.4f}"))
