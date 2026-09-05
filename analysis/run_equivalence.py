"""R4: TOST equivalence tests for the environmental convergence claim.

R3 argued convergence from a non-significant Mann-Whitney result (absence of
evidence). With 30 seed-paired runs every comparison is significant, so that
argument fails. The correct test is equivalence within a pre-specified margin.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv(os.path.join(RESULTS, 'results_30seed.csv'))

# Pre-specified equivalence margin: 2% of the static-PoS mean for that metric.
# Justification: the smallest difference that would change a deployment
# decision. Declared before testing, applied identically to every comparison.
MARGIN_FRAC = 0.02

print(f"TOST equivalence, margin = {MARGIN_FRAC:.0%} of the comparator mean\n")
rows = []
for domain in ['ENVIRONMENTAL', 'TRAFFIC']:
    for metric in ['latency', 'throughput', 'posture']:
        piv = (df[df.domain == domain]
               .pivot(index='seed', columns='system', values=metric).sort_index())
        for b in ['PoS', 'PBFT', 'DPoS', 'PoA', 'PoW']:
            d = (piv['CDACF'] - piv[b]).to_numpy()
            margin = MARGIN_FRAC * piv[b].mean()
            n, m, se = len(d), d.mean(), stats.sem(d)
            t_lo = (m + margin) / se
            t_hi = (m - margin) / se
            p_tost = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
            ci = stats.t.interval(0.90, n - 1, loc=m, scale=se)
            rows.append(dict(domain=domain, metric=metric, baseline=b,
                             mean_diff=m, margin=margin,
                             ci90_lo=ci[0], ci90_hi=ci[1],
                             p_tost=p_tost,
                             equivalent=bool(p_tost < 0.05),
                             rel_diff_pct=100 * m / piv[b].mean()))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, 'equivalence_30seed.csv'), index=False)
pd.set_option('display.width', 200)
for domain in ['ENVIRONMENTAL', 'TRAFFIC']:
    print('=' * 100)
    print(domain)
    print('=' * 100)
    print(out[out.domain == domain][
        ['metric', 'baseline', 'mean_diff', 'rel_diff_pct', 'margin',
         'ci90_lo', 'ci90_hi', 'p_tost', 'equivalent']]
        .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
