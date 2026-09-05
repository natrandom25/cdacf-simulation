"""R4: 30-seed run of CDACF + 5 static baselines, both domains, paired by seed."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import io, sys, time, contextlib, json
import numpy as np, pandas as pd
from iot_integration_bayesian_v4 import IoTDomain, ConsensusType

SEEDS = list(range(5, 35))          # 30 seeds; 5-9 are the original R3 set
STEPS = 1000
MECHS = [ConsensusType.POW, ConsensusType.POS, ConsensusType.PBFT,
         ConsensusType.POA, ConsensusType.DPOS]
SHORT = {ConsensusType.POW: 'PoW', ConsensusType.POS: 'PoS',
         ConsensusType.PBFT: 'PBFT', ConsensusType.POA: 'PoA',
         ConsensusType.DPOS: 'DPoS'}

ATTR = cdacf.DomainAwareConsensusEngine.MECHANISM_ATTRIBUTES
BY_NAME = {c.value: c for c in ConsensusType}


def blended_energy(counts):
    tot = sum(counts.values())
    if not tot:
        return float('nan')
    return sum(ATTR[BY_NAME[k]]['energy_efficiency'] * v for k, v in counts.items()) / tot


rows = []
t0 = time.time()
for domain, dlabel in [(IoTDomain.TRAFFIC, 'TRAFFIC'),
                       (IoTDomain.ENVIRONMENT, 'ENVIRONMENTAL')]:
    for seed in SEEDS:
        with contextlib.redirect_stdout(io.StringIO()):
            r = cdacf.run_simulation(domain, time_steps=STEPS, seed=seed, verbose=False)
        rows.append(dict(domain=dlabel, seed=seed, system='CDACF',
                         latency=float(np.mean(r['latency'])),
                         throughput=float(np.mean(r['throughput'])),
                         energy=blended_energy(r['mechanism_counts']),
                         posture=float(np.mean(r['posture_index'])),
                         switches=int(r['n_switches'])))
        for m in MECHS:
            with contextlib.redirect_stdout(io.StringIO()):
                s = cdacf.run_static_baseline(domain, m, time_steps=STEPS, seed=seed)
            rows.append(dict(domain=dlabel, seed=seed, system=SHORT[m],
                             latency=s['latency'], throughput=s['throughput'],
                             energy=s['energy'], posture=s['posture_index'],
                             switches=0))
        print(f"{dlabel} seed {seed} done  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, 'results_30seed.csv'), index=False)
print(df.groupby(['domain', 'system'])[['latency', 'throughput', 'posture']]
        .agg(['mean', 'std']).round(4).to_string())
