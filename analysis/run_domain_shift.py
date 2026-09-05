"""R4: out-of-distribution robustness and a held-out third domain.

Answers Reviewer 2 point 3 (cross-domain generality not demonstrated).

Part A  Domain shift. Node scale, base latency, base packet loss and telemetry
        noise are pushed well outside the values the framework was calibrated on,
        with no retuning of weights, priors, thresholds or gate parameters.
Part B  Held-out domain. A third profile (Industrial Automation, ~800 nodes,
        private 5G, security-and-finality first) never used during any tuning.

Reported outcome is the rebuilt H1: does the architecture still converge to a
domain-appropriate incumbent and keep switching bounded, without retuning.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import copy, io, contextlib, itertools, time
import numpy as np, pandas as pd
from iot_integration_bayesian_v4 import IoTDomain, ConsensusType, DomainAwareConsensusEngine as ENG, \
    TrafficNetworkSimulator as TSim, EnvironmentalNetworkSimulator as ESim

SHORT = {'Proof of Work': 'PoW', 'Proof of Stake': 'PoS',
         'Practical Byzantine Fault Tolerance': 'PBFT',
         'Proof of Authority': 'PoA', 'Delegated Proof of Stake': 'DPoS'}
SEEDS = [5, 6, 7, 8, 9]
STEPS = 1000
BASE_W = copy.deepcopy(ENG.WEIGHT_PROFILES)
T0 = dict(N=TSim.TOTAL_NODES, L=TSim.BASE_LATENCY, P=TSim.BASE_PACKET_LOSS)
E0 = dict(N=ESim.TOTAL_NODES, L=ESim.BASE_LATENCY, P=ESim.BASE_PACKET_LOSS)


def run(domain, seeds=SEEDS):
    out = []
    for s in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            r = cdacf.run_simulation(domain, time_steps=STEPS, seed=s, verbose=False)
        c = r['mechanism_counts']
        top = max(c, key=c.get)
        out.append(dict(seed=s, incumbent=SHORT[top],
                        share=c[top] / sum(c.values()),
                        switches=int(r['n_switches']),
                        latency=float(np.mean(r['latency'])),
                        throughput=float(np.mean(r['throughput'])),
                        posture=float(np.mean(r['posture_index']))))
    d = pd.DataFrame(out)
    inc = d.incumbent.mode()[0]
    return dict(incumbent=inc, incumbent_agreement=(d.incumbent == inc).mean(),
                share=d.share.mean(), switches=d.switches.mean(),
                switches_sd=d.switches.std(), latency=d.latency.mean(),
                throughput=d.throughput.mean(), posture=d.posture.mean())


def set_traffic(N=None, L=None, P=None):
    TSim.TOTAL_NODES = N or T0['N']
    TSim.BASE_LATENCY = L if L is not None else T0['L']
    TSim.BASE_PACKET_LOSS = P if P is not None else T0['P']


def set_env(N=None, L=None, P=None):
    ESim.TOTAL_NODES = N or E0['N']
    ESim.BASE_LATENCY = L if L is not None else E0['L']
    ESim.BASE_PACKET_LOSS = P if P is not None else E0['P']


# ------------------------------------------------------------------ Part A
rows = []
t0 = time.time()
GRID = {
    'node scale':   [('x0.1', 0.1), ('x0.5', 0.5), ('calibrated', 1.0),
                     ('x2', 2.0), ('x5', 5.0)],
    'base latency': [('x0.5', 0.5), ('calibrated', 1.0), ('x2', 2.0), ('x4', 4.0)],
    'packet loss':  [('x0.5', 0.5), ('calibrated', 1.0), ('x3', 3.0), ('x6', 6.0)],
}
for axis, levels in GRID.items():
    for label, mult in levels:
        for dom, dl, setter, base in [(IoTDomain.TRAFFIC, 'Traffic', set_traffic, T0),
                                      (IoTDomain.ENVIRONMENT, 'Environmental', set_env, E0)]:
            kw = {}
            if axis == 'node scale':
                kw['N'] = max(50, int(base['N'] * mult))
            elif axis == 'base latency':
                kw['L'] = base['L'] * mult
            else:
                kw['P'] = min(0.45, base['P'] * mult)
            setter(**kw)
            r = run(dom)
            setter()
            rows.append(dict(part='A: domain shift', axis=axis, level=label,
                             domain=dl, **r))
    print(f"  {axis} done ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ Part B
# Held-out domain: Industrial Automation. 800 nodes, private 5G, tight finality
# budget, security-first. Never used to tune any parameter.
HELD_OUT = {
    'normal':      {'security': 0.30, 'finality_speed': 0.30, 'scalability': 0.18,
                    'decentralization': 0.12, 'energy_efficiency': 0.10},
    'high_threat': {'security': 0.50, 'finality_speed': 0.26, 'scalability': 0.12,
                    'decentralization': 0.07, 'energy_efficiency': 0.05},
    'emergency':   {'security': 0.58, 'finality_speed': 0.30, 'scalability': 0.06,
                    'decentralization': 0.03, 'energy_efficiency': 0.03},
}
ENG.WEIGHT_PROFILES[IoTDomain.TRAFFIC] = HELD_OUT
set_traffic(N=800, L=0.030, P=0.012)
r = run(IoTDomain.TRAFFIC)
rows.append(dict(part='B: held-out domain', axis='industrial automation',
                 level='800 nodes, private 5G', domain='Industrial', **r))
ENG.WEIGHT_PROFILES = BASE_W
set_traffic()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, 'domain_shift.csv'), index=False)

pd.set_option('display.width', 200)
print('\n' + '=' * 104)
print('OUT-OF-DISTRIBUTION ROBUSTNESS (no retuning of weights, priors, thresholds or gate)')
print('=' * 104)
print(df[['part', 'axis', 'level', 'domain', 'incumbent', 'incumbent_agreement',
          'share', 'switches', 'switches_sd', 'posture']]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print('\nSummary')
a = df[df.part.str.startswith('A')]
for dl in ['Traffic', 'Environmental']:
    s = a[a.domain == dl]
    cal = s[s.level == 'calibrated'].incumbent.iloc[0]
    print(f"  {dl:<14} incumbent stays {cal:<5} in {(s.incumbent == cal).mean():.0%} "
          f"of {len(s)} shifted configurations; "
          f"switches {s.switches.min():.1f} to {s.switches.max():.1f}")
b = df[df.part.str.startswith('B')].iloc[0]
print(f"  Held-out       incumbent {b.incumbent} ({b.share:.0%} of steps), "
      f"{b.switches:.1f} switches per 1,000 steps, seed agreement "
      f"{b.incumbent_agreement:.0%}")
