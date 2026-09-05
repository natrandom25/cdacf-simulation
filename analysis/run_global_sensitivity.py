"""R4: global (probabilistic) sensitivity of the CDACF conclusions to the
mechanism attribute matrix (Table 3) and the domain weight profiles (Table 4).

R3 varied only the gate threshold, dwell and Bayesian window. Reviewer 2 asks
whether the conclusions persist when the *uncertain hand-set scores* are
perturbed. This script perturbs all 25 attribute cells and every weight profile
by Gaussian noise, renormalises the weights, and re-runs both domains.

Recorded per draw:
  Q1  which mechanism CDACF spends most steps on, per domain
  Q2  does static PoS remain the environmental incumbent
  Q3  does CDACF still hold the highest posture index among *feasible*
      mechanisms (H1 as stated: PBFT, PoA, DPoS excluded)
  Q4  does CDACF still hold the highest posture index among ALL mechanisms
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import copy, io, contextlib, sys, time
import numpy as np, pandas as pd
from iot_integration_bayesian_v4 import IoTDomain, ConsensusType, DomainAwareConsensusEngine as ENG

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SIGMAS = [0.05, 0.10]
STEPS = 1000
MECHS = list(ConsensusType)
SHORT = {ConsensusType.POW: 'PoW', ConsensusType.POS: 'PoS',
         ConsensusType.PBFT: 'PBFT', ConsensusType.POA: 'PoA',
         ConsensusType.DPOS: 'DPoS'}
FEASIBLE = [ConsensusType.POS, ConsensusType.POW]   # H1's deployable set: PBFT
# excluded on O(n^2) scalability, PoA and DPoS on decentralisation.

BASE_ATTR = copy.deepcopy(ENG.MECHANISM_ATTRIBUTES)
BASE_W = copy.deepcopy(ENG.WEIGHT_PROFILES)
rng = np.random.default_rng(12345)


def perturb(sigma):
    attr = copy.deepcopy(BASE_ATTR)
    for m in attr:
        for k in attr[m]:
            attr[m][k] = float(np.clip(attr[m][k] + rng.normal(0, sigma), 0.01, 1.0))
    w = copy.deepcopy(BASE_W)
    for dom in w:
        for prof in w[dom]:
            vals = {k: max(0.01, v + rng.normal(0, sigma)) for k, v in w[dom][prof].items()}
            tot = sum(vals.values())
            w[dom][prof] = {k: v / tot for k, v in vals.items()}
    return attr, w


rows = []
t0 = time.time()
for sigma in SIGMAS:
    for draw in range(N_DRAWS):
        attr, wts = perturb(sigma)
        ENG.MECHANISM_ATTRIBUTES = attr
        ENG.WEIGHT_PROFILES = wts
        seed = 1000 + draw
        rec = dict(sigma=sigma, draw=draw, seed=seed)
        for domain, dl in [(IoTDomain.TRAFFIC, 'T'), (IoTDomain.ENVIRONMENT, 'E')]:
            with contextlib.redirect_stdout(io.StringIO()):
                r = cdacf.run_simulation(domain, time_steps=STEPS, seed=seed, verbose=False)
            counts = r['mechanism_counts']
            top = max(counts, key=counts.get)
            rec[f'{dl}_top_mech'] = top
            rec[f'{dl}_top_share'] = counts[top] / sum(counts.values())
            rec[f'{dl}_switches'] = int(r['n_switches'])
            rec[f'{dl}_cdacf_posture'] = float(np.mean(r['posture_index']))
            for m in MECHS:
                with contextlib.redirect_stdout(io.StringIO()):
                    s = cdacf.run_static_baseline(domain, m, time_steps=STEPS, seed=seed)
                rec[f'{dl}_{SHORT[m]}_posture'] = s['posture_index']
        rows.append(rec)
        if draw % 50 == 0:
            print(f"sigma={sigma} draw={draw}  ({time.time()-t0:.0f}s)", flush=True)

ENG.MECHANISM_ATTRIBUTES = BASE_ATTR
ENG.WEIGHT_PROFILES = BASE_W

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, 'global_sensitivity.csv'), index=False)

print('\n' + '=' * 78)
print('GLOBAL SENSITIVITY OF THE CDACF CONCLUSIONS')
print('=' * 78)
for sigma in SIGMAS:
    d = df[df.sigma == sigma]
    n = len(d)
    print(f"\n--- perturbation sd = {sigma} on every Table 3 cell and Table 4 weight "
          f"(n = {n} draws) ---")

    print("\nQ1  Mechanism CDACF spends most steps on")
    for dl, name in [('T', 'Traffic'), ('E', 'Environmental')]:
        vc = d[f'{dl}_top_mech'].value_counts(normalize=True)
        print(f"  {name:<14}" + "  ".join(f"{k.split()[-1] if ' ' in k else k}: {v:.0%}"
                                          for k, v in vc.items()))

    pos_name = ConsensusType.POS.value
    q2 = (d['E_top_mech'] == pos_name).mean()
    print(f"\nQ2  PoS remains the environmental incumbent:      {q2:.1%} of draws")

    feas = [SHORT[m] for m in FEASIBLE]
    q3 = (d.apply(lambda r: r['T_cdacf_posture'] >
                  max(r[f'T_{b}_posture'] for b in feas), axis=1)).mean()
    print(f"Q3  CDACF highest posture among feasible set:     {q3:.1%} of draws")

    allb = list(SHORT.values())
    q4 = (d.apply(lambda r: r['T_cdacf_posture'] >
                  max(r[f'T_{b}_posture'] for b in allb), axis=1)).mean()
    print(f"Q4  CDACF highest posture among ALL mechanisms:   {q4:.1%} of draws")

    q5 = (d.apply(lambda r: max(allb, key=lambda b: r[f'T_{b}_posture']) == 'PBFT',
                  axis=1)).mean()
    print(f"Q5  PBFT is the top-posture static (traffic):     {q5:.1%} of draws")

    print(f"\nSwitch count, traffic:  mean {d['T_switches'].mean():.1f}  "
          f"sd {d['T_switches'].std():.1f}  range {d['T_switches'].min()}-{d['T_switches'].max()}")
    print(f"Switch count, environ:  mean {d['E_switches'].mean():.1f}  "
          f"sd {d['E_switches'].std():.1f}  range {d['E_switches'].min()}-{d['E_switches'].max()}")
