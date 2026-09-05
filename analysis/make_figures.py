"""R4: regenerate Figures 3-7 from the 30-seed data. Print/greyscale safe."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
RESULTS = os.path.join(_ROOT, 'results')
FIGURES = os.path.join(_ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(FIGURES, exist_ok=True)
import iot_integration_bayesian_v4 as cdacf
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.edgecolor': '#666666', 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.labelcolor': '#222222', 'text.color': '#222222',
})

# One hue, two steps. Greyscale separation is large, and CDACF also carries a hatch.
BASE = '#AFC9DF'
HILITE = '#1F4E79'
GRID = '#DDDDDD'
ORDER = ['PoW', 'PoS', 'PBFT', 'PoA', 'DPoS', 'CDACF']

res = pd.read_csv(os.path.join(RESULTS, 'results_30seed.csv'))
abl = pd.read_csv(os.path.join(RESULTS, 'ablation_summary.csv'))
rng = np.random.default_rng(7)


def ci(v):
    m = rng.choice(v, size=(20000, len(v))).mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return v.mean() - lo, hi - v.mean()


def single(metric, domain, ylabel, title, fname, w, h, fmt='{:.3f}'):
    fig, ax = plt.subplots(figsize=(w, h))
    sub = res[res.domain == domain]
    means, errs = [], [[], []]
    for s in ORDER:
        v = sub[sub.system == s][metric].to_numpy(float)
        means.append(v.mean())
        lo, hi = ci(v)
        errs[0].append(lo)
        errs[1].append(hi)
    colors = [HILITE if s == 'CDACF' else BASE for s in ORDER]
    bars = ax.bar(ORDER, means, color=colors, width=0.62,
                  edgecolor='white', linewidth=2)
    bars[-1].set_hatch('///')
    bars[-1].set_edgecolor('white')
    ax.errorbar(ORDER, means, yerr=errs, fmt='none', ecolor='#444444',
                elinewidth=0.8, capsize=2.5)
    for x, m in zip(ORDER, means):
        ax.text(x, m, ' ' + fmt.format(m), ha='center', va='bottom', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, loc='left', pad=6)
    ax.set_ylim(0, max(means) * 1.20)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.5)
    fig.savefig(fname, dpi=600)
    plt.close(fig)


single('latency', 'TRAFFIC', 'Normalised latency',
       'Figure 3  Normalised latency, traffic profile',
       os.path.join(FIGURES, 'Fig3_Latency_Traffic.png'), 5.42, 2.08)

single('throughput', 'ENVIRONMENTAL', 'Normalised throughput',
       'Figure 4  Normalised throughput, environmental profile',
       os.path.join(FIGURES, 'Fig4_Throughput_Environmental.png'), 5.42, 2.11)


def grouped(metric, ylabel, title, fname, w, h, pct=False):
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(ORDER))
    for k, (dom, lab, col, hatch) in enumerate([
            ('TRAFFIC', 'Traffic profile', HILITE, ''),
            ('ENVIRONMENTAL', 'Environmental profile', BASE, '///')]):
        sub = res[res.domain == dom]
        m = [sub[sub.system == s][metric].mean() * (100 if pct else 1) for s in ORDER]
        b = ax.bar(x + (k - 0.5) * 0.36, m, width=0.34, color=col, label=lab,
                   edgecolor='white', linewidth=1.6, hatch=hatch)
        for xi, mi in zip(x + (k - 0.5) * 0.36, m):
            ax.text(xi, mi, f'{mi:.1f}' if pct else f'{mi:.2f}',
                    ha='center', va='bottom', fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, loc='left', pad=6)
    ax.legend(frameon=False, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 118 if pct else 1.12)
    fig.tight_layout(pad=0.5)
    fig.savefig(fname, dpi=600)
    plt.close(fig)


grouped('energy', 'Energy-efficiency attribute',
        'Figure 5  Energy-efficiency attribute of the active mechanism',
        os.path.join(FIGURES, 'Fig5_Energy_Attribute.png'), 5.62, 2.07)

grouped('posture', 'Composite posture index (%)',
        'Figure 6  Composite security-posture index (Equation 10)',
        os.path.join(FIGURES, 'Fig6_Posture_Index.png'), 5.62, 2.02, pct=True)

# Figure 7: gate ablation (the gate-attributable result) with ADACON as context
fig, ax = plt.subplots(figsize=(5.62, 3.4))
t = abl[abl.domain == 'Traffic']
labels = ['ADACON\nunconstrained\n[4]', 'ADACON\npost-hysteresis\n[4] Table 3',
          'CDACF\ngate disabled', 'CDACF\ndwell only', 'CDACF\nthreshold only',
          'CDACF\nfull gate']
vals = [699, 287,
        float(t[t.configuration.str.startswith('Gate disabled')].switches_mean.iloc[0]),
        float(t[t.configuration.str.startswith('No improvement')].switches_mean.iloc[0]),
        float(t[t.configuration.str.startswith('No dwell')].switches_mean.iloc[0]),
        float(t[t.configuration.str.startswith('Full stability')].switches_mean.iloc[0])]
cols = ['#CCCCCC', '#CCCCCC', BASE, BASE, BASE, HILITE]
bars = ax.bar(labels, vals, color=cols, width=0.6, edgecolor='white', linewidth=2)
bars[-1].set_hatch('///')
bars[-1].set_edgecolor('white')
ax.set_yscale('log')
ax.set_ylim(0.7, 2200)
for xi, v in zip(labels, vals):
    ax.text(xi, v, f' {v:.1f}' if v < 100 else f' {v:.0f}',
            ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Consensus transitions per 1,000 steps (log scale)', fontsize=8)
ax.set_title('Figure 7  Consensus switching per 1,000 steps', fontsize=8.5,
             loc='left', pad=6)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=6.6)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(FIGURES, 'Fig7_Switching.png'), dpi=600)
plt.close(fig)
print('figures written: fig3 fig4 fig5 fig6 fig7')
