"""RQ3 — Variation of the change with antecedent precipitation, soil moisture and catchment size, at patch level:
change in predicted flooded area from T0 to T30 per family, in 3 x 3 classes of two patch descriptors.

Descriptors (ch5_rq3_patch_antecedent.py, from the dynamic hydroclimatic variables the T30 networks receive):
  API   antecedent precipitation index of the upstream precipitation, k = 0.9: the daily precipitation of the
        30 days before the mapping date, weighted by 0.9 per day back in time from day -1 (mm)
  soil moisture of day -4, before the last 3 days (m3 m-3); upstream catchment area of the patch (km2)
Classes: terciles of each descriptor over the 51,606 test patches. In a class, TP and FP are summed over the scored
cells of its patches and over seeds 1-3; the change is (TP + FP) of T30 over that of T0, in per cent.
Bold values: 95% cluster bootstrap interval excludes zero (61 clusters, 4,000 replicates, clusters_boot.npz).
The number of patches is written under each value. Replicates that draw none of a class's clusters are left out
of that class's interval.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_21_t30_antecedent.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm

import rq1_common as rc
import rq3_common as r3
from rq1_common import ps

C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
API = 'api_up_0.90'
PAIRS = (('sm_before', 'Soil moisture of day −4 (m³ m⁻³)', '{:.2f}'),
         ('area_km2', 'Upstream catchment area (km²)', '{:.0f}'))
LIM = 30                                        # colour scale of the change, per cent


def data():
    p = pd.read_csv(C / 'patches_test.csv.gz').sort_values('row_index').reset_index(drop=True)
    cb = np.load(C / 'clusters_boot.npz')
    p['cluster'] = cb['cluster']
    a = pd.read_csv(r3.D / 'patch_antecedent.csv')
    x = p.merge(a, on=['folder_name', 'patch_number'], how='left')
    if x[API].isna().any() or len(x) != len(p):
        raise SystemExit('patches without antecedent descriptors')
    return x, cb['weights'].astype(float)


def area_change(s, f, W):
    """Per cent change of predicted flood cells (TP + FP, summed over seeds) from T0 to T30, with 95% interval."""
    nC = W.shape[1]
    pred = {}
    for w in r3.WINDOWS:
        cl = sum(np.bincount(s.cluster, s[f'{f}_{w}_s{k}_tp'] + s[f'{f}_{w}_s{k}_fp'], nC) for k in rc.SEEDS)
        pred[w] = (cl.sum(), W @ cl)
    point = 100 * (pred['t30'][0] / pred['t0'][0] - 1)
    ok = pred['t0'][1] > 0                      # replicates that draw none of the class's clusters are left out
    lo, hi = np.percentile(100 * (pred['t30'][1][ok] / pred['t0'][1][ok] - 1), [2.5, 97.5])
    return point, lo, hi


def tercile_labels(edges, fmt):
    e = [fmt.format(v) for v in edges]
    return [f'<{e[1]}', f'{e[1]}–{e[2]}', f'≥{e[2]}']


def main():
    rc.style()
    x, W = data()
    a, ea = pd.qcut(x[API], 3, labels=False, retbins=True)
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH_IN, 5.0))
    norm = TwoSlopeNorm(vmin=-LIM, vcenter=0, vmax=LIM)
    base = plt.get_cmap('BrBG')(np.linspace(0, 1, 256))[:, :3]
    cmap = ListedColormap(1 - 0.55 * (1 - base))  # BrBG blended towards white, so black labels stay readable
    report = []
    for i, (col, name, fmt) in enumerate(PAIRS):
        b, eb = pd.qcut(x[col], 3, labels=False, retbins=True)
        for j, f in enumerate(rc.FAMILIES):
            ax = axes[i, j]
            M = np.full((3, 3), np.nan)
            for r in range(3):
                for c in range(3):
                    s = x[(a == r) & (b == c)]
                    v, lo, hi = area_change(s, f, W)
                    M[r, c] = v
                    sig = lo > 0 or hi < 0
                    ax.text(c, r - 0.08, f'{v:+.0f}%'.replace('-', '−'), ha='center', va='center', fontsize=ps.FONT_SIZE - 0.5,
                            fontweight='bold' if sig else 'normal', color='black')
                    ax.text(c, r + 0.24, f'n = {len(s):,}', ha='center', va='center', fontsize=ps.FONT_SIZE - 3,
                            color='#333333')
                    report.append((col, f, r, c, len(s), v, lo, hi, 100 * s.flood.sum() / s.scored.sum()))
            im = ax.imshow(M, cmap=cmap, norm=norm, origin='lower', aspect='auto')
            ax.set_xticks(range(3), tercile_labels(eb, fmt), fontsize=ps.FONT_SIZE - 2)
            ax.set_yticks(range(3), tercile_labels(ea, '{:.0f}'), fontsize=ps.FONT_SIZE - 2)
            ax.set_xlabel(name, fontsize=ps.FONT_SIZE - 1.5)
            if j == 0:
                ax.set_ylabel('API of upstream\nprecipitation (mm)', fontsize=ps.FONT_SIZE - 1.5)
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            rc.panel_title(ax, f'({"abcd"[2 * i + j]}) {rc.LABEL[f]}')
    cax = fig.add_axes([0.25, 0.045, 0.5, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Change in predicted flooded area, T30 vs T0 (%)', fontsize=ps.FONT_SIZE - 1.5)
    cb.ax.tick_params(labelsize=ps.FONT_SIZE - 2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.4, w_pad=1.5)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    r = pd.DataFrame(report, columns=['descriptor', 'family', 'api_class', 'class', 'patches', 'change', 'lo', 'hi',
                                      'flooded_pct'])
    pd.set_option('display.width', 200)
    print(r.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
