"""RQ4 — Response of the networks of every input-history length to the precipitation of single days (encoder probes,
scripts/encoder_probe.py; the same 2,000 test patches for every fit; each network receives its
last T days; probe.npz of T30 under final_t30/, of T3, T7 and T15 under final_window/).

For each removed day, the change of the predicted flood cells (prob >= the fit's selected threshold, scored cells) of
the 2,000 patches when the local and upstream precipitation of that day is set to zero, in per cent of the predicted
flood cells with the true inputs; averaged over the days of a range (days -3 to -1, -7 to -4, -15 to -8, -30 to -16)
and over seeds 1-3 (Ravi, 14 Oct 2026: seed means only). A length has a bar only in the ranges it receives. T3 to T30
in the family colour, lighter for shorter histories.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_30_window_probes.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import rq1_common as rc
import rq4_common as r4
from rq1_common import ps

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
LENGTHS = (('t3', 3), ('t7', 7), ('t15', 15), ('t30', 30))
RANGES = ((1, 3, 'Days −3 to −1'), (4, 7, 'Days −7 to −4'), (8, 15, 'Days −15 to −8'), (16, 30, 'Days −30 to −16'))
STRENGTH = {'t3': 0.35, 't7': 0.55, 't15': 0.78, 't30': 1.0}
WIDTH = 0.19


def probe(f, w, s):
    d = RUNS / ('final_t30' if w == 't30' else 'final_window') / f'seed{s}' / (f if w == 't30' else f'{f}_{w}')
    return np.load(d / 'inference_probe' / 'probe.npz')


def per_day(f, w):
    """Mean over seeds of the change of predicted flood cells (%) for removing each day; index 0 = day -1."""
    out = []
    for s in rc.SEEDS:
        z = probe(f, w, s)
        v = z['occl_dpred'].sum(1) / z['pred0'].sum() * 100           # stored order: day -T ... day -1
        out.append(v[::-1])
    return np.mean(out, axis=0)


def main():
    rc.style()
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 4.8))
    for ax, (k, f) in zip(axes, enumerate(rc.FAMILIES)):
        lo = 0.0
        for j, (w, T) in enumerate(LENGTHS):
            v = per_day(f, w)
            for g, (a, b, _) in enumerate(RANGES):
                if a > T:
                    continue
                m = float(v[a - 1:min(b, T)].mean())
                x = g + (j - 1.5) * (WIDTH + 0.01)
                ax.bar(x, m, WIDTH, color=ps.alpha_on_white(rc.COLOR[f], STRENGTH[w]), edgecolor=rc.COLOR[f],
                       lw=0.6, zorder=2)
                ax.annotate(f'{m:.2f}'.replace('-', '−'), (x, m), xytext=(0, -1.5), textcoords='offset points',
                            ha='center', va='top', fontsize=ps.FONT_SIZE - 3, zorder=4)
                lo = min(lo, m)
                print(rc.LABEL[f], r4.WLABEL[w], RANGES[g][2], f'{m:.3f}')
        ax.set_ylim(lo * 1.3, -lo * 0.08)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        ax.set_xticks(range(len(RANGES)), [n for _, _, n in RANGES])
        ax.tick_params(axis='x', length=0, pad=3)
        ax.set_xlim(-0.55, len(RANGES) - 0.45)
        ax.set_ylabel('Change per day (%)')
        rc.ygrid(ax)
        rc.panel_title(ax, f'({"ab"[k]}) {rc.LABEL[f]}, change in predicted flood cells when one day is removed')
        h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], STRENGTH[w]), edgecolor=rc.COLOR[f], lw=0.6,
                   label=r4.WLABEL[w]) for w, _ in LENGTHS]
        ax.legend(handles=h, loc='lower right', ncol=4, frameon=False, fontsize=ps.FONT_SIZE - 1.5,
                  handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(h_pad=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
