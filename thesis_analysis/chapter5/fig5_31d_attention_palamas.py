"""One AOI mapped six times: attention of U-Net-T30 and ViT-T15 on the calendar days (Section 5.4.3).

EMSR692 AOI02 Palamas (Thessaly, Greece), mapped on 10, 12, 14, 15, 17 and 19 September 2023. The six maps share
210 patches with identical bounds; the 131 that are flood patches in all six maps are used. Each map's input history
ends the day before its mapping date, so the storm precipitation of 4-6 September sits at a different position in
each history. Rows = maps, columns = calendar days, colour = seed-mean attention averaged over the 131 patches,
relative to uniform. A box marks each map's input history, hatching the days outside it, a tick the mapping date.
Attention that follows the precipitation forms an upright stripe on the storm days; attention fixed to the position
in the input history forms a diagonal that moves with the mapping date. The numbers of all four T30/T15 networks are
printed for the text.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import rq1_common as rc
from rq1_common import ps
import rq4_attention_common as ac

AOI = 'EMSR692_AOI02_Palamas'
STORM = ('2023-09-04', '2023-09-05', '2023-09-06')
HATCH = dict(facecolor='white', hatch='////', edgecolor='#BDBDBD', lw=0)


def main():
    fn, pn, flood, act, area = ac.rows()
    prods = sorted({f for f in fn if f.startswith(AOI)}, key=lambda s: s[-8:])
    count = {}
    for p in prods:
        for i in np.where((fn == p) & (flood > 0))[0]:
            count[pn[i]] = count.get(pn[i], 0) + 1
    keep = sorted(k for k, v in count.items() if v == len(prods))
    rows = {p: np.array([np.where((fn == p) & (pn == k))[0][0] for k in keep]) for p in prods}
    mdates = [pd.Timestamp(p[-8:]) for p in prods]
    cal = pd.date_range(mdates[0] - pd.Timedelta(days=30), mdates[-1])
    col = {d: j for j, d in enumerate(cal)}
    P30 = ac.upstream(30)
    rain = np.full(len(cal), np.nan)
    for p, d0 in zip(prods, mdates):
        P = P30[rows[p]].mean(0)
        for j in range(30):
            rain[col[d0 - pd.Timedelta(days=30 - j)]] = P[j]

    def grid(arch, T):
        A = ac.seed_mean(arch, T)
        M = np.full((len(prods), len(cal)), np.nan)
        for i, (p, d0) in enumerate(zip(prods, mdates)):
            m = A[rows[p]].mean(0)
            for j in range(T):
                M[i, col[d0 - pd.Timedelta(days=T - j)]] = m[j]
        return M

    storm = [col[pd.Timestamp(d)] for d in STORM]
    print(f'{len(keep)} patches flooded in all {len(prods)} maps; storm rain ' +
          ', '.join(f'{cal[j]:%d %b} {rain[j]:.0f} mm' for j in storm))
    for arch, T in (('unet', 30), ('vit', 30), ('unet', 15), ('vit', 15)):
        M = grid(arch, T)
        for i, d0 in enumerate(mdates):
            r = M[i]; inside = ~np.isnan(r); st = r[storm]; other = r[inside & ~np.isin(np.arange(len(cal)), storm)]
            top = int(np.nanargmax(r))
            print(f'  {rc.LABEL[arch]}-T{T} map {d0:%d %b}: storm days {np.nanmean(st):.3f}, other days {other.mean():.3f}, '
                  f'range {np.nanmin(r):.3f}-{np.nanmax(r):.3f}, top day {cal[top]:%d %b} = day {(cal[top] - d0).days}')

    ps.apply_clear()
    norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.8)
    cmap = mpl.colormaps['RdBu_r'].copy()
    cmap.set_bad((0, 0, 0, 0))                                                    # empty cells show the hatching
    ticks = [j for j, d in enumerate(cal) if (d.month == 8 and d.day in (15, 20, 25, 30))
             or (d.month == 9 and d.day in (5, 10, 15))]

    def date_axis(ax):
        ax.set_xticks(ticks, [f'{cal[j].day} {cal[j]:%b}' for j in ticks], fontsize=ps.FONT_SIZE - 2)
        ax.set_xticks(np.arange(len(cal)), minor=True)
        ax.tick_params(axis='x', which='minor', length=2, width=0.5, color='#555555')
        ax.tick_params(axis='x', which='major', length=4)
        ax.set_xlim(-0.5, len(cal) - 0.5)

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 5.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.2, 1.2], hspace=0.62, left=0.13, right=0.98, top=0.955,
                          bottom=0.2)
    ax = fig.add_subplot(gs[0])
    ax.bar(np.arange(len(cal)), rain, 0.8, color='#BDBDBD')
    date_axis(ax)
    rc.ygrid(ax)
    ax.set_ylabel('mm/day')
    rc.panel_title(ax, f'(a) Upstream precipitation, mean of the {len(keep)} flood patches')
    for r, (arch, T) in enumerate((('unet', 30), ('vit', 15))):
        M = grid(arch, T)
        ax = fig.add_subplot(gs[r + 1])
        ax.add_patch(Rectangle((-0.5, -0.5), len(cal), len(prods), zorder=0, **HATCH))
        im = ax.imshow(M, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest', zorder=1)
        for i, d0 in enumerate(mdates):
            s0 = col[d0 - pd.Timedelta(days=T)]
            ax.add_patch(Rectangle((s0 - 0.5, i - 0.5), T, 1, fill=False, edgecolor='#333333', lw=0.7, zorder=3))
            ax.plot(col[d0], i, marker='|', color='black', ms=6, mew=1.4, zorder=4)
        date_axis(ax)
        ax.set_yticks(range(len(prods)), [f'{d.day} {d:%b}' for d in mdates], fontsize=ps.FONT_SIZE - 2.5)
        ax.set_ylim(len(prods) - 0.5, -0.5)
        ax.set_ylabel('Map of', fontsize=ps.FONT_SIZE - 1.5)
        rc.panel_title(ax, f'({"bc"[r]}) {rc.LABEL[arch]}-T{T}')
        for s in ax.spines.values():
            s.set_visible(False)
    fig.legend(handles=[Patch(facecolor='white', edgecolor='#333333', lw=0.7, label='Input history of the map'),
                        Patch(label='Outside the input history', **HATCH),
                        Line2D([], [], color='black', marker='|', ls='none', ms=7, mew=1.4, label='Mapping date')],
               loc='lower left', bbox_to_anchor=(0.01, 0.0), frameon=False, fontsize=ps.FONT_SIZE - 2,
               handlelength=1.6)
    cax = fig.add_axes([0.46, 0.065, 0.50, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    cb.set_label('Attention relative to uniform, mean of three seeds', fontsize=ps.FONT_SIZE - 1.5)
    cb.ax.tick_params(labelsize=ps.FONT_SIZE - 2)
    with mpl.rc_context({'savefig.bbox': None}):
        ps.save(fig, __file__, subdir='5_4')


if __name__ == '__main__':
    main()
