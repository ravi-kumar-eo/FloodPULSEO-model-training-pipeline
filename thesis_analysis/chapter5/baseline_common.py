"""Shared data and drawing of the comparison with the GFF models (Section 5.6, Appendix A.6).
Data: plot/data/baseline/sites.csv (ch5_baseline_data.py). Ours = mean of seeds 1-3 of one architecture and
input-history length; GFF = U-TAE and MaxViT U-Net as released by Victor et al. (2024)."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'baseline'
GFF_MODELS = ('gff_utae', 'gff_maxvit')
LABEL = {'gff_utae': 'GFF U-TAE', 'gff_maxvit': 'GFF MaxViT U-Net'}
for _a in ('unet', 'vit'):
    for _w in ('t0', 't3', 't7', 't15', 't30'):
        LABEL[f'{_a}_{_w}'] = f"{rc.LABEL[_a]}-{_w.upper()}"
LABEL.update({'lf': 'LISFLOOD-FP', 'unet_best': 'Best U-Net', 'vit_best': 'Best ViT'})
COLOR = {'gff_utae': '#4D4D4D', 'gff_maxvit': '#A6A6A6', 'lf': '#4D4D4D', 'unet': rc.COLOR['unet'],
         'vit': rc.COLOR['vit']}
MARKER = {'gff_utae': 's', 'gff_maxvit': 'D', 'lf': 's', 'unet': 'o', 'vit': '^'}
MONTH = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
         'November', 'December')


def read(label_type):
    d = pd.read_csv(D / 'sites.csv')
    return d[d.label_type == label_type].copy()


def colour(group):
    return COLOR[group] if group in COLOR else COLOR[group.split('_')[0]]


def marker(group):
    return MARKER[group] if group in MARKER else MARKER[group.split('_')[0]]


def site_order(d):
    """Sites grouped by flood event (events in date order), largest flood first within an event."""
    s = d.drop_duplicates('site')[['site', 'event', 'flood_date', 'flood_cells', 'country', 'our_test_event']]
    return s.sort_values(['flood_date', 'event', 'flood_cells'], ascending=[True, True, False]).reset_index(drop=True)


def site_panel(ax, d, groups, order, title, event_labels=True):
    """IoU of each group at each site; sites on x, grouped by event with alternating background bands."""
    x = {s: i for i, s in enumerate(order.site)}
    piv = d.pivot_table(index='site', columns='group', values='IoU')
    offs = np.linspace(-0.27, 0.27, len(groups))
    for g, dx in zip(groups, offs):
        v = piv.loc[order.site, g].to_numpy()
        ax.scatter(np.arange(len(order)) + dx, v, s=16, marker=marker(g), color=colour(g), edgecolor='black',
                   linewidth=0.3, zorder=3)
    starts = order.groupby('event', sort=False).site.apply(lambda s: (x[s.iloc[0]], x[s.iloc[-1]]))
    for k, (ev, (a, b)) in enumerate(starts.items()):
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color='#F2F2F2', zorder=0, lw=0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel('IoU')
    ax.set_ylim(0, None)
    rc.ygrid(ax)
    rc.panel_title(ax, title)
    names = ['-'.join(s.split('-')[:2]) if event_labels else s.split('-')[0] for s in order.site]
    names = [n + ('†' if te else '') for n, te in zip(names, order.our_test_event)]
    ax.set_xticks(np.arange(len(order)), names, fontsize=ps.FONT_SIZE - 2, rotation=90)
    ax.tick_params(axis='x', length=0, pad=2)
    return starts


def event_axis(ax, order, starts):
    """Event name and country under the site numbers; a dagger marks floods that are also test events of the thesis."""
    first = order.drop_duplicates('event').set_index('event')
    for ev, (a, b) in starts.items():
        r = first.loc[ev]
        name = f"EMSR{ev}{'†' if r.our_test_event else ''}\n{r.country}"
        ax.annotate(name, ((a + b) / 2, 0), xycoords=('data', 'axes fraction'), xytext=(0, -14),
                    textcoords='offset points', ha='center', va='top', fontsize=ps.FONT_SIZE - 3, linespacing=1.1)


def handles(groups):
    return [Line2D([], [], ls='none', marker=marker(g), ms=4.5, mfc=colour(g), mec='black', mew=0.3, label=LABEL[g])
            for g in groups]


def site_names(order, long=False):
    """Names of the sites in the order of site_order(): country and month of the flood, and the number of the site
    within its flood event (1 = most labelled flood) where an event has several sites (Ravi, 27 Oct 2026: no dataset
    identifiers). long=True writes the full date and 'site n'."""
    n = order.groupby('event', sort=False).cumcount() + 1
    size = order.groupby('event', sort=False).site.transform('count')
    out = []
    for r, k, m in zip(order.itertuples(), n, size):
        y, mo, dd = r.flood_date.split('-')
        if long:
            s = f'{r.country}, {int(dd)} {MONTH[int(mo) - 1]} {y}' + (f', site {k}' if m > 1 else '')
        else:
            s = f'{r.country}, {MONTH[int(mo) - 1][:3]} {y}' + (f', {k}' if m > 1 else '')
        out.append(s)
    return out


def bar_rows(d, groups, labels, xmax=(0.6, 1.0, 0.5), values=(True, True, True), bar=0.2, gap=0.3, height=6.6):
    """Horizontal grouped bars: one group per row of d (top to bottom), one bar per model of groups, with (a) IoU,
    (b) POD and (c) POFD side by side (columns of d: '<group>_<IoU|POD|POFD>'). values: write the value at the end of
    each bar in that panel. Rows are separated by alternating grey bands."""
    import matplotlib.pyplot as plt
    n, k = len(d), len(groups)
    step = k * bar + gap
    y0 = np.arange(n) * step
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, height), sharey=True)
    for ax, met, title, xm, show in zip(axes, ('IoU', 'POD', 'POFD'), ('(a) IoU', '(b) POD', '(c) POFD'), xmax, values):
        for i in range(0, n, 2):
            ax.axhspan(y0[i] - gap / 2, y0[i] + k * bar - gap / 2 + gap, color='#F2F2F2', lw=0, zorder=0)
        for j, g in enumerate(groups):
            v = d[f'{g}_{met}'].to_numpy(float)
            yy = y0 + j * bar + bar / 2
            ax.barh(yy, v, bar * 0.92, color=ps.alpha_on_white(colour(g), 0.8), edgecolor=colour(g), lw=0.5, zorder=3)
            if show:
                for x_, y_ in zip(v, yy):
                    s = f'{x_:.3f}'
                    ax.text(x_ + xm * 0.012, y_, s[1:] if s.startswith('0.') else s, va='center', ha='left',
                            fontsize=ps.FONT_SIZE - 3.5, zorder=4)
        ax.set_xlim(0, xm)
        ax.set_xticks(np.arange(0, xm + 1e-9, 0.1 if xm <= 0.5 else 0.2))
        ax.set_ylim(y0[-1] + k * bar + gap / 2, -gap / 2)
        ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=ps.FONT_SIZE - 2)
        rc.panel_title(ax, title)
    axes[0].set_yticks(y0 + k * bar / 2, labels, fontsize=ps.FONT_SIZE - 2)
    from matplotlib.patches import Patch
    hl = [Patch(facecolor=ps.alpha_on_white(colour(g), 0.8), edgecolor=colour(g), lw=0.5, label=LABEL[g]) for g in groups]
    fig.legend(handles=hl, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(groups), frameon=False,
               fontsize=ps.FONT_SIZE - 1.5, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.3 / fig.get_figheight(), 1, 1), w_pad=0.6)
    return fig


def stacked_shares(ax, shares, cats, title, values=True, bar=0.62):
    """Horizontal stacked bars, one per row of shares (fractions summing to 1); cats: (key, label, colour). values:
    write each share as a percentage in or beside its segment."""
    for i, row in enumerate(shares):
        left = 0.0
        for (k, _, c), v in zip(cats, row):
            ax.barh(i, v, bar, left=left, color=ps.alpha_on_white(c, 0.6), edgecolor=c, lw=0.6, zorder=2)
            if values and v >= 0.0005:
                t = f'{v * 100:.1f}%'
                if v >= 0.06:
                    ax.text(left + v / 2, i, t, ha='center', va='center', fontsize=ps.FONT_SIZE - 2.5)
                else:
                    ax.annotate(t, (left + v, i), xytext=(2.5, 0), textcoords='offset points', ha='left',
                                va='center', fontsize=ps.FONT_SIZE - 2.5)
            left += v
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylim(len(shares) - 0.5, -0.55)
    ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5, zorder=0)
    rc.panel_title(ax, title)
