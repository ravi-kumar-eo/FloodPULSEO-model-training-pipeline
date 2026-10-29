"""Shared drawing of the basin-regime figures (RQ2): basin-seen against basin-unseen test ground, T0 networks, seed
means. Regime colours: a pair with a large difference in lightness and hue, so adjacent bars stay apart (the blue and
green of the Chapter 4 split map have nearly equal lightness and blend side by side).
Data: plot/data/conditions/rq2_*.csv (ch5_regime_screen.py, ch5_regime_regions.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
REG = ('seen', 'unseen')
REG_FACE = {'seen': '#5BA36B', 'unseen': '#9277B8'}              # green and purple (Ravi chose E, 13 Oct)
REG_COLOR = {'seen': '#3F7A4C', 'unseen': '#6C5490'}             # outlines
REG_DARK = {'seen': '#2F7A40', 'unseen': '#6A4C9C'}              # basins on the climate map (Figure 5.13)
REG_LABEL = {'seen': 'Basin-seen', 'unseen': 'Basin-unseen'}
FILL = 0.8                                                       # slightly lighter fill (Ravi, 13 Oct); outlines keep the full colour


def face(r):
    return ps.alpha_on_white(REG_FACE[r], FILL)


def pair_bars(ax, x, vals, width=0.34, labels=None, fmt='{:.3f}'):
    """Basin-seen and basin-unseen bars side by side at positions x; vals: {regime: array}."""
    for r, dx in (('seen', -width / 2 - 0.025), ('unseen', width / 2 + 0.025)):
        ax.bar(x + dx, vals[r], width, color=face(r), edgecolor=REG_COLOR[r], lw=0.6, zorder=2)
        if labels:
            for xi, v in zip(x + dx, vals[r]):
                t = fmt.format(v)
                t = t[1:] if t.startswith('0.') else t
                ax.text(xi, v, t, ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5, color='black')


def family_bars(ax, vals, fmt=None, ref=None, top=None):
    """U-Net and ViT at x = 0 and 1, each with a basin-seen and a basin-unseen bar and its value on top.
    vals: {(family, regime): value}; top: upper limit of the value axis (default: 1.16 x the largest value).
    When the two bars of a family differ by less than one label line, the lower bar keeps its label and the label of
    the taller bar sits one line above it; the same holds for the neighbouring labels of U-Net and ViT."""
    w, gap = 0.38, 0.03
    top = top or max(vals.values()) * 1.16
    line = 0.07 * top                                          # height of one label line in data units
    lab = lambda v: (fmt or ('{:.3f}' if v < 1 else '{:.2f}' if v < 10 else '{:.1f}')).format(v)
    ys = {}
    for f in rc.FAMILIES:
        vs, vu = vals[(f, 'seen')], vals[(f, 'unseen')]
        ys[(f, 'seen')], ys[(f, 'unseen')] = vs, vu
        if abs(vs - vu) < line:
            hi = 'seen' if vs >= vu else 'unseen'
            ys[(f, hi)] = min(vs, vu) + line
    a, b, c = (rc.FAMILIES[0], 'unseen'), (rc.FAMILIES[1], 'seen'), (rc.FAMILIES[1], 'unseen')
    if abs(ys[a] - ys[b]) < line:                              # neighbours across the families
        ys[b] = ys[a] + line
        if abs(ys[c] - ys[b]) < line:
            ys[c] = ys[b] + line
    for x, f in enumerate(rc.FAMILIES):
        for r, dx in (('seen', -w / 2 - gap / 2), ('unseen', w / 2 + gap / 2)):
            v = vals[(f, r)]
            ax.bar(x + dx, v, w, color=face(r), edgecolor=REG_COLOR[r], lw=0.6, zorder=2)
            t = lab(v)
            ax.text(x + dx, ys[(f, r)], t[1:] if t.startswith('0.') else t, ha='center', va='bottom',
                    fontsize=ps.FONT_SIZE - 2.5, color='black')
    if ref is not None:
        ax.axhline(ref, color='#555555', lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks([0, 1], [rc.LABEL[f].replace('-T0', '') for f in rc.FAMILIES])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, top)
    rc.ygrid(ax)


def world_axes(fig, rect):
    """Robinson world map with the background of Figure 4.3, land left blank for an overlay."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = fig.add_axes(rect, projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -57, 84], crs=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False)
    ne = cfeature.NaturalEarthFeature
    ax.add_feature(ne('physical', 'ocean', '50m', facecolor='#D6E4F0', edgecolor='none'), zorder=0)
    ax.add_feature(ne('physical', 'coastline', '50m', facecolor='none', edgecolor='#A9B6C3', linewidth=0.3), zorder=3)
    ax.add_feature(ne('cultural', 'admin_0_boundary_lines_land', '50m', facecolor='none', edgecolor='#CBC6BD',
                      linewidth=0.3), zorder=3)
    return ax


def draw_basins(ax, dark=False):
    """HydroBASINS Level-5 basins of the test spatial blocks, filled with the regime colour (ch5_regime_basins.py);
    dark=True uses the darker pair, which stands out on the coloured climate classes."""
    import cartopy.crs as ccrs
    import geopandas as gpd
    g = gpd.read_file(D / 'rq2_basins.gpkg', layer='basins')
    for z, r in enumerate(REG):
        q = g[g.regime == r]
        ax.add_geometries(q.geometry, crs=ccrs.PlateCarree(), facecolor=REG_DARK[r] if dark else face(r),
                          edgecolor='black' if dark else REG_COLOR[r], linewidth=0.25, zorder=4 + z)
    return {r: int((g.regime == r).sum()) for r in REG}


def share_bars(ax, comp, keys, colors, fill=0.6):
    """100% stacked horizontal bars of the two regimes (basin-seen on top); shares of at least 6% printed."""
    for y, r in enumerate(REG):
        left = 0.0
        for k, c in zip(keys, colors):
            w = 100 * comp.loc[r, k]
            ax.barh(y, w, left=left, height=0.62, color=ps.alpha_on_white(c, fill), edgecolor='white', lw=0.5, zorder=2)
            if w >= 6:
                ax.text(left + w / 2, y, f'{w:.0f}', ha='center', va='center', fontsize=ps.FONT_SIZE - 2.5)
            left += w
    ax.set_yticks(range(2), [REG_LABEL[r] for r in REG], fontsize=ps.FONT_SIZE - 2)
    ax.set_xlim(0, 100)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 50, 100], ['0', '50', '100'], fontsize=ps.FONT_SIZE - 2.5)
    ax.tick_params(axis='both', length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.patch.set_alpha(0)


def class_labels(g, feat):
    """Class bounds as tick labels: '<b' for the first class, '≥a' for the last, 'a–b' between."""
    lo, hi = g.lower.to_numpy(), g.upper.to_numpy()
    fmt = (lambda v: f'{v:.1f}'.rstrip('0').rstrip('.').replace('-', '−')) if feat == 'ndvi' else (lambda v: f'{v:.0f}')
    out = []
    for i, (a, b) in enumerate(zip(lo, hi)):
        if not np.isfinite(a) or i == 0:
            out.append(f'<{fmt(b)}')
        elif not np.isfinite(b) or (feat == 'ndvi' and i == len(lo) - 1):
            out.append(f'≥{fmt(a)}')
        else:
            out.append(f'{fmt(a)}–{fmt(b)}')
    return out


def feature_classes(features, script_file, height):
    """Basin-seen and basin-unseen bars of POD and POFD within the classes of land surface characteristics
    (rq2_features.csv from ch5_regime_blocks.py), pooled over each class and averaged over seeds 1-3, one row per
    family and measure. features: (name, panel title, x-axis label). The U-Net and ViT
    rows of a measure share their vertical scale."""
    import matplotlib.pyplot as plt
    rc.style()
    d = pd.read_csv(D / 'rq2_features.csv')
    rows = [(m, f) for m in ('pod', 'pofd') for f in rc.FAMILIES]
    n = len(features)
    fig, axes = plt.subplots(len(rows), n, figsize=(ps.TEXT_WIDTH_IN, height), sharey='row', squeeze=False)
    from matplotlib.ticker import MultipleLocator
    hi = max(d.loc[d.feature == feat, f'{f}_pofd_{r}'].max() for feat, _, _ in features for f in rc.FAMILIES for r in REG)
    pofd_top = round(np.ceil((hi + 0.01) / 0.05) * 0.05, 2)
    ticks = {'pod': (1.0, 0.2, 0.1), 'pofd': (pofd_top, 0.05, None) if pofd_top <= 0.3 else (pofd_top, 0.1, 0.05)}
    for j, (feat, title, xlab) in enumerate(features):
        g = d[d.feature == feat].reset_index(drop=True)
        x = np.arange(len(g))
        for i, (m, f) in enumerate(rows):
            ax = axes[i, j]
            for r, dx in (('seen', -0.2), ('unseen', 0.2)):
                ax.bar(x + dx, g[f'{f}_{m}_{r}'], 0.38, color=face(r), edgecolor=REG_COLOR[r], lw=0.5, zorder=2)
            last = i == len(rows) - 1
            ax.set_xticks(x, class_labels(g, feat) if last else [''] * len(x), fontsize=ps.FONT_SIZE - 1)
            ax.tick_params(axis='x', length=0, pad=1.5)
            ax.set_xlim(-0.6, len(g) - 0.4)
            top, major, minor = ticks[m]
            ax.set_ylim(0, top)
            ax.yaxis.set_major_locator(MultipleLocator(major))
            if minor:
                ax.yaxis.set_minor_locator(MultipleLocator(minor))
            ax.yaxis.grid(True, which='both', color=ps.GREY_GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis='y', which='minor', length=2)
            if j == 0:
                ax.set_ylabel(f'{m.upper()}\n{rc.LABEL[f]}', fontsize=ps.FONT_SIZE - 0.5)
        axes[-1, j].set_xlabel(xlab, fontsize=ps.FONT_SIZE - 0.5)
        rc.panel_title(axes[0, j], f'({"abcd"[j]}) {title}')
    rc.yticks_everywhere(axes)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.6, w_pad=1.0)
    legend(fig, y=0.035)
    ps.save(fig, script_file, subdir=rc.section_dir(script_file))


def legend(fig, y=0.0):
    h = [Patch(facecolor=face(r), edgecolor=REG_COLOR[r], lw=0.6, label=REG_LABEL[r]) for r in REG]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, y), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, columnspacing=1.6)


def t0_means():
    f = pd.read_csv(D / 'rq2_regime_fits.csv')
    f = f[f.fit.str.contains('_t0_')].copy()
    f['family'] = f.fit.str.split('_').str[0]
    return f.groupby(['family', 'regime'])[['iou', 'pod', 'pofd', 'area_bias', 'block_mean_iou']].mean()
