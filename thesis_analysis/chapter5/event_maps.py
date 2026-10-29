"""Shared drawing of the event maps of Chapter 5 and Appendix A: reference and predicted flood maps of whole test
mapping products (events).

A product's test patches sit on its 2,560 m patch grid (experiments/metadata/patch_grid, splits/patches.csv); every
patch is drawn at its grid position, so the map covers the whole mapped area of the product. Grid positions without a
test patch lie outside the area that CEMS mapped (patch selection, Section 4.2.3) and are drawn white.
Prediction of a family and configuration: the cells predicted as flood by at least two of the three seeds, each seed at
its validation-selected threshold (preds_test_bits.npz), so that no single seed decides the map.
The first panel is the flood label of the product (flood and dry cells of the 80 m target, Section 4.1.1), titled with
the product code, AOI name, product type and mapping date (Ravi, 15 Oct 2026: never 'reference'). Categories of a
prediction: flood detected (labelled flood, predicted), missed detection (labelled flood, not predicted), false
detection (labelled dry, predicted), dry (labelled dry, not predicted), permanent water or mixed (excluded from scoring);
Ravi, 19 Oct 2026: "false detection" and "missed detection" replace "false alarm" and "flood missed".
Under every map (draw_product): the labelled flooded area and its share of the scored cells, and for every prediction
the evaluation metrics IoU (intersection over union), POD (probability of detection, the detected share of the labelled
flood) and POFD (probability of false detection, the share of the dry cells of the label predicted as flood) on the
product's scored cells (Ravi, 17 and 19 Oct 2026: IoU, POD and POFD only).
"""
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
META = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata')
COL = {'tp': '#0257CE', 'fn': '#C1408F', 'fp': '#EE8A00', 'dry': '#DCDCDC', 'pw': '#9FD5CB',
       'out': '#FFFFFF'}                      # darker and opaque (Ravi, 15 Oct 2026); Okabe-Ito-like, colour-blind safe
LEGEND = (('tp', 'Flood, detected'), ('fn', 'Missed detection'), ('fp', 'False detection'), ('dry', 'Dry, not predicted'),
          ('pw', 'Permanent water or mixed'), ('out', 'Outside the mapped area'))
# legend in two groups: the flood label (first panel) and the predictions (Ravi, 15 Oct 2026: blue in the label panel is
# flood of the label, not detected flood); None = group heading
LEGEND_GROUPS = ((None, 'Flood label'), ('tp', 'Flood'), ('dry', 'Dry'),
                 (None, 'Prediction'), ('tp', 'Flood, detected'), ('fn', 'Missed detection'), ('fp', 'False detection'),
                 ('dry', 'Dry, not predicted'),
                 (None, 'All maps'), ('pw', 'Permanent water or mixed,\nnot scored'), ('out', 'Outside the mapped area'))
KIND = {'DEL': 'delineation', 'DELINEATION': 'delineation', 'GRA': 'grading', 'GRADING': 'grading',
        'FEP': 'first estimate'}
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December')


def product_title(product):
    """(code and AOI name with product type, mapping date) of a mapping product folder name, e.g.
    'EMSR692_AOI03_Larissa_DEL_PRODUCT_20230910' -> ('EMSR692 AOI03 Larissa, delineation', '10 September 2023');
    monitoring products: 'delineation, monitoring 2'; old names such as 'EMSR238_01STAGNO_01DELINEATION_MAP_...'."""
    t = product.split('_')
    emsr, date = t[0], t[-1]
    if t[1].startswith('AOI'):
        aoi, rest = t[1], t[2:-1]
        kind_i = next(i for i, x in enumerate(rest) if x in KIND)
        name, kind, sub = ' '.join(rest[:kind_i]), KIND[rest[kind_i]], rest[kind_i + 1]
    else:
        m = re.match(r'(\d+)([A-Z]+)', t[1])
        aoi, name = f'AOI{m.group(1)}', m.group(2).title()
        kind, sub = KIND[re.sub(r'^\d+', '', t[2])], t[3]
    name = {'Ille et vilaine': 'Ille-et-Vilaine'}.get(name, name)
    if sub.startswith('MONIT'):
        kind += f', monitoring {int(sub[5:])}'
    d = f'{int(date[6:])} {MONTHS[int(date[4:6]) - 1]} {date[:4]}'
    return f"{emsr} {aoi}{' ' + name if name else ''}, {kind}", d


def label_title(product):
    code, d = product_title(product)
    return f'Flood label\n{code}\n{d}'


def legend_handles():
    """Grouped legend handles and the indices of the group headings (drawn bold, without a patch)."""
    handles, heads = [], []
    for i, (k, l) in enumerate(LEGEND_GROUPS):
        if k is None:
            handles.append(Patch(facecolor='none', edgecolor='none', label=l))
            heads.append(i)
        else:
            handles.append(Patch(facecolor=COL[k], edgecolor='#999999' if k in ('out', 'dry') else 'none', lw=0.5,
                                 label=l))
    return handles, heads


def bold_heads(leg, heads):
    for i in heads:
        leg.get_texts()[i].set_fontweight('bold')
        leg.legend_handles[i].set_visible(False)
LABEL = {'t0': 'T0', 't3': 'T3', 't7': 'T7', 't15': 'T15', 't30': 'T30'}


@lru_cache(maxsize=None)
def data():
    p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number', 'scored', 'flood'])
    p = p.sort_values('row_index').reset_index(drop=True)
    lab = np.load(C / 'cells_test.npz')['label']
    pt = pd.read_csv(META / 'splits/patches.csv', usecols=['folder_name', 'patch_number', 'bounds_minx', 'bounds_maxy'],
                     low_memory=False)
    pos = p[['folder_name', 'patch_number']].merge(pt, on=['folder_name', 'patch_number'], how='left', validate='1:1')
    grid = pd.read_csv(META / 'patch_grid/patch_grid_all_meta.csv').set_index('folder_name')
    return p, lab, pos, grid


@lru_cache(maxsize=None)
def prediction(family, window):
    """(N, 32, 32) bool: cells predicted as flood by at least two of the three seeds."""
    B = np.load(C / 'preds_test_bits.npz')
    shape = tuple(B['shape'])
    n = int(np.prod(shape))
    votes = sum(np.unpackbits(B[f'{family}_{window}_s{s}'])[:n].astype(np.uint8) for s in rc.SEEDS)
    return (votes >= 2).reshape(shape)


def layout(product):
    """Row indices of the product's test patches and their grid rows and columns (relative), and the map size."""
    p, _, pos, grid = data()
    rr = np.flatnonzero(p.folder_name.to_numpy() == product)
    if not len(rr):
        raise KeyError(f'{product}: no test patches')
    g = grid.loc[product]
    cols = np.rint((pos.bounds_minx.to_numpy()[rr] - g.origin_minx) / 2560).astype(int)
    rows = np.rint((g.origin_maxy - pos.bounds_maxy.to_numpy()[rr]) / 2560).astype(int)
    return rr, rows - rows.min(), cols - cols.min()


def mosaic(product, key):
    """RGB image of a product: key None = reference, else (family, window)."""
    _, lab, _, _ = data()
    rr, rows, cols = layout(product)
    H, W = (rows.max() + 1) * 32, (cols.max() + 1) * 32
    img = np.empty((H, W, 3))
    img[:] = to_rgb(COL['out'])
    q = None if key is None else prediction(*key)
    for k, r in enumerate(rr):
        y = lab[r]
        tile = np.empty((32, 32, 3))
        if q is None:
            tile[y == 1], tile[y == 0] = to_rgb(COL['tp']), to_rgb(COL['dry'])
        else:
            m = q[r]
            tile[(y == 0) & ~m], tile[(y == 0) & m] = to_rgb(COL['dry']), to_rgb(COL['fp'])
            tile[(y == 1) & m], tile[(y == 1) & ~m] = to_rgb(COL['tp']), to_rgb(COL['fn'])
        tile[y < 0] = to_rgb(COL['pw'])
        img[rows[k] * 32:(rows[k] + 1) * 32, cols[k] * 32:(cols[k] + 1) * 32] = tile
    return img


def scores(product, key):
    """IoU, predicted, detected, false-detection, missed and dry cells of the majority map on the product's scored cells."""
    _, lab, _, _ = data()
    rr, _, _ = layout(product)
    y, m = lab[rr], prediction(*key)[rr]
    tp, fp, fn = int((m & (y == 1)).sum()), int((m & (y == 0)).sum()), int((~m & (y == 1)).sum())
    dry = int((y == 0).sum())
    return dict(iou=tp / max(tp + fp + fn, 1), predicted=tp + fp, tp=tp, fp=fp, fn=fn, flood=tp + fn, dry=dry)


def size_km(product):
    _, rows, cols = layout(product)
    return (cols.max() + 1) * 2.56, (rows.max() + 1) * 2.56


def draw(events, columns, script_file, width=None, row_titles=None, col_titles=None, landscape=False,
         legend_rows=2, subdir=None):
    """One row per event, one column per map. events: product names; columns: None (reference) or (family, window).
    row_titles: text to the left of each row (e.g. '(a)'); col_titles: titles over the first row."""
    rc.style()
    imgs = [[mosaic(e, k) for k in columns] for e in events]
    ratios = [im[0].shape[0] / im[0].shape[1] for im in imgs]
    width = width or (9.0 if landscape else ps.TEXT_WIDTH_IN)
    left, right, wsp = 0.05, 0.995, 0.04
    panel_w = width * (right - left) / (len(columns) + (len(columns) - 1) * wsp)
    gap, top_pad, bottom_pad = 0.30, 0.25, 0.35 + 0.18 * legend_rows
    fig_h = panel_w * sum(ratios) + gap * (len(events) - 1) + top_pad + bottom_pad
    fig = plt.figure(figsize=(width, fig_h))
    gs = fig.add_gridspec(len(events), len(columns), height_ratios=ratios, wspace=wsp,
                          hspace=gap / (panel_w * float(np.mean(ratios))), left=left, right=right,
                          top=1 - top_pad / fig_h, bottom=bottom_pad / fig_h)
    for i, row in enumerate(imgs):
        for j, img in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(0.5)
            if i == 0 and col_titles:
                ax.set_title(col_titles[j], fontsize=ps.FONT_SIZE - 0.5, pad=3)
            if j == 0 and row_titles:
                ax.text(-0.04, 1.0, row_titles[i], transform=ax.transAxes, fontsize=ps.FONT_SIZE - 0.5,
                        ha='right', va='top')
    handles = [Patch(facecolor=COL[k], edgecolor='#999999' if k == 'out' else 'none', lw=0.5, label=l)
               for k, l in LEGEND]
    fig.legend(handles=handles, loc='lower center', ncol=int(np.ceil(len(handles) / legend_rows)), frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, handleheight=0.8, columnspacing=1.2,
               bbox_to_anchor=(0.52, 0.0))
    ps.save(fig, script_file, subdir=subdir or rc.section_dir(script_file))
    return fig


def draw_single(product, script_file, title=None, subdir=None, window='t30'):
    """One event as a 2 x 3 grid: Reference, U-Net-T0, U-Net-T<window> over legend, ViT-T0, ViT-T<window>
    (portrait text width, panels about 2 inches wide)."""
    rc.style()
    keys = [[None, ('unet', 't0'), ('unet', window)], ['legend', ('vit', 't0'), ('vit', window)]]
    titles = [['Reference', 'U-Net-T0', f'U-Net-{LABEL[window]}'], ['', 'ViT-T0', f'ViT-{LABEL[window]}']]
    ref = mosaic(product, None)
    ratio = ref.shape[0] / ref.shape[1]
    width = ps.TEXT_WIDTH_IN
    left, right, wsp = 0.01, 0.99, 0.05
    panel_w = width * (right - left) / (3 + 2 * wsp)
    top_pad, gap = 0.30, 0.30
    fig_h = 2 * panel_w * ratio + gap + top_pad + 0.08
    fig = plt.figure(figsize=(width, fig_h))
    gs = fig.add_gridspec(2, 3, wspace=wsp, hspace=gap / (panel_w * ratio), left=left, right=right,
                          top=1 - top_pad / fig_h, bottom=0.08 / fig_h)
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            if keys[i][j] == 'legend':
                ax.axis('off')
                handles = [Patch(facecolor=COL[k], edgecolor='#999999' if k == 'out' else 'none', lw=0.5, label=l)
                           for k, l in LEGEND]
                ax.legend(handles=handles, loc='center', frameon=False, fontsize=ps.FONT_SIZE - 1, handlelength=1.0,
                          handleheight=0.8)
                continue
            img = ref if keys[i][j] is None else mosaic(product, keys[i][j])
            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(0.5)
            ax.set_title(titles[i][j], fontsize=ps.FONT_SIZE - 0.5, pad=3)
    ps.save(fig, script_file, subdir=subdir or rc.section_dir(script_file))
    return fig


def measures(product, key):
    """Evaluation metrics of a majority map on the product's scored cells (text under the panels)."""
    c = scores(product, key)
    tp, fp, fn = c['tp'], c['fp'], c['fn']
    return dict(iou=tp / max(tp + fp + fn, 1), pod=tp / max(tp + fn, 1), pofd=fp / max(c['dry'], 1))


def label_text(product):
    _, lab, _, _ = data()
    rr, _, _ = layout(product)
    y = lab[rr]
    flood, scored = int((y == 1).sum()), int((y >= 0).sum())
    return f'Flooded area {flood * 0.0064:,.0f} km²\n{100 * flood / scored:.1f}% of the scored cells'


def prediction_label(product, key, two_lines=False):
    m = measures(product, key)
    sep = '\n' if two_lines else '   '
    return f"IoU {m['iou']:.2f}   POD {m['pod']:.2f}{sep}POFD {m['pofd']:.3f}"


def column_legend(fig, y=0.0):
    """The grouped legend with one group per column (flood label, prediction, all maps), padded with blank entries so
    that no group breaks across columns."""
    handles, heads = legend_handles()
    groups, cur = [], []
    for i, h in enumerate(handles):
        if i in heads and cur:
            groups.append(cur); cur = []
        cur.append((i, h))
    groups.append(cur)
    n = max(len(g) for g in groups)
    blank = Patch(facecolor='none', edgecolor='none', label=' ')
    out, bold = [], []
    for g in groups:
        for i, h in g:
            if i in heads:
                bold.append(len(out))
            out.append(h)
        out += [blank] * (n - len(g))
    leg = fig.legend(handles=out, loc='lower center', ncol=len(groups), frameon=False, fontsize=ps.FONT_SIZE - 0.5,
                     bbox_to_anchor=(0.5, y), handlelength=1.2, handleheight=0.9, columnspacing=2.0)
    for i in bold:
        leg.get_texts()[i].set_fontweight('bold')
        leg.legend_handles[i].set_visible(False)
    for i, h in enumerate(out):
        if h is blank:
            leg.legend_handles[i].set_visible(False)
    return leg


def draw_product(product, keys, titles, script_file, subdir=None, width=None, max_height=7.6):
    """One mapping product per figure, panels as large as the page allows (Ravi, 15 Oct 2026: one product per figure,
    big maps, evaluation metrics under every map). keys: None (the flood label) and (family, window) tuples, the flood
    label first; titles: panel titles of the predictions (the flood label panel is titled by label_title). Layouts
    with one, two or three columns are tried, the legend in the free slot after the flood label (or below), and the one
    with the larger panels within max_height (inches) is drawn."""
    rc.style()
    imgs = [mosaic(product, k) for k in keys]
    titles = [label_title(product) if k is None else t for k, t in zip(keys, titles)]
    ratio = imgs[0].shape[0] / imgs[0].shape[1]
    width = width or ps.TEXT_WIDTH_IN
    label_h, gap_w = 0.42, 0.12
    title_h = 0.06 + 0.17 * max(t.count('\n') + 1 for t in titles)
    best = None
    for ncol in (1, 2, 3):                              # one column suits wide, thin areas
        slots = [0, 'legend'] + list(range(1, len(keys))) if ncol == 2 else \
            ([0] + list(range(1, (len(keys) + 1) // 2)) + ['legend'] + list(range((len(keys) + 1) // 2, len(keys))))
        if ncol == 3 and len(keys) == 3:
            slots = [0, 1, 2]
        if ncol == 1:
            slots = list(range(len(keys)))
        nrow = -(-len(slots) // ncol)
        extra = 0.5 if 'legend' not in slots else 0.1
        pw = (width - gap_w * (ncol - 1) - 0.1) / ncol
        pw = min(pw, (max_height - extra - nrow * (label_h + title_h)) / (nrow * ratio))   # fit the page height
        h = nrow * (pw * ratio + label_h + title_h) + extra
        if best is None or pw > best[0]:
            best = (pw, ncol, nrow, slots, h)
    pw, ncol, nrow, slots, _ = best
    # title height of each row from the titles in it (the three-line flood-label title only in its own row)
    row_t = [0.06 + 0.17 * max([titles[s].count('\n') + 1 for s in slots[r * ncol:(r + 1) * ncol] if s != 'legend']
                               or [1]) for r in range(nrow)]
    extra = 0.5 if 'legend' not in slots else 0.1
    fig_h = sum(pw * ratio + label_h + h for h in row_t) + extra
    fig = plt.figure(figsize=(width, fig_h))
    top = fig_h - 0.05
    row_top = [top - sum(pw * ratio + label_h + h for h in row_t[:r]) for r in range(nrow)]
    x0 = (width - (ncol * pw + (ncol - 1) * gap_w)) / 2
    handles, heads = legend_handles()
    for n, slot in enumerate(slots):
        r, c = divmod(n, ncol)
        left = x0 + c * (pw + gap_w)
        bottom = row_top[r] - row_t[r] - pw * ratio
        ax = fig.add_axes([left / width, bottom / fig_h, pw / width, pw * ratio / fig_h])
        if slot == 'legend':
            ax.axis('off')
            leg = ax.legend(handles=handles, loc='center', frameon=False, fontsize=ps.FONT_SIZE - 0.5,
                            handlelength=1.2, handleheight=0.9, labelspacing=0.45)
            bold_heads(leg, heads)
            continue
        ax.imshow(imgs[slot], interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(0.6)
        ax.set_title(titles[slot], fontsize=ps.FONT_SIZE + 0.5, pad=4)
        text = label_text(product) if keys[slot] is None else prediction_label(product, keys[slot])
        ax.text(0.5, -0.03, text, transform=ax.transAxes, ha='center', va='top', fontsize=ps.FONT_SIZE - 0.5,
                linespacing=1.3)
    if 'legend' not in slots:
        leg = fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=ps.FONT_SIZE - 0.5,
                         bbox_to_anchor=(0.5, 0.0))
        bold_heads(leg, heads)
    ps.save(fig, script_file, subdir=subdir or rc.section_dir(script_file))
    return fig



def wrap_title(title, width=34):
    """Break a long line of a flood-label title after its comma (e.g. 'EMSR788 AOI06 Redon, delineation, monitoring 3')."""
    out = []
    for line in title.split('\n'):
        if len(line) > width and ', ' in line:
            a, b = line.split(', ', 1)
            out += [a + ',', b]
        else:
            out.append(line)
    return '\n'.join(out)


def draw_rows(products, keys, titles, script_file, subdir=None, width=None, max_height=8.4, row_labels=None):
    """Several mapping products in one figure, one product per row (Ravi, 20 Oct 2026: the Dax and Redon maps side by
    side, as on the pre-defence slide). keys and titles as in draw_product, the same for every row; row_labels such as
    ('(a)', '(b)') go in front of the flood-label title. Evaluation metrics under every map; one legend below."""
    rc.style()
    width = width or ps.TEXT_WIDTH_IN
    imgs = [[mosaic(p, k) for k in keys] for p in products]
    ratios = [im[0].shape[0] / im[0].shape[1] for im in imgs]
    ncol, gap_w = len(keys), 0.12
    label_h = 0.58 if ncol > 2 else 0.42                # two lines of text under the maps
    row_t = []
    for i, p in enumerate(products):
        t = [wrap_title(label_title(p)) if k is None else tt for k, tt in zip(keys, titles)]
        row_t.append(0.06 + 0.17 * max(x.count('\n') + 1 for x in t))
    legend_h = 1.05
    pw = (width - gap_w * (ncol - 1) - 0.1) / ncol
    pw = min(pw, (max_height - legend_h - sum(label_h + h for h in row_t)) / sum(ratios))
    fig_h = sum(pw * r + label_h + h for r, h in zip(ratios, row_t)) + legend_h
    fig = plt.figure(figsize=(width, fig_h))
    x0 = (width - (ncol * pw + (ncol - 1) * gap_w)) / 2
    top = fig_h - 0.05
    for i, p in enumerate(products):
        bottom = top - row_t[i] - pw * ratios[i]
        for j, k in enumerate(keys):
            ax = fig.add_axes([(x0 + j * (pw + gap_w)) / width, bottom / fig_h, pw / width, pw * ratios[i] / fig_h])
            ax.imshow(imgs[i][j], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(0.6)
            title = wrap_title(label_title(p)) if k is None else titles[j]
            if k is None and row_labels:
                title = f'{row_labels[i]} {title}'
            ax.set_title(title, fontsize=ps.FONT_SIZE + 0.5, pad=4)
            text = label_text(p) if k is None else prediction_label(p, k, two_lines=ncol > 2)
            ax.text(0.5, -0.03, text, transform=ax.transAxes, ha='center', va='top', fontsize=ps.FONT_SIZE - 0.5,
                    linespacing=1.3)
        top = bottom - label_h
    column_legend(fig)
    ps.save(fig, script_file, subdir=subdir or rc.section_dir(script_file))
    return fig

CONFIGS = [(f, w) for f in ('unet', 'vit') for w in ('t0', 't3', 't7', 't15', 't30')]


def metric_table(product, shown):
    """Markdown rows of the evaluation metrics of the majority maps of all ten configurations (U-Net and ViT at T0, T3, T7, T15
    and T30) on the product's scored cells; configurations in `shown` are marked '(map)' (Ravi, 15 Oct 2026: a table
    under every event map, so that the configurations not drawn have their numbers too)."""
    rows = ['| Configuration | IoU | POD | POFD |', '|---|---|---|---|']
    for k in CONFIGS:
        c = scores(product, k)
        m = measures(product, k)
        name = f"{'U-Net' if k[0] == 'unet' else 'ViT'}-{LABEL[k[1]]}" + (' (map)' if k in shown else '')
        rows.append(f"| {name} | {m['iou']:.2f} | {m['pod']:.2f} | {m['pofd']:.3f} |")
    return rows


ratio_table = metric_table          # former name, kept for older scripts
