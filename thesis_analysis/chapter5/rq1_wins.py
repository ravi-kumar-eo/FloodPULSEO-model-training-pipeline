"""Shared code of the RQ1 win-share figures (rq1_04 ... rq1_06), a macro view: for every flood-positive test mapping
product (394), whether ViT-T0 or U-Net-T0 has the higher IoU of the product (each family averaged over its three seeds;
equal = identical IoU, in practice both 0), grouped by a characteristic of the product. One horizontal bar per group:
share of products in which ViT is higher, equal, U-Net higher; numbers in the bars count products, labels give the
group size. Characteristics: plot/data/rq1/product_features.csv (rq1_features.py), the reference-map sensor
(released_events_metadata.csv, event_sensor; radar = Sentinel-1, COSMO-SkyMed, PAZ, ICEYE, RADARSAT, TerraSAR-X,
optical = all other sensors) and the basin regime of the test index.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402
import rq1_common as rc  # noqa: E402
from rq1_common import ps  # noqa: E402

TIE = '#B0B0B0'
CATALOGUE = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/metadata/released_events_metadata.csv')
RADAR = ('Sentinel-1', 'COSMO', 'PAZ', 'ICEYE', 'RADARSAT', 'TerraSAR')


def winners():
    pr = rc.products()
    m = pr.pivot_table(index='product', columns='family', values='iou', aggfunc='mean')
    d = m.vit - m.unet
    feat = pd.read_csv(rc.DATA / 'product_features.csv', index_col=0)
    out = feat.loc[d.index].copy()
    out['winner'] = np.where(d > 0, 'vit', np.where(d < 0, 'unet', 'equal'))
    sensor = pd.read_csv(CATALOGUE, usecols=['folder_name', 'event_sensor']).set_index('folder_name').event_sensor
    out['sensor_type'] = np.where(sensor.reindex(out.index).str.startswith(RADAR), 'radar', 'optical')
    out['regime'] = cd.test_set().index.groupby('folder_name').basin_regime.first().reindex(out.index)
    return out


def binned(series, edges, labels):
    return pd.cut(series, edges, labels=labels, right=False)


def panel(ax, groups, title, rows=None):
    """groups: list of (label, winner Series); rows: number of bar slots of the panel (equal bar thickness)."""
    for i, (lab, w) in enumerate(groups):
        n = len(w)
        left = 0.0
        for key, c in (('vit', rc.COLOR['vit']), ('equal', TIE), ('unet', rc.COLOR['unet'])):
            k = int((w == key).sum())
            face, edge = ps.soft(c, fill=0.45, edge=0.9)
            ax.barh(i, k / n, left=left, height=0.72, color=face, edgecolor=edge, lw=0.5)
            if k / n >= 0.07:
                ax.text(left + k / n / 2, i, str(k), ha='center', va='center', fontsize=ps.FONT_SIZE - 2,
                        color='black')
            left += k / n
    ax.set_yticks(range(len(groups)), [f'{lab}  ({len(w)})' for lab, w in groups], fontsize=ps.FONT_SIZE - 2)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim((rows or len(groups)) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f}%')
    ax.tick_params(axis='x', labelsize=ps.FONT_SIZE - 2)
    ax.spines['left'].set_visible(False)
    rc.panel_title(ax, title)


def legend(fig, y):
    h = [Patch(facecolor=ps.soft(c, 0.45, 0.9)[0], edgecolor=ps.soft(c, 0.45, 0.9)[1], lw=0.5, label=l)
         for c, l in ((rc.COLOR['vit'], 'ViT higher IoU'), (TIE, 'equal IoU'), (rc.COLOR['unet'], 'U-Net higher IoU'))]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, y), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, columnspacing=1.4)


def report(name, groups):
    for lab, w in groups:
        print(f'{name:28s} {lab:22s} n {len(w):3d}  ViT {int((w == "vit").sum()):3d}  equal {int((w == "equal").sum()):3d}  '
              f'U-Net {int((w == "unet").sum()):3d}  ViT share {np.mean(w == "vit"):.2f}')
