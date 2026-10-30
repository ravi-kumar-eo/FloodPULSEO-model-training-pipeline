"""IoU of the T0 networks per HydroBASINS Level-5 basin against the labelled flooded share of the basin's scored cells
(seed means; the basins and IoU of fig_basin_iou.py, data plot/data/conditions/basin_iou.csv). One marker per basin with
labelled flood, U-Net-T0 and ViT-T0, Spearman correlation in the legend. Full text width (Ravi, 25 Oct 2026).
    python fig_basin_iou_share.py <out dir>
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, '/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/scripts')
import rq1_common as rc
from rq1_common import ps

C5 = Path('/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data/conditions')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
G = pd.read_csv(C5 / 'basin_iou.csv')

rc.style()
W = ps.TEXT_WIDTH_IN
fig, ax = plt.subplots(figsize=(W, W * 0.58), layout='constrained')
for a, mk in (('unet_t0', 'o'), ('vit_t0', 's')):
    r = spearmanr(G.share, G[a]).correlation
    ax.scatter(100 * G.share, G[a], s=30, marker=mk, facecolor='none', edgecolor=rc.COLOR[a[:-3]], lw=1.0,
               label=f'{"U-Net-T0" if a == "unet_t0" else "ViT-T0"} (Spearman ρ = {r:.2f})')
ax.set_xscale('log')
ax.set_xticks([0.1, 1, 10, 100], ['0.1', '1', '10', '100'])
ax.set_xlabel('Labelled flooded share of the mapped area of the basin (%)')
ax.set_ylabel('IoU of the basin')
ax.set_ylim(bottom=-0.02)
rc.ygrid(ax)
ax.legend(frameon=False, loc='upper left')
fig.savefig(OUT / 'fig_basin_iou_share.png', dpi=300, facecolor='white'); plt.close(fig)
print(f'{len(G)} basins; Spearman', {a: round(spearmanr(G.share, G[a]).correlation, 2) for a in ('unet_t0', 'vit_t0')})
