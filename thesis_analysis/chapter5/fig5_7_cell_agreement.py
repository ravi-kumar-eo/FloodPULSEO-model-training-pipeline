"""RQ1 — Agreement of U-Net-T0 and ViT-T0 of the same seed on the same test cells, for each seed.

(a) The reference flood cells: detected by both, by ViT-T0 only, by U-Net-T0 only, or by neither.
(b) The reference non-flood cells: correct (dry) in both, correct in ViT-T0 only (a false detection of U-Net-T0 alone),
    correct in U-Net-T0 only, or false detections of both. Shares of the cells of each class, in per cent; the axis of (b)
    starts at 80%, since 85.6-88.3% of the non-flood cells are correct in both.
From plot/data/rq1/agreement.csv (rq1_data.py); 2,769,452 flood and 46,499,674 non-flood scored test cells.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_7_cell_agreement.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

N_FLOOD, N_DRY = 2_769_452, 46_499_674
BOTH_RIGHT, BOTH_WRONG = '#E6E6E6', '#9E9E9E'


def main():
    rc.style()
    a = pd.read_csv(rc.DATA / 'agreement.csv').set_index('seed')
    flood = pd.DataFrame({'both': a.flood_both, 'vit': a.flood_vit_only, 'unet': a.flood_unet_only,
                          'neither': a.flood_neither}) / N_FLOOD * 100
    dry = pd.DataFrame({'both': N_DRY - a.fp_both - a.fp_vit_only - a.fp_unet_only, 'vit': a.fp_unet_only,
                        'unet': a.fp_vit_only, 'neither': a.fp_both}) / N_DRY * 100
    assert ((flood.sum(1) - 100).abs() < 1e-6).all() and ((dry.sum(1) - 100).abs() < 1e-6).all()
    face = {'both': BOTH_RIGHT, 'vit': ps.alpha_on_white(rc.COLOR['vit'], 0.55),
            'unet': ps.alpha_on_white(rc.COLOR['unet'], 0.55), 'neither': BOTH_WRONG}
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 1.75), sharey=True)
    for ax, df, title, xlim in zip(axes, (flood, dry), ('(a) Flood cells of the label', '(b) Dry cells of the label'), ((0, 100), (80, 100))):
        for i, s in enumerate(rc.SEEDS):
            left = 0.0
            for k in ('both', 'vit', 'unet', 'neither'):
                w = df.loc[s, k]
                ax.barh(i, w, left=left, height=0.62, color=face[k], lw=0, zorder=2)
                x0, x1 = max(left, xlim[0]), left + w
                if x1 - x0 >= (xlim[1] - xlim[0]) * 0.035:
                    ax.text((x0 + x1) / 2, i, f'{w:.1f}', ha='center', va='center', fontsize=ps.FONT_SIZE - 2,
                            color='black')
                left += w
        ax.set_yticks(range(3), [f'Seed {s}' for s in rc.SEEDS])
        ax.set_xlim(*xlim)
        ax.set_xticks(range(xlim[0], xlim[1] + 1, 20 if xlim[0] == 0 else 5))
        ax.set_xlabel('Share of cells (%)')
        ax.tick_params(axis='y', length=0)
        rc.panel_title(ax, title)
        print(title, '\n', df.round(1).to_string())
    h = [Patch(color=face['both'], label='Both correct'), Patch(color=face['vit'], label='ViT only correct'),
         Patch(color=face['unet'], label='U-Net only correct'), Patch(color=face['neither'], label='Both wrong')]
    axes[0].invert_yaxis()                            # seed 1 on top (shared y axis: invert once)
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, columnspacing=1.6)
    rc.yticks_everywhere(axes)
    fig.tight_layout(w_pad=1.5)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
