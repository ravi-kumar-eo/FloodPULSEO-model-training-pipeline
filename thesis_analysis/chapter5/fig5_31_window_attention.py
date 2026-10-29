"""Day-attention weights of T3, T7, T15 and T30 on all 51,606 test patches (26 Oct 2026; was a sample of 2,000).

Port of the attention calculation in Pre-Defence/ppt/figures/scripts/p_rq4_day_profiles.py.
Read-only source: experiments/runs_final/{final_window,final_t30}/seed*/<fit>/inference_attention_all/attention.npz.
The mean daily weights across patches and seeds are divided by their highest day.
Each panel has its own scale; grey shading marks days outside the input history.
U-Net uses temporal attention masks; ViT uses image-to-day cross-attention weights.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import rq1_common as rc
from rq1_common import ps

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
LENGTHS = (('t3', 3), ('t7', 7), ('t15', 15), ('t30', 30))


def attention(family, window, length):
    weights = []
    for seed in rc.SEEDS:
        directory = RUNS / ('final_t30' if window == 't30' else 'final_window') / f'seed{seed}'
        fit = family if window == 't30' else f'{family}_{window}'
        with np.load(directory / fit / 'inference_attention_all' / 'attention.npz') as probe:
            assert probe['att'].shape == (51606, length)
            att = probe['att'] / probe['att'].sum(1, keepdims=True)
            weights.append(att.mean(0) * length)
    mean = np.mean(weights, axis=0)
    return mean / mean.max()


def main():
    ps.apply_clear()
    fig, axes = plt.subplots(4, 2, figsize=(ps.TEXT_WIDTH_IN, 6.8))
    for row, (window, length) in enumerate(LENGTHS):
        for col, family in enumerate(rc.FAMILIES):
            ax = axes[row, col]
            weights = attention(family, window, length)
            span = max(1 - weights.min(), 0.001)
            low = weights.min() - 0.25 * span
            ax.bar(np.arange(-length, 0), weights - low, 0.8, bottom=low,
                   color=ps.alpha_on_white(rc.COLOR[family], ps.FILL_ALPHA),
                   edgecolor=rc.COLOR[family], lw=0.4, zorder=2)
            ax.set_ylim(low, 1 + 0.1 * (1 - low))
            ax.set_xlim(-30.7, -0.3)
            ax.set_xticks([-30, -25, -20, -15, -10, -5, -1])
            ax.tick_params(axis='x', labelbottom=row == 3)
            if length < 30:
                ax.axvspan(-30.7, -length - 0.5, color='#EEEEEE', zorder=0)
            if row == 3:
                ax.set_xlabel('Day before the mapping date')
            if col == 0:
                ax.set_ylabel('Relative to top day')
            ticks = MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(low, 1 + 0.1 * (1 - low))
            ax.set_yticks(ticks[(ticks >= low) & (ticks <= 1 + 0.1 * (1 - low))])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            rc.ygrid(ax)
            rc.panel_title(ax, f'({"abcdefgh"[2 * row + col]}) {rc.LABEL[family]}-T{length}')
            print(f'{rc.LABEL[family]}-T{length}: min/top={weights.min():.6f}, peak day={np.argmax(weights)-length}')
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 7.5, text.get_text()
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.x1 <= fig.bbox.width, text.get_text()
            assert box.y0 >= 0 and box.y1 <= fig.bbox.height, text.get_text()
    # Keep the saved width exactly equal to the thesis text width.
    with mpl.rc_context({'savefig.bbox': None}):
        ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
