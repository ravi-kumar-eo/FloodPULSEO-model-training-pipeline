"""Section 5.4.4 case time series and AOI-level contrast counts."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'repeat'
SERIES_NAME = 'fig5_32_repeat_cases'
COUNTS_NAME = 'fig5_37_repeat_aoi_counts'
LENGTHS = ('t0', 't3', 't7', 't15', 't30')
# All five lengths (Ravi, 29 Oct 2026: T3 and T15 were missing, and at Osijek they are the extremes). Each length has
# its own hue, marker and line style, and a small sideways offset so that equal values stay separate markers.
SHOW_CONFIGS = LENGTHS
CONFIG_COLOR = {'t0': '#8C8C8C', 't3': '#1B9E77', 't7': '#5E3C99', 't15': '#A6761D', 't30': '#B2182B'}
LINE_STYLE = {'t0': (0, (4, 2)), 't3': '-', 't7': '-', 't15': '-', 't30': '-'}
MARKER = {'t0': 's', 't3': 'v', 't7': '^', 't15': 'P', 't30': 'D'}
DODGE = {'t0': -0.12, 't3': -0.06, 't7': 0.0, 't15': 0.06, 't30': 0.12}
PATCH_SOURCE = {'EMSR848_AOI03'}
CASES = (
    ('EMSR848_AOI03', 'Da Nang'),
    ('EMSR766_AOI01', 'Osijek'),
    ('EMSR692_AOI04', 'Stefanovikio'),
)
CONFIGS = ('t3', 't7', 't15', 't30')


def series_data(aoi):
    products = pd.read_csv(D / 'products.csv.gz')
    products = products[(products.aoi == aoi) & (products.kind == 'same_activation') &
                        products.config.isin(LENGTHS)]
    label = products.groupby('date').label_share.first() * 100
    prediction = (products.groupby(['family', 'config', 'date']).predicted_share.mean() * 100).sort_index()
    inputs = pd.read_csv(D / 'product_inputs.csv')
    rain = inputs[inputs.folder_name.isin(products.folder_name)].groupby('folder_name').precip_up_7.first()
    rain.index = rain.index.str[-8:].astype(int)
    dates = sorted(label.index)
    return dates, label, prediction, rain.reindex(dates)


def series_data_patches(aoi):
    """Seed means from the per-patch counts for the Da Nang products retained by the current analysis."""
    patches = pd.read_csv(D.parent / 'conditions/patches_test.csv.gz')
    patches = patches[patches.folder_name.str.match(aoi + '_')]
    counts = patches.groupby('folder_name').size()
    patches = patches[patches.folder_name.isin(counts[counts >= counts.max() / 2].index)]
    rows = {}
    for folder, group in patches.groupby('folder_name'):
        date = int(folder[-8:])
        scored = group.scored.sum()
        rows[date] = {'label': group.flood.sum() / scored * 100}
        for family in rc.FAMILIES:
            for config in LENGTHS:
                seed_shares = [
                    (group[f'{family}_{config}_s{seed}_tp'].sum() +
                     group[f'{family}_{config}_s{seed}_fp'].sum()) / scored
                    for seed in rc.SEEDS
                ]
                rows[date][(family, config)] = np.mean(seed_shares) * 100
    dates = sorted(rows)
    label = pd.Series({date: rows[date]['label'] for date in dates})
    prediction = pd.Series({
        (family, config, date): rows[date][(family, config)]
        for date in dates for family in rc.FAMILIES for config in LENGTHS
    }).sort_index()
    inputs = pd.read_csv(D / 'product_inputs.csv').drop_duplicates('folder_name')
    rain = inputs[inputs.folder_name.isin(patches.folder_name.unique())].set_index('folder_name').precip_up_7
    rain.index = rain.index.str[-8:].astype(int)
    return dates, label, prediction, rain.reindex(dates)


def case_data(aoi):
    return (series_data_patches if aoi in PATCH_SOURCE else series_data)(aoi)


def series_panel(ax, data, family, title):
    dates, label, prediction, rain = data
    x = np.arange(len(dates))
    ax.plot(x, label.reindex(dates).to_numpy(), color='black', lw=1.6,
            marker='o', ms=4.0, label='Flood label', zorder=5)
    for config in SHOW_CONFIGS:
        values = prediction.loc[(family, config)].reindex(dates).to_numpy()
        ax.plot(x + DODGE[config], values, color=CONFIG_COLOR[config], lw=1.3, ls=LINE_STYLE[config],
                marker=MARKER[config], ms=4.8, mec='white', mew=0.5, zorder=4)
    tick_labels = [
        f"{pd.Timestamp(str(date)).strftime('%-d %b')}\n{rain.loc[date]:.0f} mm"
        for date in dates
    ]
    ax.set_xticks(x, tick_labels)
    ax.set_xlim(-0.3, len(dates) - 0.7)
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc='left', pad=5)
    ps.ygrid(ax)
    return max(
        float(label.reindex(dates).max()),
        *(float(prediction.loc[(family, config)].reindex(dates).max()) for config in SHOW_CONFIGS),
    )


def save_full_width(fig, name):
    with mpl.rc_context({'savefig.bbox': None}):
        return ps.save(fig, Path(__file__).with_name(name + '.py'), subdir='5_4')


def series_figure():
    data = {aoi: case_data(aoi) for aoi, _ in CASES}
    fig, axes = plt.subplots(3, 2, figsize=(ps.TEXT_WIDTH_IN, 6.6), squeeze=False)
    letters = iter('abcdef')
    for row, (aoi, place) in enumerate(CASES):
        tops = []
        for col, family in enumerate(rc.FAMILIES):
            title = f'({next(letters)}) {place}, {rc.LABEL[family]}'
            tops.append(series_panel(axes[row, col], data[aoi], family, title))
        row_top = max(tops) * 1.10
        for col in range(2):
            axes[row, col].set_ylim(0, row_top)
        axes[row, 0].set_ylabel('Flooded share of the\nmapped area (%)')

    handles = [Line2D([], [], color='black', lw=1.6, marker='o', ms=4, label='Flood label')]
    handles.extend(
        Line2D([], [], color=CONFIG_COLOR[config], lw=1.3, ls=LINE_STYLE[config],
               marker=MARKER[config], ms=4.8, mec='white', mew=0.5, label=config.upper())
        for config in SHOW_CONFIGS
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.962, bottom=0.115, hspace=0.55, wspace=0.28)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.014), ncol=6,
               columnspacing=1.6, handlelength=2.0)
    return save_full_width(fig, SERIES_NAME)


def dax_figure():
    """Appendix Figure A.47: Dax (EMSR416 AOI02), drawn as the case figure of Section 5.4.4."""
    data = case_data('EMSR416_AOI02')
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.75), squeeze=False)
    tops = [series_panel(axes[0, col], data, family, f'({"ab"[col]}) Dax, {rc.LABEL[family]}')
            for col, family in enumerate(rc.FAMILIES)]
    for col in range(2):
        axes[0, col].set_ylim(0, max(tops) * 1.10)
    axes[0, 0].set_ylabel('Flooded share of the\nmapped area (%)')
    handles = [Line2D([], [], color='black', lw=1.6, marker='o', ms=4, label='Flood label')]
    handles.extend(
        Line2D([], [], color=CONFIG_COLOR[config], lw=1.3, ls=LINE_STYLE[config],
               marker=MARKER[config], ms=4.8, mec='white', mew=0.5, label=config.upper())
        for config in SHOW_CONFIGS
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.30, wspace=0.28)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=6,
               columnspacing=1.6, handlelength=2.0)
    with mpl.rc_context({'savefig.bbox': None}):
        return ps.save(fig, Path(__file__).with_name('figA_46_repeat_case_dax_series.py'), subdir='appendix')


def count_data():
    gains = pd.read_csv(D / 'cases/aoi_gains.csv')
    expected = np.select(
        [(gains['mean'] >= 0.01) & (gains['min'] > 0),
         (gains['mean'] <= -0.01) & (gains['max'] < 0)],
        ['helps', 'harms'], default='mixed',
    )
    assert np.array_equal(expected, gains.cls.to_numpy())
    counts = gains.groupby(['config', 'family', 'cls']).size()
    return gains, counts


def counts_figure():
    _, counts = count_data()
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 3.25), sharey=True)
    classes = ('helps', 'harms')
    titles = (
        '(a) AOIs with consistently higher\nIoU than T0',
        '(b) AOIs with consistently lower\nIoU than T0',
    )
    x = np.arange(len(CONFIGS))
    width = 0.31
    maximum = 0
    plotted = {}
    for ax, cls, title in zip(axes, classes, titles):
        plotted[cls] = {}
        for index, family in enumerate(rc.FAMILIES):
            values = np.array([int(counts.get((config, family, cls), 0)) for config in CONFIGS])
            plotted[cls][family] = values
            maximum = max(maximum, int(values.max()))
            xpos = x + (-0.18 if family == 'unet' else 0.18)
            bars = ax.bar(
                xpos, values, width=width, bottom=0,
                color=ps.alpha_on_white(rc.COLOR[family], ps.CLEAR_FILL),
                edgecolor=rc.COLOR[family], linewidth=0.8, zorder=3,
            )
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, str(value),
                        ha='center', va='bottom', fontsize=ps.FONT_SIZE - 0.5)
        ax.set_xticks(x, [config.upper() for config in CONFIGS])
        ax.set_xlim(-0.58, len(CONFIGS) - 0.42)
        ax.tick_params(axis='x', length=0)
        ax.set_title(title, loc='left', pad=5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ps.ygrid(ax)
    for ax in axes:
        ax.set_ylim(0, maximum + 2.0)
        ax.tick_params(axis='y', labelleft=True)
    axes[0].set_ylabel('Number of AOIs')
    handles = [
        Patch(facecolor=ps.alpha_on_white(rc.COLOR[family], ps.CLEAR_FILL),
              edgecolor=rc.COLOR[family], label=rc.LABEL[family])
        for family in rc.FAMILIES
    ]
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.22, wspace=0.25)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=2,
               columnspacing=1.7)
    out = save_full_width(fig, COUNTS_NAME)
    return out, plotted


def report_path(path):
    with Image.open(path) as image:
        print(f'{path}: {image.width} x {image.height} px')


def main():
    ps.apply_clear()
    series_path = series_figure()
    report_path(dax_figure())
    counts_path, plotted = counts_figure()
    report_path(series_path)
    report_path(counts_path)
    print('Figure 5.37 plotted AOI counts (T3, T7, T15, T30):')
    for cls, description in (('helps', 'consistently higher'), ('harms', 'consistently lower')):
        print(f'  {description}:')
        for family in rc.FAMILIES:
            values = ', '.join(f'{config.upper()} {value}' for config, value in zip(CONFIGS, plotted[cls][family]))
            print(f'    {rc.LABEL[family]}: {values}')


if __name__ == '__main__':
    main()
