"""Section 5.4.4 compact native-grid maps for Quang Ngai and Osijek."""
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

import event_maps as em
from ch5_repeat_data import load_index, ratio
from rq1_common import COLOR, ps

BASE = Path(__file__).resolve().parents[1]
RAIN = (pd.read_csv(BASE / 'data/repeat/product_inputs.csv')
        .drop_duplicates('folder_name').set_index('folder_name').precip_up_7.to_dict())
FIGURES = (
    # T0 and T30 only (Ravi, 30 Oct 2026: the seven rows of the appendix figure were too small to read)
    ('EMSR848_AOI01', [None, ('unet', 't0'), ('unet', 't30'),
                       ('vit', 't0'), ('vit', 't30')],
     'fig5_35_repeat_quang_ngai'),
    ('EMSR766_AOI01', [None, ('unet', 't0'), ('unet', 't30'),
                       ('vit', 't0'), ('vit', 't30')],
     'fig5_39_repeat_case_osijek_clean'),
)


def tile_codes(label, prediction):
    """Outside, dry, flood/detected, missed, false detection, PW/mixed."""
    codes = np.full(label.shape, 5, np.uint8)
    codes[label == 0] = 1
    codes[label == 1] = 2
    if prediction is not None:
        codes[(label == 1) & ~prediction] = 3
        codes[(label == 0) & prediction] = 4
    return codes


def scored_extent(group, labels):
    """Union of native scored-cell bounds with a small crop margin."""
    product_index, row, col = np.where(labels[group.row_index.to_numpy()] >= 0)
    x = group.bounds_minx.to_numpy()[product_index] + col * 80
    y = group.bounds_maxy.to_numpy()[product_index] - row * 80
    xmin, xmax = x.min(), x.max() + 80
    ymin, ymax = y.min() - 80, y.max()
    margin = 0.012 * max(xmax - xmin, ymax - ymin)
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def product_code(product):
    match = re.search(r'_MONIT0*(\d+)_', product)
    if match:
        return f'M{int(match.group(1))}'
    for code in ('DEL', 'GRA', 'FEP'):
        if f'_{code}_PRODUCT_' in product:
            return code
    raise ValueError(f'Unrecognised product type: {product}')


def load_sequence(group, labels, packed, keys):
    products = group[['folder_name', 'date']].drop_duplicates().sort_values('date').folder_name.tolist()
    all_rows = group.row_index.to_numpy()
    predictions = {}
    for family, window in keys[1:]:
        votes = np.zeros((len(all_rows), 32, 32), np.uint8)
        for seed in (1, 2, 3):
            bits = packed[f'{family}_{window}_s{seed}'].reshape(-1, 128)[all_rows]
            votes += np.unpackbits(bits, axis=1).reshape(-1, 32, 32)
        predictions[(family, window)] = votes >= 2

    sequence = {}
    for product in products:
        take = np.flatnonzero(group.folder_name.to_numpy() == product)
        product_group = group.iloc[take]
        yy = labels[product_group.row_index.to_numpy()]
        origin_x, origin_y = product_group.bounds_minx.min(), product_group.bounds_maxy.max()
        columns = np.rint((product_group.bounds_minx.to_numpy() - origin_x) / 2560).astype(int)
        rows = np.rint((origin_y - product_group.bounds_maxy.to_numpy()) / 2560).astype(int)
        assert np.allclose(origin_x + columns * 2560, product_group.bounds_minx)
        assert np.allclose(origin_y - rows * 2560, product_group.bounds_maxy)
        height, width = (rows.max() + 1) * 32, (columns.max() + 1) * 32
        maps = {}
        for key in keys:
            prediction = None if key is None else predictions[key][take]
            tiles = tile_codes(yy, prediction)
            mosaic = np.zeros((height, width), np.uint8)
            for tile, row, column in zip(tiles, rows, columns):
                mosaic[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32] = tile

            scored = int((yy >= 0).sum())
            flooded = int((yy == 1).sum())
            if prediction is None:
                area = flooded * 0.0064
                share = 100 * flooded / scored
                annotation = f'{area:,.0f} km², {share:.1f}% flooded'
            else:
                true_positive = int((prediction & (yy == 1)).sum())
                false_positive = int((prediction & (yy == 0)).sum())
                false_negative = flooded - true_positive
                iou = ratio(true_positive, true_positive + false_positive + false_negative)
                area, share = em.area_share(true_positive + false_positive, scored)
                annotation = f'Predicted {area} km², {share}%\nIoU {iou:.2f}'
            maps[key] = mosaic, annotation

        extent = (origin_x, origin_x + width * 80, origin_y - height * 80, origin_y)
        date = pd.Timestamp(product[-8:]).strftime('%-d %b %Y')
        rain = RAIN[product]
        sequence[product] = maps, extent, product_code(product), date, rain
    return products, sequence


def row_label(key):
    if key is None:
        return 'Flood label', 'black'
    family, config = key
    family_name = 'U-Net' if family == 'unet' else 'ViT'
    return f'{family_name}-{config.upper()}', COLOR[family]


def draw(products, sequence, crop, keys, filename):
    width = ps.TEXT_WIDTH_IN
    plot_left, right, column_gap = 0.82, 0.05, 0.045
    map_width = (width - plot_left - right - (len(products) - 1) * column_gap) / len(products)

    xmin, xmax, ymin, ymax = crop
    rotated = ymax - ymin > xmax - xmin
    if rotated:
        xmin, xmax, ymin, ymax = ymin, ymax, -xmax, -xmin
    map_height = map_width * (ymax - ymin) / (xmax - xmin)
    header_height, annotation_height, legend_height = 0.62, 0.36, 0.48
    row_height = map_height + annotation_height
    height = header_height + len(keys) * row_height + legend_height
    fig = plt.figure(figsize=(width, height))

    palette = ListedColormap([em.COL[k] for k in ('out', 'dry', 'tp', 'fn', 'fp', 'pw')])
    top = height - header_height
    fig.text(0.08 / width, (height - 0.28) / height, 'N →' if rotated else 'N ↑',
             ha='left', va='center', fontsize=ps.FONT_SIZE - 1)

    for column, product in enumerate(products):
        x0 = plot_left + column * (map_width + column_gap)
        _, _, kind, date, rain = sequence[product]
        fig.text((x0 + map_width / 2) / width, (height - 0.10) / height,
                 f'{kind}, {date}', ha='center', va='top', fontsize=ps.FONT_SIZE - 0.5,
                 weight='bold')
        fig.text((x0 + map_width / 2) / width, (height - 0.30) / height,
                 f'7-day rain {rain:.0f} mm', ha='center', va='top', fontsize=ps.FONT_SIZE - 1.5)

    for row, key in enumerate(keys):
        map_bottom = top - row * row_height - map_height
        label, colour = row_label(key)
        fig.text((plot_left - 0.08) / width, (map_bottom + map_height / 2) / height,
                 label, ha='right', va='center', fontsize=ps.FONT_SIZE - 0.5,
                 weight='bold', color=colour)
        for column, product in enumerate(products):
            x0 = plot_left + column * (map_width + column_gap)
            maps, native_extent, _, _, _ = sequence[product]
            mosaic, annotation = maps[key]
            if rotated:
                mosaic = np.rot90(mosaic, -1)
                xlo, xhi, ylo, yhi = native_extent
                native_extent = (ylo, yhi, -xhi, -xlo)
            ax = fig.add_axes([x0 / width, map_bottom / height, map_width / width, map_height / height])
            ax.imshow(mosaic, cmap=palette, vmin=0, vmax=5, extent=native_extent,
                      interpolation='nearest')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal', adjustable='box')
            ax.set_facecolor(em.COL['out'])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('black')
            fig.text((x0 + map_width / 2) / width, (map_bottom - 0.035) / height,
                     annotation, ha='center', va='top', fontsize=ps.FONT_SIZE - 2)

    handles = [
        Patch(facecolor=em.COL[key], edgecolor='#777777', linewidth=0.3, label=label)
        for key, label in (
            ('tp', 'Flood / detected'), ('fn', 'Missed detection'), ('fp', 'False detection'),
            ('dry', 'Dry / correct dry'), ('pw', 'Permanent water / mixed'), ('out', 'Outside footprint'),
        )
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.008), ncol=3,
               fontsize=ps.FONT_SIZE - 1.5, columnspacing=1.0, handlelength=1.0,
               handletextpad=0.45, labelspacing=0.45)
    with mpl.rc_context({'savefig.bbox': None}):
        return ps.save(fig, Path(__file__).with_name(filename + '.py'), subdir='5_4')


def report_path(path):
    with Image.open(path) as image:
        print(f'{path}: {image.width} x {image.height} px')


def main():
    ps.apply_clear()
    index, _, _ = load_index()
    labels = np.load(em.C / 'cells_test.npz')['label']
    outputs = []
    with np.load(em.C / 'preds_test_bits.npz') as packed:
        for aoi, keys, filename in FIGURES:
            group = index[index.aoi == aoi]
            assert group.crs.nunique() == 1
            products, sequence = load_sequence(group, labels, packed, keys)
            outputs.append(draw(products, sequence, scored_extent(group, labels), keys, filename))
    for path in outputs:
        report_path(path)


if __name__ == '__main__':
    main()
