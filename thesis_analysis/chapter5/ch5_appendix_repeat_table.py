"""Appendix A.6 table: every mapping date of the 16 appendix AOIs with at least four mapping products (Ravi, 30 Oct 2026:
the 27 page-size maps of these AOIs were unreadable; five maps stay (Figures A.6 to A.10) and the values of every date go into one table).

Per mapping product: seven-day upstream precipitation, labelled flooded share and, for U-Net-T0, U-Net-T30, ViT-T0 and
ViT-T30, the predicted flooded share and IoU of the map in which a cell is flood when at least two of the three seeds
predict it, on the mapped area of that mapping product. These are the values printed under the maps
(plot/data/repeat/map_metrics.csv, written by ch5_repeat_data.py). A dagger marks a date whose patches differ from those
of the previous date (map_manifest.csv, unsupported_transitions). A dash marks the five mapping products without a
precipitation value in product_inputs.csv, whose map headers are also blank. Writes the LaTeX rows and the Markdown rows to
plot/data/repeat/appendix_table.tex and appendix_table.md.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_appendix_repeat_table.py
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data' / 'repeat'
CONFIGS = [('unet', 't0'), ('unet', 't30'), ('vit', 't0'), ('vit', 't30')]
# place and period as in the captions of the former Figures A.15 to A.41; the figure numbers of the five kept maps (Figures A.6 to A.10)
AOIS = {'EMSR410_AOI01': ('Djibouti City, Djibouti', 'November 2019', 'Figure A.6'),
        'EMSR692_AOI02': ('Palamas, Greece', 'September 2023', ''),
        'EMSR692_AOI03': ('Larissa, Greece', 'September 2023', ''),
        'EMSR692_AOI04': ('Stefanovikio, Greece', 'September 2023', 'Figures A.7 and A.8'),
        'EMSR692_AOI05': ('Karditsa, Greece', 'September 2023', ''),
        'EMSR706_AOI07': ('Montreuil-sur-Mer, France', 'November 2023', ''),
        'EMSR708_AOI01': ('Westhoek, Belgium', 'November 2023', 'Figure A.9'),
        'EMSR788_AOI01': ('Ille-et-Vilaine, France', 'January–February 2025', ''),
        'EMSR788_AOI04': ('Guichen, France', 'January–February 2025', ''),
        'EMSR788_AOI06': ('Redon, France', 'January–February 2025', ''),
        'EMSR789_AOI01': ('Tosagua, Ecuador', 'February–March 2025', ''),
        'EMSR789_AOI02': ('Portoviejo, Ecuador', 'February–March 2025', ''),
        'EMSR841_AOI01': ('Gottne, Sweden', 'September 2025', ''),
        'EMSR848_AOI02': ('Tam Ky, central Viet Nam', 'October–November 2025', ''),
        'EMSR848_AOI03': ('Da Nang, central Viet Nam', 'October–November 2025', ''),
        'EMSR848_AOI04': ('Hue, central Viet Nam', 'October–November 2025', 'Figure A.10')}
MONTH = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()


def share(x):
    v = 100 * x
    return '<0.1' if 0 < v < 0.05 else f'{v:.1f}'


def main():
    m = pd.read_csv(D / 'map_metrics.csv').set_index(['product', 'family', 'config'])
    man = pd.read_csv(D / 'map_manifest.csv').set_index('aoi')
    rain = (pd.read_csv(D / 'product_inputs.csv').drop_duplicates('folder_name')
            .set_index('folder_name').precip_up_7)
    tex, md = [], []
    for aoi, (place, period, fig) in AOIS.items():
        head = f"{place}, {aoi.replace('_', ' ')}, {period}" + (f' ({fig})' if fig else '')
        tex.append(f'\\multicolumn{{11}}{{@{{}}l@{{}}}}{{\\textbf{{{head}}}}} \\\\*')
        md.append(f'| **{head}** |' + ' |' * 10)
        unsup = str(man.loc[aoi, 'unsupported_transitions'])
        moved = {s.split(' -> ')[1] for s in unsup.split(';') if ' -> ' in s}
        for p in man.loc[aoi, 'product_order'].split(';'):
            ymd = p.rsplit('_', 1)[1]
            date = f'{int(ymd[6:])} {MONTH[int(ymd[4:6]) - 1]}' + ('†' if p in moved else '')
            row = [date, f'{rain[p]:.0f}' if p in rain.index else '–', share(m.loc[(p, 'unet', 't0'), 'label_share'])]
            row += [share(m.loc[(p, f, c), 'predicted_share']) for f, c in CONFIGS]
            row += [f"{m.loc[(p, f, c), 'iou']:.2f}" for f, c in CONFIGS]
            tex.append(' & '.join(r.replace('<', '$<$') for r in row) + ' \\\\')
            md.append('| ' + ' | '.join(row) + ' |')
    (D / 'appendix_table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    (D / 'appendix_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(len(AOIS), 'AOIs,', sum(1 for r in md if not r.startswith('| **')), 'mapping products')


if __name__ == '__main__':
    main()
