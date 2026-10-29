"""Appendix Table A.6: response of every T30 fit and seed to the perturbed dynamic hydroclimatic variables
(perturb.csv, ch5_rq3_perturb_data.py), as a Markdown table written to data/rq3/table_a6.md and printed.

Columns: network, seed, perturbation, change of the predicted flood cells (%), of POD, of POFD and of micro IoU, the
share of the added or removed predicted cells that are flood in the reference (blank when fewer than 1,000 cells
change, where the share is not meaningful), and the mean absolute change of the flood probability over the scored cells.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_appendix_perturb_table.py
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data' / 'rq3'
NAME = {'p0': 'Zero', 'p05': 'Half', 'p2': 'Double', 'swap': 'Swapped'}
LABEL = {'unet': 'U-Net-T30', 'vit': 'ViT-T30'}


def z(x, fmt):
    s = fmt.format(x)
    return (s[1:] if float(s) == 0 and s[0] in '+-' else s).replace('-', '\u2212')   # true minus, as the other tables


def main():
    d = pd.read_csv(D / 'perturb.csv')
    d = d[d.seed.astype(str) != 'mean']
    lines = ['| Network | Seed | Precipitation | Predicted flood cells (%) | POD | POFD | Micro IoU | Flood share of the net change | Mean probability change |',
             '|---|---|---|---|---|---|---|---|---|']
    for f in ('unet', 'vit'):
        for s in sorted(d[d.family == f].seed.astype(int).unique()):
            for p in ('p0', 'p05', 'p2', 'swap'):
                r = d[(d.family == f) & (d.seed.astype(int) == s) & (d.perturbation == p)].iloc[0]
                n_changed = abs(r.pred_change_pct) / 100 * r.pred_none
                share = '' if pd.isna(r.changed_flood_share) or n_changed < 1000 else f'{r.changed_flood_share:.3f}'
                lines.append(f'| {LABEL[f]} | {s} | {NAME[p]} | {z(r.pred_change_pct, "{:+.2f}")} | '
                             f'{z(r.pod_change, "{:+.4f}")} | {z(r.pofd_change, "{:+.5f}")} | {z(r.iou_change, "{:+.4f}")} | '
                             f'{share} | {r.mean_abs_prob_change:.4f} |')
    md = '\n'.join(lines)
    (D / 'table_a6.md').write_text(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
