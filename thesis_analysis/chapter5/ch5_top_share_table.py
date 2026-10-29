"""Section 5.1.2 and 5.3.1 — IoU reached by the best 1, 2, 5, 10, 20, 30, 50 and 100% of the 394 flood-positive test mapping products
(Ravi, 20 Oct 2026: "top 5% of our flood events has IoU …, top 10% …"). Product IoU = mean of seeds 1-3 (rq3/products.csv);
for each share the lowest IoU among the best products ('at least') and their mean.

Output  plot/data/tables/top_share_iou.csv

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_top_share_table.py
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data'

if __name__ == '__main__':
    p = pd.read_csv(D / 'rq3/products.csv')
    p = p[p.flood > 0]
    p['cfg'] = p.family.map({'unet': 'U-Net-', 'vit': 'ViT-'}) + p.window.str.upper()
    m = p.groupby(['folder_name', 'cfg']).iou.mean().unstack()
    rows = []
    for q in (1, 2, 5, 10, 20, 30, 50, 100):
        k = int(round(len(m) * q / 100))
        r = {'share_pct': q, 'products': k}
        for c in ('U-Net-T0', 'ViT-T0', 'U-Net-T30', 'ViT-T30'):
            s = m[c].sort_values(ascending=False)
            r[c] = round(s.iloc[k - 1], 3)
            r[c + '_mean'] = round(s.iloc[:k].mean(), 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(D / 'tables/top_share_iou.csv', index=False)
    print(t.to_string(index=False))
