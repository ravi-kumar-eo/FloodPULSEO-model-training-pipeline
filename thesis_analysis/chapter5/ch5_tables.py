"""Numbers for the tables and text of Chapter 5 (Tables 5.1, 5.2, 5.4, 5.5, 5.6 and the per-product views of RQ1).

Reads the saved evaluation outputs of every finished fit in experiments/runs_final (run_manifest.json,
weather_check.json, inference_final/metrics.csv, patches.csv, preds/) through ch5_data.py, which checks every
recomputed score against the evaluator's metrics.csv. Test subset, full rows, flood-only cells, each fit at its own
validation-selected global threshold (rule 'global'); the patch threshold head is rule 'head'. Fits that have not
finished are listed and skipped, so the script can be rerun as the remaining fits arrive.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_tables.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

import ch5_data as cd

ROWS = 'full'
META = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'tables'
CONFIGS = [  # (label, window, variant, seeds)
    ('T0', 't0', None, (1, 2, 3)),
    ('T3', 't3', None, (1, 2, 3)),
    ('T7', 't7', None, (1, 2, 3)),
    ('T15', 't15', None, (1, 2, 3)),
    ('T30', 't30', None, (1, 2, 3)),
    ('T30 local', 't30', 'noup', (1,)),
    ('T30 daily', 't30', 'nosum', (1,)),
    ('T30 statistics', 't30', 'sumonly', (1,)),
    ('T30 random init', 't30', 'scratch', (1,)),
]


def finished(f, w, v, s):
    try:
        cd.run_dir(f, w, s, v)
        return True
    except FileNotFoundError:
        return False


def metrics_row(f, w, v, s, rule='global', cohort='test'):
    m = pd.read_csv(cd.run_dir(f, w, s, v) / 'inference_final/metrics.csv')
    return m[(m.cohort == cohort) & (m.rows == ROWS) & (m.rule == rule)].iloc[0]


def table_5_1(avail):
    rows = []
    for f, label, w, v, s in avail:
        d = cd.run_dir(f, w, s, v)
        m = json.loads((d / 'run_manifest.json').read_text())
        sel = m['selection']
        start = ('random initialisation' if w == 't0' else
                 'random initialisation, T0 recipe' if v == 'scratch' else f'T0 of seed {s}')
        rows.append({'family': cd.FAMILY_LABEL[f], 'config': label, 'seed': s, 'training rows': m['rows'],
                     'start': start, 'selected epoch': sel['selected_epoch'], 'epochs run': sel['epochs_run'],
                     'host': m['environment'].get('compute', m['environment'].get('host')),
                     'hours': m.get('hours'), 'swap test': (d / 'weather_check.json').exists()})
    return pd.DataFrame(rows)


def table_5_2(avail):
    rows = []
    for f, label, w, v, s in avail:
        cd.check_against_metrics(f, w, s, v, rows=ROWS)
        r = metrics_row(f, w, v, s)
        rows.append({'family': cd.FAMILY_LABEL[f], 'config': label, 'seed': s, 'threshold': r.threshold,
                     'pooled IoU': r.pooled_iou, 'precision': r.pooled_precision, 'recall': r.pooled_recall,
                     'pooled AUPRC': r.pooled_auprc, 'product IoU': r.product_mean_iou,
                     'ground-unit IoU': r.ground_unit_mean_iou, 'product AP': r.product_mean_auprc,
                     'dry false-alarm share': r.dry_false_alarm_share, 'dry products': r.products_dry})
    per_seed = pd.DataFrame(rows)
    num = ['pooled IoU', 'precision', 'recall', 'pooled AUPRC', 'product IoU', 'ground-unit IoU', 'product AP',
           'dry false-alarm share']
    g = per_seed.groupby(['family', 'config'], sort=False)
    summary = g[num].mean().round(3)
    sd = g[num].std(ddof=1).round(3)
    summary['seeds'] = g.size()
    return per_seed, summary, sd


def table_5_5(avail):
    rows = []
    for f, label, w, v, s in avail:
        if w == 't0':
            continue
        d = cd.run_dir(f, w, s, v)
        if not (d / 'weather_check.json').exists():
            continue
        j = json.loads((d / 'weather_check.json').read_text())
        rows.append({'family': cd.FAMILY_LABEL[f], 'config': label, 'seed': s,
                     'mean |dp|': j['mean_abs_p_true_minus_swapped'], 'max |dp|': j['max_abs_p_true_minus_swapped'],
                     'probe IoU true': j['true_weather']['iou'], 'probe IoU swapped': j['swapped_weather']['iou'],
                     'probe IoU T0': j['t0_parent']['iou'], 'true - swapped': j['iou_gain_true_minus_swapped'],
                     'true - T0': j['iou_gain_true_minus_t0'], 'probe rows': j['probe_rows']})
    return pd.DataFrame(rows)


def table_5_6(avail):
    rows = []
    for f, label, w, v, s in avail:
        if label not in ('T0', 'T30'):
            continue
        g, h = metrics_row(f, w, v, s, 'global'), metrics_row(f, w, v, s, 'head')
        rows.append({'family': cd.FAMILY_LABEL[f], 'config': label, 'seed': s,
                     'pooled IoU global': g.pooled_iou, 'pooled IoU head': h.pooled_iou,
                     'product IoU global': g.product_mean_iou, 'product IoU head': h.product_mean_iou,
                     'ground-unit IoU global': g.ground_unit_mean_iou, 'ground-unit IoU head': h.ground_unit_mean_iou,
                     'dry FA global': g.dry_false_alarm_share, 'dry FA head': h.dry_false_alarm_share,
                     'precision global': g.pooled_precision, 'precision head': h.pooled_precision,
                     'recall global': g.pooled_recall, 'recall head': h.pooled_recall})
    return pd.DataFrame(rows)


def product_views():
    """RQ1 per mapping product: ViT − U-Net (T0, seeds averaged), wins, ties, the largest products, leave-one-cluster-out."""
    ts = cd.test_set()
    P = {}
    for f in cd.FAMILIES:
        vals = [cd.product_values(cd.counts(f, 't0', s), ROWS) for s in cd.SEEDS]
        P[f] = pd.concat([x.iou for x in vals], axis=1).mean(axis=1)
        P[f + '_seeds'] = pd.concat([x.iou for x in vals], axis=1)
        P['flood'] = vals[0].flood_cells
        P['cluster'] = vals[0].cluster
    pos = P['flood'] > 0
    diff = (P['vit'] - P['unet'])[pos]
    out = {'products': int(pos.sum()), 'vit_higher': int((diff > 0).sum()), 'unet_higher': int((diff < 0).sum()),
           'tied': int((diff == 0).sum()), 'median': float(diff.median()),
           'q25': float(diff.quantile(.25)), 'q75': float(diff.quantile(.75))}
    # seed-to-seed difference within a family, per product (median absolute difference over seed pairs)
    within = []
    for f in cd.FAMILIES:
        S = P[f + '_seeds'][pos].to_numpy()
        for a, b in ((0, 1), (0, 2), (1, 2)):
            within.append(np.abs(S[:, a] - S[:, b]))
    thr = float(np.median(np.concatenate(within)))
    out['median_seed_diff'] = thr
    out['beyond_seed_diff'] = int((np.abs(diff) > thr).sum())
    # ten largest products by flood cells: their share of the flood cells and their mean difference
    big = P['flood'][pos].sort_values(ascending=False).index[:10]
    out['top10_flood_share'] = float(P['flood'][big].sum() / P['flood'][pos].sum())
    out['top10_mean_diff'] = float(diff[big].mean())
    # leave-one-cluster-out range of the pooled ViT − U-Net difference (family means over seeds)
    tf = {f: np.mean([cd.counts(f, 't0', s)[['tp', 'fp', 'fn']].to_numpy(float) for s in cd.SEEDS], axis=0)
          for f in cd.FAMILIES}
    loco = []
    for c in range(ts.n_clusters):
        keep = ts.cluster != c
        iou = {f: tf[f][keep, 0].sum() / tf[f][keep].sum() for f in cd.FAMILIES}
        loco.append(iou['vit'] - iou['unet'])
    full = {f: tf[f][:, 0].sum() / tf[f].sum() for f in cd.FAMILIES}
    out['pooled_diff_seedmean_counts'] = full['vit'] - full['unet']
    out['loco_min'], out['loco_max'] = float(min(loco)), float(max(loco))
    return out


def conditions():
    """Table 5.4: T0 per-product IoU (seed mean) in classes of product attributes, with support counts."""
    ts = cd.test_set()
    terr = pd.read_csv(META / 'acquisition_terrain.csv').set_index('folder_name')
    vals = {}
    for f in cd.FAMILIES:
        v = [cd.product_values(cd.counts(f, 't0', s), ROWS) for s in cd.SEEDS]
        vals[f] = pd.concat([x.iou for x in v], axis=1).mean(axis=1)
        base = v[0]
    df = pd.DataFrame({'unet': vals['unet'], 'vit': vals['vit'], 'flood_cells': base.flood_cells,
                       'scored': base.scored_cells}, index=base.index)
    df = df[df.flood_cells > 0]
    df['flood_fraction'] = df.flood_cells / df.scored
    df['area_km2'] = df.flood_cells * 0.0064            # 80 m cells, flood-only, selected patches
    df = df.join(terr[['hand_med', 'elev_med']])
    unit = pd.Series(ts.units[ts.unit], index=ts.index.folder_name).groupby(level=0).first()
    df['ground_unit'] = unit.reindex(df.index)
    bounds = {'hand_med': [0, 2, 5, 10, np.inf], 'elev_med': [-np.inf, 50, 200, 500, np.inf],
              'flood_fraction': [0, .005, .015, .04, 1.0001], 'area_km2': [0, 1, 10, 100, np.inf]}
    out = []
    for var, b in bounds.items():
        cls = pd.cut(df[var], b, right=False)
        for c, g in df.groupby(cls, observed=False):
            out.append({'variable': var, 'class': str(c), 'products': len(g), 'ground units': g.ground_unit.nunique(),
                        'pooled flood fraction': g.flood_cells.sum() / g.scored.sum() if len(g) else np.nan,
                        'U-Net T0': g.unet.mean() if len(g) else np.nan, 'ViT T0': g.vit.mean() if len(g) else np.nan})
    q = df[['hand_med', 'elev_med', 'flood_fraction', 'area_km2']].quantile([.1, .25, .5, .75, .9]).round(4)
    return pd.DataFrame(out), q, df


def geography():
    """T0 per-product IoU by continent and main climate class, with support and flood prevalence."""
    ts = cd.test_set()
    meta = ts.index.groupby('folder_name').agg(continent=('continent', 'first'), climate=('climate', 'first'),
                                                ground_unit=('ground_unit', 'first'))
    vals = {}
    for f in cd.FAMILIES:
        v = [cd.product_values(cd.counts(f, 't0', s), ROWS) for s in cd.SEEDS]
        vals[f] = pd.concat([x.iou for x in v], axis=1).mean(axis=1)
        base = v[0]
    df = meta.join(pd.DataFrame({'flood_cells': base.flood_cells, 'scored_cells': base.scored_cells,
                                 'U-Net T0': vals['unet'], 'ViT T0': vals['vit']}))
    df = df[df.flood_cells > 0].copy()
    df['flood_fraction'] = df.flood_cells / df.scored_cells
    rows = []
    for variable in ('continent', 'climate'):
        for label, g in df.groupby(variable):
            rows.append({'variable': variable, 'class': label, 'products': len(g),
                         'ground units': g.ground_unit.nunique(),
                         'pooled flood fraction': g.flood_cells.sum() / g.scored_cells.sum(),
                         'median product flood fraction': g.flood_fraction.median(),
                         'U-Net T0': g['U-Net T0'].mean(), 'ViT T0': g['ViT T0'].mean()})
    return pd.DataFrame(rows), df


def main():
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 200)
    OUT.mkdir(parents=True, exist_ok=True)
    avail, missing = [], []
    for f in cd.FAMILIES:
        for label, w, v, seeds in CONFIGS:
            for s in seeds:
                (avail if finished(f, w, v, s) else missing).append((f, label, w, v, s))
    print(f'{len(avail)} fits finished; not finished: ' + (', '.join(f'{cd.FAMILY_LABEL[f]} {l} s{s}' for f, l, w, v, s in missing) or 'none'))
    t1 = table_5_1(avail)
    print('\n== Table 5.1 evaluated fits'); print(t1.to_string(index=False))
    per_seed, summ, sd = table_5_2(avail)
    print('\n== Table 5.2 per seed'); print(per_seed.round(4).to_string(index=False))
    print('\n== Table 5.2 mean over seeds'); print(summ.to_string())
    print('\n== Table 5.2 SD over seeds'); print(sd.to_string())
    t5 = table_5_5(avail)
    print('\n== Table 5.5 record sensitivity'); print(t5.to_string(index=False))
    t6 = table_5_6(avail)
    print('\n== Table 5.6 threshold rules per seed'); print(t6.round(4).to_string(index=False))
    print('\n== Table 5.6 means'); print(t6.groupby(['family', 'config']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
    t6_mean = t6.groupby(['family', 'config']).mean(numeric_only=True).drop(columns='seed').round(3)
    rq1 = product_views()
    print('\n== RQ1 per product'); print(json.dumps(rq1, indent=1))
    t4, q, t4_products = conditions()
    geo, geo_products = geography()
    print('\n== Table 5.4 quantiles of product attributes (flood-positive test products)'); print(q.to_string())
    print('\n== Table 5.4 classes'); print(t4.round(3).to_string(index=False))
    print('\n== Table 5.4 geography'); print(geo.round(3).to_string(index=False))
    t1.to_csv(OUT / 'table5_1_fit_inventory.csv', index=False)
    per_seed.to_csv(OUT / 'table5_2_per_seed.csv', index=False)
    summ.to_csv(OUT / 'table5_2_summary_mean.csv')
    sd.to_csv(OUT / 'table5_2_summary_sd.csv')
    t4.to_csv(OUT / 'table5_4_conditions.csv', index=False)
    q.to_csv(OUT / 'table5_4_condition_quantiles.csv')
    t4_products.to_csv(OUT / 'table5_4_product_source.csv')
    geo.to_csv(OUT / 'table5_4_geography.csv', index=False)
    geo_products.to_csv(OUT / 'table5_4_geography_product_source.csv')
    t5.to_csv(OUT / 'table5_5_record_sensitivity.csv', index=False)
    t6.to_csv(OUT / 'table5_6_threshold_rules_per_seed.csv', index=False)
    t6_mean.to_csv(OUT / 'table5_6_threshold_rules_mean.csv')
    (OUT / 'rq1_product_summary.json').write_text(json.dumps(rq1, indent=2) + '\n')
    print(f'\nSaved source tables to {OUT}')


if __name__ == '__main__':
    main()
