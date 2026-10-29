"""Added Chapter 5 analyses (12 October 2026), all 38 fits, from saved outputs only.

Reads plot/data/tables/extra_patches_test.parquet (ch5_extra_build.py) and extra_fit_records.csv, the run records
(run_manifest.json, training_log.csv, metrics.csv, weather_check.json) and, for the repeated observations, every fit's
saved test probabilities. Prints every number used in the added text and writes the source tables to
plot/data/tables/extra_*.csv. Runs locally on one core.

Fit groups by the swap test (Section 4.7.6): weakly responding, mean |dp| < 1e-3; responding, mean |dp| >= 1e-2;
intermediate otherwise. Pairing: every fit with a record against the T0 of its own seed (ablations: seed 1).
Products: the 394 flood-positive test products; ground units: the 122 flood-positive units. Cluster intervals:
4,000 replicates of the 61 dependence clusters, the same draw for every fit (ch5_data.boot_weights).

    nice -n 19 python3 chapters/chapter5/plot/scripts/ch5_extra_analyses.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

T = Path(__file__).resolve().parents[1] / 'data/tables'
META = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata')
LABEL = {'t3': 'T3', 't7': 'T7', 't15': 'T15', 't30': 'T30', 'noup': 'T30 local', 'nosum': 'T30 daily',
         'sumonly': 'T30 statistics', 'scratch': 'T30 random init', 't0': 'T0'}
FAM = {'unet': 'U-Net', 'vit': 'ViT'}
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 400)


def name(key):
    f, c, s = key.split('_')
    return f'{FAM[f]}-{LABEL[c]} {s}'


def group(dp):
    return 'weak' if dp < 1e-3 else ('responding' if dp >= 1e-2 else 'intermediate')


def iou(t, f, n):
    d = t + f + n
    return np.divide(t, d, out=np.full(np.shape(d), np.nan, float), where=d > 0)


def agg(df, by, key):
    g = df.groupby(by)[[f'{key}_tp', f'{key}_fp', f'{key}_fn']].sum()
    return pd.Series(iou(*(g[c].to_numpy(float) for c in g.columns)), index=g.index)


def pairs(rec):
    """(fit, its T0) for every fit with a record."""
    out = []
    for r in rec.itertuples():
        if r.config == 't0':
            continue
        s = r.seed
        out.append((r.fit, f'{r.family}_t0_s{s}'))
    return out


def section(t):
    print(f'\n{"=" * 100}\n{t}\n{"=" * 100}')


def main():
    df = pd.read_parquet(T / 'extra_patches_test.parquet')
    rec = pd.read_csv(T / 'extra_fit_records.csv')
    rec['group'] = [group(x) if np.isfinite(x) else '' for x in rec.swap_dp.fillna(np.nan)]
    ts = cd.test_set()
    W = cd.boot_weights()
    PAIRS = pairs(rec)
    pos_prod = df.groupby('folder_name').n_flood.sum() > 0
    pos_unit = df.groupby('ground_unit').n_flood.sum() > 0
    df['activation'] = df.folder_name.str.split('_').str[0]
    prod_cluster = pd.Series(ts.product_cluster, index=ts.products)
    unit_cluster = pd.Series(ts.unit_cluster, index=ts.units)
    unit_regime = pd.Series(ts.unit_regime, index=ts.units)

    # ---------------------------------------------------------------- 12, 13: selection, thresholds, val vs test
    section('12-13. Selection, thresholds, validation against test')
    print(rec[['fit', 'group', 'swap_dp', 'threshold', 'sel_epoch', 'epochs_run', 'loss_epoch', 'sel_value', 'val_ep1',
               'val_pooled', 'test_pooled', 'val_unit', 'test_unit']].to_string(float_format=lambda x: f'{x:.4f}'))
    for f in ('unet', 'vit'):
        for c in ('t0', 't3', 't7', 't15', 't30'):
            r = rec[(rec.family == f) & (rec.config == c)]
            print(f'{f} {c}: sel epochs {list(r.sel_epoch)} of {list(r.epochs_run)}; thresholds {list(r.threshold)}; '
                  f'val ep1/sel {[round(a / b, 3) for a, b in zip(r.val_ep1, r.sel_value)]}')
    tmp = rec[rec.config != 't0'].copy()
    tmp['ep1_share'] = tmp.val_ep1 / tmp.sel_value
    print('temporal fits selected at epoch <= 2:', (tmp.sel_epoch <= 2).sum(), 'of', len(tmp),
          '| by family', tmp.groupby('family').apply(lambda g: f'{(g.sel_epoch <= 2).sum()}/{len(g)}, <=7: {(g.sel_epoch <= 7).sum()}').to_dict())
    print('val epoch-1 share of selected value, by family:', tmp.groupby('family').ep1_share.agg(['min', 'median', 'max']).round(3).to_dict('index'))
    # val vs test gap of T30 - T0
    for f in ('unet', 'vit'):
        for s in (1, 2, 3):
            a, b = rec.set_index('fit').loc[f'{f}_t30_s{s}'], rec.set_index('fit').loc[f'{f}_t0_s{s}']
            print(f'{f} s{s} T30-T0: val pooled {a.val_pooled - b.val_pooled:+.4f} test pooled {a.test_pooled - b.test_pooled:+.4f} '
                  f'| val unit {a.val_unit - b.val_unit:+.4f} test unit {a.test_unit - b.test_unit:+.4f}')
    r = rec.copy()
    print('rank corr val vs test pooled over 38 fits:', spearmanr(r.val_pooled, r.test_pooled)[0].round(3),
          '| unit:', spearmanr(r.val_unit, r.test_unit)[0].round(3))
    for f in ('unet', 'vit'):
        q = r[r.family == f]
        print(f, 'rank corr val/test pooled', spearmanr(q.val_pooled, q.test_pooled)[0].round(3), 'unit', spearmanr(q.val_unit, q.test_unit)[0].round(3),
              '| best on val', q.loc[q.val_pooled.idxmax(), 'fit'], 'best on test', q.loc[q.test_pooled.idxmax(), 'fit'])
    print('T0 thresholds', rec[rec.config == 't0'].groupby('family').threshold.agg(list).to_dict(),
          '| temporal thresholds', rec[rec.config != 't0'].groupby('family').threshold.agg(['min', 'median', 'max']).to_dict('index'))
    rec.to_csv(T / 'extra_fit_records.csv', index=False)

    # ---------------------------------------------------------------- 10: single-observation rows
    section('10. Single-observation test rows (sensitivity)')
    for f in ('unet', 'vit'):
        for s in (1, 2, 3):
            a, b = rec.set_index('fit').loc[f'{f}_t30_s{s}'], rec.set_index('fit').loc[f'{f}_t0_s{s}']
            print(f'{f} s{s} T30-T0 single: pooled {a.single_pooled - b.single_pooled:+.4f} unit {a.single_unit - b.single_unit:+.4f} '
                  f'prod {a.single_prod - b.single_prod:+.4f} | full: pooled {a.test_pooled - b.test_pooled:+.4f} unit {a.test_unit - b.test_unit:+.4f}')
    t0 = rec[rec.config == 't0'].set_index(['family', 'seed'])
    for k in ('single_pooled', 'single_unit', 'single_prod'):
        print(k, 'ViT-U-Net per seed', [round(t0.loc[('vit', s), k] - t0.loc[('unet', s), k], 4) for s in (1, 2, 3)],
              'means', round(t0.loc['unet'][k].mean(), 4), round(t0.loc['vit'][k].mean(), 4))
    for f in ('unet', 'vit'):
        for c in ('t0', 't3', 't7', 't15', 't30'):
            q = rec[(rec.family == f) & (rec.config == c)]
            print(f'{f} {c}: single pooled {q.single_pooled.mean():.4f}±{q.single_pooled.std():.4f} unit {q.single_unit.mean():.4f}±{q.single_unit.std():.4f}')
    sr = df.exp1_keep.astype(str).str.lower().isin(['true', '1'])
    print('single rows: patches', int(sr.sum()), 'products', df[sr].folder_name.nunique(), 'flood-positive products',
          int((df[sr].groupby('folder_name').n_flood.sum() > 0).sum()), 'units', df[sr].ground_unit.nunique())
    # paired cluster intervals, single rows, T30 - T0 per ground unit
    for f in ('unet', 'vit'):
        for s in (1, 2, 3):
            reps = []
            for key in (f'{f}_t30_s{s}', f'{f}_t0_s{s}'):
                c = pd.DataFrame({k: df[f'{key}_{k}'].to_numpy() for k in ('tp', 'fp', 'fn')}).assign(scored_cells=df.n_scored.to_numpy())
                u = cd.unit_values(c, 'single')
                v = u.iou.to_numpy().copy()
                v[u.flood_cells.to_numpy() == 0] = np.nan
                reps.append((np.nanmean(v), cd.boot_mean(v, ts.unit_cluster)))
            d = reps[0][1] - reps[1][1]
            print(f'{f} s{s} single unit T30-T0 {reps[0][0] - reps[1][0]:+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]')

    # ---------------------------------------------------------------- 1: groups by response; shifts in both groups
    section('1. Response groups and the shifts from T0')
    rows = []
    prod_change = {}
    for key, k0 in PAIRS:
        r = rec.set_index('fit').loc[key]
        fp1, fn1, fp0, fn0 = (df[f'{x}'].sum() for x in (f'{key}_fp', f'{key}_fn', f'{k0}_fp', f'{k0}_fn'))
        dry = df.n_flood == 0
        p1 = agg(df, 'folder_name', key)[pos_prod]
        p0 = agg(df, 'folder_name', k0)[pos_prod]
        prod_change[key] = (p1 - p0)
        rows.append({'fit': key, 'group': r.group, 'swap_dp': r.swap_dp, 'fp_change_pct': 100 * (fp1 / fp0 - 1),
                     'fn_change_pct': 100 * (fn1 / fn0 - 1),
                     'fp_dry_change_pct': 100 * (df.loc[dry, f'{key}_fp'].sum() / df.loc[dry, f'{k0}_fp'].sum() - 1),
                     'pooled_diff': r.test_pooled - rec.set_index('fit').loc[k0, 'test_pooled'],
                     'unit_diff': r.test_unit - rec.set_index('fit').loc[k0, 'test_unit'],
                     'prec_diff': df[f'{key}_tp'].sum() / (df[f'{key}_tp'].sum() + fp1) - df[f'{k0}_tp'].sum() / (df[f'{k0}_tp'].sum() + fp0),
                     'rec_diff': df[f'{key}_tp'].sum() / (df[f'{key}_tp'].sum() + fn1) - df[f'{k0}_tp'].sum() / (df[f'{k0}_tp'].sum() + fn0)})
    G = pd.DataFrame(rows)
    G.to_csv(T / 'extra_groups_shifts.csv', index=False)
    print(G.sort_values(['group', 'fit']).to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nby family x group: n, FP change, FN change (median, min, max)')
    print(G.assign(family=G.fit.str.split('_').str[0]).groupby(['family', 'group']).agg(
        n=('fit', 'size'), fp_med=('fp_change_pct', 'median'), fp_min=('fp_change_pct', 'min'), fp_max=('fp_change_pct', 'max'),
        fn_med=('fn_change_pct', 'median'), fn_min=('fn_change_pct', 'min'), fn_max=('fn_change_pct', 'max'),
        pooled_med=('pooled_diff', 'median'), unit_med=('unit_diff', 'median')).round(3).to_string())
    print('fits with fewer FP and more FN than T0:', int(((G.fp_change_pct < 0) & (G.fn_change_pct > 0)).sum()), 'of', len(G),
          G.assign(family=G.fit.str.split('_').str[0]).groupby('family').apply(lambda g: int(((g.fp_change_pct < 0) & (g.fn_change_pct > 0)).sum())).to_dict())
    print('configurations: group per seed')
    for f in ('unet', 'vit'):
        for c in ('t3', 't7', 't15', 't30'):
            q = rec[(rec.family == f) & (rec.config == c)]
            print(f'  {f} {c}: {list(q.group)} {[f"{x:.1e}" for x in q.swap_dp]}')
    # correlation of per-product changes between fits
    PC = pd.DataFrame(prod_change)
    C = PC.corr(method='spearman')
    grp = rec.set_index('fit').group
    for f in ('unet', 'vit'):
        ks = [k for k in PC.columns if k.startswith(f)]
        out = {}
        for a_ in ('weak', 'responding'):
            for b_ in ('weak', 'responding'):
                vals = [C.loc[x, y] for i, x in enumerate(ks) for y in ks[i + 1:] if {grp[x], grp[y]} == {a_, b_} and (a_ == b_ or grp[x] != grp[y])]
                if vals:
                    out[f'{a_}-{b_}'] = (len(vals), round(float(np.median(vals)), 3), round(float(np.min(vals)), 3), round(float(np.max(vals)), 3))
        print(f, 'Spearman of per-product IoU change between fits (n, median, min, max):', out)
    allv = [C.loc[x, y] for i, x in enumerate(PC.columns) for y in PC.columns[i + 1:] if x.split('_')[0] == y.split('_')[0]]
    print('all within-family fit pairs: median', round(float(np.median(allv)), 3), 'range', round(min(allv), 3), round(max(allv), 3))
    # the seed-3 U-Net T0 (lowest T0) and its children
    print('U-Net seed-3 temporal fits pooled diff:', G[G.fit.str.startswith('unet') & G.fit.str.endswith('s3')][['fit', 'pooled_diff', 'unit_diff']].round(4).values.tolist())

    # ---------------------------------------------------------------- 2: probability level (pattern D)
    section('2. Probability level against T0, by region and soil moisture')
    df['europe'] = np.where(df.continent == 'Europe', 'Europe', 'Other')
    nd = (df.n_scored - df.n_flood).astype(float)
    q = pd.qcut(df.SM1, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print('SM1 quartile bounds', np.round(np.nanpercentile(df.SM1, [0, 25, 50, 75, 100]), 3))
    rows = []
    for key in rec.fit:
        for reg in ('Europe', 'Other'):
            for qq in ('all', 'Q1', 'Q2', 'Q3', 'Q4'):
                m = (df.europe == reg) & ((q == qq) if qq != 'all' else True)
                rows.append({'fit': key, 'region': reg, 'sm_q': qq, 'p_dry': df.loc[m, f'{key}_spd'].sum() / nd[m].sum(),
                             'p_flood': df.loc[m, f'{key}_spf'].sum() / df.loc[m, 'n_flood'].sum()})
    P = pd.DataFrame(rows)
    P['t0'] = [f"{k.split('_')[0]}_t0_{k.split('_')[2]}" for k in P.fit]
    P = P.merge(P[['fit', 'region', 'sm_q', 'p_dry', 'p_flood']].rename(columns={'fit': 't0', 'p_dry': 'p_dry_t0', 'p_flood': 'p_flood_t0'}),
                on=['t0', 'region', 'sm_q'])
    P['ratio_dry'] = P.p_dry / P.p_dry_t0
    P['ratio_flood'] = P.p_flood / P.p_flood_t0
    P = P.merge(rec[['fit', 'group']], on='fit')
    P.to_csv(T / 'extra_probability_level.csv', index=False)
    a = P[(P.sm_q == 'all') & (P.fit.str.contains('_t0_') == False)]  # noqa: E712
    print(a.pivot_table(index='fit', columns='region', values=['ratio_dry', 'ratio_flood']).round(3).to_string())
    print('T0 mean p dry / flood by region:')
    print(P[(P.sm_q == 'all') & P.fit.str.contains('_t0_')].pivot_table(index='fit', columns='region', values=['p_dry', 'p_flood']).round(3).to_string())
    print('ratio_dry across SM quartiles (Q1..Q4), T30 fits:')
    for key in [k for k in rec.fit if '_t30_' in k]:
        for reg in ('Europe', 'Other'):
            v = P[(P.fit == key) & (P.region == reg) & (P.sm_q != 'all')].sort_values('sm_q')
            print(f'  {key} {reg}: dry {v.ratio_dry.round(2).tolist()} flood {v.ratio_flood.round(2).tolist()}')
    s = a.assign(family=a.fit.str.split('_').str[0]).groupby(['family', 'region']).agg(dry_min=('ratio_dry', 'min'), dry_med=('ratio_dry', 'median'),
                                                                                          dry_max=('ratio_dry', 'max'), fl_min=('ratio_flood', 'min'),
                                                                                          fl_med=('ratio_flood', 'median'), fl_max=('ratio_flood', 'max'))
    print(s.round(3).to_string())
    print('fits with dry ratio < 1 in both regions:', int(a.groupby('fit').ratio_dry.max().lt(1).sum()), 'of', a.fit.nunique())
    print('Europe dry ratio lower than Other in', int((a.pivot(index='fit', columns='region', values='ratio_dry').eval('Europe < Other')).sum()), 'fits')
    # flatness across SM quartiles: max/min of the dry ratio across quartiles
    fl = P[P.sm_q != 'all'].groupby(['fit', 'region']).ratio_dry.agg(lambda x: x.max() / x.min())
    print('dry-ratio spread across SM quartiles (max/min) median', round(float(fl.median()), 3), 'range', round(float(fl.min()), 3), round(float(fl.max()), 3))
    print('europe share of test products:', df.groupby('folder_name').europe.first().value_counts().to_dict(),
          '| basin-seen products in Europe:', df[df.basin_regime == 'seen'].groupby('folder_name').europe.first().value_counts().to_dict())

    # ---------------------------------------------------------------- 3: regimes at equal flooded fraction (pattern F)
    section('3. Basin regimes at equal flooded fraction (fit - T0 per ground unit)')
    uf = df.groupby('ground_unit')[['n_flood', 'n_scored']].sum()
    ff = uf.n_flood / uf.n_scored
    band = pd.cut(100 * ff, [-1, 1.5, 4, 101], labels=['<1.5%', '1.5-4%', '>=4%'], right=False)
    rows = []
    for key, k0 in PAIRS:
        d = (agg(df, 'ground_unit', key) - agg(df, 'ground_unit', k0))[pos_unit]
        for u, v in d.items():
            rows.append({'fit': key, 'ground_unit': u, 'regime': unit_regime[u], 'band': band[u], 'ff': ff[u], 'diff': v})
    F = pd.DataFrame(rows).merge(rec[['fit', 'group']], on='fit')
    F.to_csv(T / 'extra_regime_bands.csv', index=False)
    print('units per regime x band:', F[F.fit == PAIRS[0][0]].groupby(['regime', 'band'], observed=True).size().to_dict())
    S = F[F.regime.isin(['seen', 'unseen'])].groupby(['fit', 'regime', 'band'], observed=True)['diff'].agg(
        n='size', above=lambda x: int((x > 0).sum()), below=lambda x: int((x < 0).sum()), median='median').reset_index()
    print(S[S.fit.str.contains('_t30_')].to_string(float_format=lambda x: f'{x:+.4f}'))
    hi = F[F.regime.isin(['seen', 'unseen']) & (F.band != '<1.5%')]
    H = hi.groupby(['fit', 'regime'])['diff'].agg(n='size', above=lambda x: int((x > 0).sum()), below=lambda x: int((x < 0).sum()), median='median').reset_index()
    Hp = H.pivot(index='fit', columns='regime', values=['above', 'below', 'median', 'n'])
    print('\n>=1.5% flooded, per fit:')
    print(Hp.to_string(float_format=lambda x: f'{x:+.4f}'))
    for f in ('unet', 'vit'):
        h = H[H.fit.str.startswith(f)]
        for reg in ('unseen', 'seen'):
            x = h[h.regime == reg]
            print(f'{f} {reg} >=1.5%: n units {x.n.iloc[0]}; fits with more units above than below {int((x.above > x.below).sum())}/{len(x)}; '
                  f'median of medians {x["median"].median():+.4f}; medians range {x["median"].min():+.4f}..{x["median"].max():+.4f}')
    t0u = pd.DataFrame({k: agg(df, 'ground_unit', k) for k in rec.fit[rec.config == 't0']})
    print('units >4% flooded by regime:', pd.Series(unit_regime[ff[ff >= 0.04].index]).value_counts().to_dict(),
          '| 1.5-4%:', pd.Series(unit_regime[ff[(ff >= 0.015) & (ff < 0.04)].index]).value_counts().to_dict())
    # T0 unseen-seen within bands
    for f in ('unet', 'vit'):
        m0 = t0u[[c for c in t0u.columns if c.startswith(f)]].mean(axis=1)[pos_unit]
        for b in ('<1.5%', '1.5-4%', '>=4%'):
            sel = band[m0.index] == b
            ss = m0[sel & (unit_regime[m0.index] == 'seen').to_numpy()]
            uu = m0[sel & (unit_regime[m0.index] == 'unseen').to_numpy()]
            print(f'T0 {f} {b}: unseen median {uu.median():.4f} (n {len(uu)}) seen median {ss.median():.4f} (n {len(ss)}); means {uu.mean():.4f} / {ss.mean():.4f}')

    # ---------------------------------------------------------------- 8: T30 - T0 by regime
    section('8. T30 - T0 by basin regime, per ground unit and pooled, with cluster intervals')
    rows = []
    for f in ('unet', 'vit'):
        for c in ('t3', 't7', 't15', 't30'):
            for s in (1, 2, 3):
                key, k0 = f'{f}_{c}_s{s}', f'{f}_t0_s{s}'
                for reg in ('seen', 'unseen'):
                    reps, vals = [], []
                    for k in (key, k0):
                        u = agg(df, 'ground_unit', k)
                        v = pd.Series(np.nan, index=ts.units)
                        sel = pos_unit & (unit_regime == reg)
                        v[sel[sel].index] = u[sel[sel].index]
                        vals.append(np.nanmean(v.to_numpy()))
                        reps.append(cd.boot_mean(v.to_numpy(), ts.unit_cluster))
                    d = reps[0] - reps[1]
                    m = df.basin_regime == reg
                    pool = [df.loc[m, f'{k}_tp'].sum() / df.loc[m, [f'{k}_tp', f'{k}_fp', f'{k}_fn']].sum().sum() for k in (key, k0)]
                    rows.append({'fit': key, 'regime': reg, 'unit_diff': vals[0] - vals[1], 'lo': np.nanpercentile(d, 2.5),
                                 'hi': np.nanpercentile(d, 97.5), 'pooled_diff': pool[0] - pool[1]})
    R8 = pd.DataFrame(rows)
    R8.to_csv(T / 'extra_regime_change.csv', index=False)
    print(R8.to_string(float_format=lambda x: f'{x:+.4f}'))
    for f in ('unet', 'vit'):
        x = R8[R8.fit.str.startswith(f)].pivot(index='fit', columns='regime', values='unit_diff')
        print(f, 'temporal fits with unseen gain > seen gain:', int((x.unseen > x.seen).sum()), 'of', len(x),
              '| unseen > 0:', int((x.unseen > 0).sum()), '| seen > 0:', int((x.seen > 0).sum()))

    # ---------------------------------------------------------------- 5: climate and resolution (pattern G)
    section('5. T0 by climate and reference-map resolution, with prevalence')
    rows = []
    for var in ('climate', 'resolution_class', 'continent'):
        for cls, g in df.groupby(var):
            pp = g.groupby('folder_name').n_flood.sum()
            pp = pp[pp > 0].index
            r = {'variable': var, 'class': cls, 'products': len(pp), 'units': g[g.folder_name.isin(pp)].ground_unit.nunique(),
                 'flood_pct': 100 * g.n_flood.sum() / g.n_scored.sum()}
            for f in ('unet', 'vit'):
                vals = []
                for s in (1, 2, 3):
                    k = f'{f}_t0_s{s}'
                    vals.append(agg(g, 'folder_name', k)[pp].mean())
                r[f'{f}_prod_iou'] = np.mean(vals)
                uu = []
                for s in (1, 2, 3):
                    k = f'{f}_t0_s{s}'
                    gu = agg(g, 'ground_unit', k)
                    up = g.groupby('ground_unit').n_flood.sum()
                    uu.append(gu[up[up > 0].index].mean())
                r[f'{f}_unit_iou'] = np.mean(uu)
            # within prevalence: products >= 1.5% flooded
            pf = g.groupby('folder_name')[['n_flood', 'n_scored']].sum()
            hp = pf[(pf.n_flood / pf.n_scored) >= 0.015].index
            r['products_ge1.5'] = len(hp)
            for f in ('unet', 'vit'):
                r[f'{f}_prod_iou_ge1.5'] = np.mean([agg(g, 'folder_name', f'{f}_t0_s{s}')[hp].mean() for s in (1, 2, 3)]) if len(hp) else np.nan
            r['median_product_ff_pct'] = 100 * float((pf.loc[pp].n_flood / pf.loc[pp].n_scored).median())
            rows.append(r)
    C5 = pd.DataFrame(rows)
    C5.to_csv(T / 'extra_climate_resolution.csv', index=False)
    print(C5.to_string(float_format=lambda x: f'{x:.3f}'))

    # ---------------------------------------------------------------- 11: flood-free patches
    section('11. False alarms in flood-free patches')
    dry = df.n_flood == 0
    print('flood-free patches', int(dry.sum()), 'of', len(df), f'({100 * dry.mean():.1f}%), scored cells share {100 * df.loc[dry, "n_scored"].sum() / df.n_scored.sum():.1f}%')
    rows = []
    for key in rec.fit:
        fp, fpd = df[f'{key}_fp'].sum(), df.loc[dry, f'{key}_fp'].sum()
        rows.append({'fit': key, 'fp_total': fp, 'fp_dry': fpd, 'share_dry': fpd / fp,
                     'dry_cells_flagged_pct': 100 * fpd / df.loc[dry, 'n_scored'].sum(),
                     'dry_patches_with_fp_pct': 100 * (df.loc[dry, f'{key}_fp'] > 0).mean()})
    D = pd.DataFrame(rows).merge(rec[['fit', 'group']], on='fit')
    D['t0'] = [f"{k.split('_')[0]}_t0_{k.split('_')[2]}" for k in D.fit]
    D = D.merge(D[['fit', 'fp_dry', 'fp_total']].rename(columns={'fit': 't0', 'fp_dry': 'fp_dry_t0', 'fp_total': 'fp_total_t0'}), on='t0')
    D['fp_dry_change_pct'] = 100 * (D.fp_dry / D.fp_dry_t0 - 1)
    D['fp_wet_change_pct'] = 100 * ((D.fp_total - D.fp_dry) / (D.fp_total_t0 - D.fp_dry_t0) - 1)
    D.to_csv(T / 'extra_flood_free.csv', index=False)
    print(D[['fit', 'group', 'fp_total', 'fp_dry', 'share_dry', 'dry_cells_flagged_pct', 'dry_patches_with_fp_pct', 'fp_dry_change_pct', 'fp_wet_change_pct']].to_string(float_format=lambda x: f'{x:.3f}'))
    x = D[D.fit.str.contains('_t30_')]
    print('T30 dry change', x[['fit', 'fp_dry_change_pct', 'fp_wet_change_pct']].round(1).values.tolist())
    x = D[~D.fit.str.contains('_t0_')]
    print('temporal fits with fewer dry-patch FP than T0:', int((x.fp_dry_change_pct < 0).sum()), 'of', len(x),
          '| larger relative cut in dry than in flood-bearing patches:', int((x.fp_dry_change_pct < x.fp_wet_change_pct).sum()))

    # ---------------------------------------------------------------- 7: event-level view for every paired comparison
    section('7. Event-level view of every paired comparison')
    big = df.groupby('folder_name').n_flood.sum().sort_values(ascending=False).index[:10]
    act_pos = df.groupby('activation').n_flood.sum() > 0
    comps = []
    for f in ('unet', 'vit'):
        for s in (1, 2, 3):
            comps.append((f'{f}_t30_s{s}', f'{f}_t0_s{s}', f'{FAM[f]} T30-T0 s{s}'))
        for c in ('t3', 't7', 't15'):
            comps.append(([f'{f}_{c}_s{s}' for s in (1, 2, 3)], [f'{f}_t30_s{s}' for s in (1, 2, 3)], f'{FAM[f]} {LABEL[c]}-T30 (3-seed mean)'))
        comps.append(([f'{f}_t30_s{s}' for s in (1, 2, 3)], [f'{f}_t0_s{s}' for s in (1, 2, 3)], f'{FAM[f]} T30-T0 (3-seed mean)'))
        for v in ('noup', 'nosum', 'sumonly', 'scratch'):
            comps.append((f'{f}_{v}_s1', f'{f}_t30_s1', f'{FAM[f]} {LABEL[v]}-T30 s1'))
    rows = []
    clusters = np.unique(ts.cluster)
    for a_, b_, lab in comps:
        A = [a_] if isinstance(a_, str) else a_
        B = [b_] if isinstance(b_, str) else b_
        pa = pd.concat([agg(df, 'folder_name', k) for k in A], axis=1).mean(axis=1)[pos_prod]
        pb = pd.concat([agg(df, 'folder_name', k) for k in B], axis=1).mean(axis=1)[pos_prod]
        d = (pa - pb)
        aa = pd.concat([agg(df, 'activation', k) for k in A], axis=1).mean(axis=1)[act_pos]
        ab = pd.concat([agg(df, 'activation', k) for k in B], axis=1).mean(axis=1)[act_pos]
        da = aa - ab
        # pooled difference and leave-one-cluster-out
        cl = pd.Series(ts.cluster)

        def pooled(keys, mask=None):
            out = []
            for k in keys:
                t, f_, n = (df[f'{k}_{x}'].to_numpy(float) for x in ('tp', 'fp', 'fn'))
                if mask is not None:
                    t, f_, n = t * mask, f_ * mask, n * mask
                out.append(t.sum() / (t + f_ + n).sum())
            return np.mean(out)
        full = pooled(A) - pooled(B)
        loo = [pooled(A, (cl != c).to_numpy()) - pooled(B, (cl != c).to_numpy()) for c in clusters]
        rows.append({'comparison': lab, 'pooled_diff': full, 'prod_higher': int((d > 0).sum()), 'prod_lower': int((d < 0).sum()),
                     'prod_tied': int((d == 0).sum()), 'prod_median': d.median(), 'prod_q1': d.quantile(.25), 'prod_q3': d.quantile(.75),
                     'top10_mean': d[big].mean(), 'act_higher': int((da > 0).sum()), 'act_lower': int((da < 0).sum()),
                     'n_act': len(da), 'loo_min': min(loo), 'loo_max': max(loo)})
    E = pd.DataFrame(rows)
    E.to_csv(T / 'extra_event_level.csv', index=False)
    print(E.to_string(float_format=lambda x: f'{x:+.4f}'))
    print('ten largest products hold', round(100 * df[df.folder_name.isin(big)].n_flood.sum() / df.n_flood.sum(), 1), '% of flood cells')

    # ---------------------------------------------------------------- 4: repeated observations with identical T0 inputs (pattern H)
    section('4. Repeated observations: date pairs with identical T0 predictions')
    repeated(df, rec)

    # ---------------------------------------------------------------- 6: inputs without a network (pattern J)
    section('6. Input association with the flooded fraction (no network; review_unet_s1/data/fd_fig1_rho.csv)')
    J = pd.read_csv(Path(__file__).resolve().parents[2] / 'review_unet_s1/data/fd_fig1_rho.csv')
    print(J.pivot_table(index=['group', 'input'], columns='scale', values='rho').round(3).to_string())
    g = df.groupby('folder_name').P30
    tot = df.P30.var()
    within = (df.P30 - g.transform('mean')).var()
    print(f'30-day rain: within-product share of variance {100 * within / tot:.1f}%')

    # ---------------------------------------------------------------- 14: recorded time per epoch, all fits
    section('14. Recorded training time per epoch')
    rec['min_per_epoch'] = 60 * rec.hours / rec.epochs_run
    print(rec[['fit', 'host', 'gpu', 'epochs_run', 'hours', 'min_per_epoch']].to_string(float_format=lambda x: f'{x:.2f}'))
    b = rec[rec.host == 'bolzano']
    print(b.groupby(['family', 'config']).min_per_epoch.agg(['count', 'mean', 'min', 'max']).round(2).to_string())
    print('total recorded GPU hours', round(rec.hours.sum(), 1), 'by family', rec.groupby('family').hours.sum().round(1).to_dict())


def repeated(df, rec):
    """Successive dates of each test AOI on common patches; per fit the predicted flooded cells at its threshold."""
    pt = pd.read_csv(META / 'splits/patches.csv', usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_miny'])
    g = df[['folder_name', 'patch_number', 'aoi', 'mapping_date', 'P30', 'P7', 'SM1', 'activation']].merge(
        pt, on=['folder_name', 'patch_number'], how='left', validate='1:1')
    g['key'] = g.crs + '|' + g.bounds_minx.round(1).astype(str) + '|' + g.bounds_miny.round(1).astype(str)
    Y = np.load(cd.run_dir('unet', 't0', 1) / 'inference_final/preds/test__labels.i8.npy')
    plist = []
    for aoi, h in g.groupby('aoi'):
        dates = h.groupby('folder_name').mapping_date.first().sort_values()
        names = list(dates.index)
        for a, b in zip(names[:-1], names[1:]):
            ia = pd.Series(h.index[h.folder_name == a], index=h.key[h.folder_name == a].values)
            ib = pd.Series(h.index[h.folder_name == b], index=h.key[h.folder_name == b].values)
            ia = ia[~ia.index.duplicated()]
            ib = ib[~ib.index.duplicated()]
            common = ia.index.intersection(ib.index)
            if len(common):
                plist.append((aoi, a, b, (pd.Timestamp(dates[b]) - pd.Timestamp(dates[a])).days, ia[common].to_numpy(), ib[common].to_numpy()))
    print('pairs with common patches', len(plist), 'AOIs', len({p[0] for p in plist}))
    rows = {}
    base = []
    for aoi, a, b, gap, i1, i2 in plist:
        m = (Y[i1] >= 0) & (Y[i2] >= 0)
        o1, o2 = ((Y[i1] == 1) & m).sum(), ((Y[i2] == 1) & m).sum()
        base.append({'aoi': aoi, 'a': a, 'b': b, 'gap': gap, 'cells': int(m.sum()), 'dO': int(o2 - o1),
                     'dP30': float(g.P30.to_numpy()[i2].mean() - g.P30.to_numpy()[i1].mean()),
                     'dP7': float(g.P7.to_numpy()[i2].mean() - g.P7.to_numpy()[i1].mean()),
                     'dSM1': float(np.nanmean(g.SM1.to_numpy()[i2]) - np.nanmean(g.SM1.to_numpy()[i1])),
                     'activation': a.split('_')[0]})
    Bdf = pd.DataFrame(base)
    for key in rec.fit:
        d = cd.run_dir(*parse(key)) / 'inference_final'
        thr = np.float32(json.loads((d / 'threshold.json').read_text())['threshold'])
        P = np.load(d / 'preds/test__prob.f16.npy', mmap_mode='r')
        dA, mx = [], []
        for aoi, a, b, gap, i1, i2 in plist:
            m = (Y[i1] >= 0) & (Y[i2] >= 0)
            p1, p2 = np.asarray(P[i1], np.float32), np.asarray(P[i2], np.float32)
            dA.append(int(((p2 >= thr) & m).sum() - ((p1 >= thr) & m).sum()))
            mx.append(float(np.abs(p2 - p1)[m].max()) if m.any() else np.nan)
        rows[key] = (dA, mx)
    for key, (dA, mx) in rows.items():
        Bdf[f'{key}_dA'] = dA
        Bdf[f'{key}_maxdp'] = mx
    Bdf.to_csv(T / 'extra_date_pairs.csv', index=False)
    out = []
    for key in rec.fit:
        f, c, s = key.split('_')
        k0 = f'{f}_t0_{s}'
        same = Bdf[f'{k0}_maxdp'] == 0
        sub = Bdf[same]
        r = {'fit': key, 'identical_pairs': int(same.sum()), 'aois': sub.aoi.nunique(), 'activations': sub.activation.nunique(),
             'rho_dA_dP30': spearmanr(sub[f'{key}_dA'], sub.dP30)[0] if c != 't0' else np.nan,
             'rho_dA_dO': spearmanr(sub[f'{key}_dA'], sub.dO)[0] if c != 't0' else np.nan,
             'rho_dA_dO_all': spearmanr(Bdf[f'{key}_dA'], Bdf.dO)[0],
             'share_changed_identical': float((sub[f'{key}_dA'] != 0).mean())}
        out.append(r)
    O = pd.DataFrame(out).merge(rec[['fit', 'group', 'swap_dp']], on='fit')
    O.to_csv(T / 'extra_identical_pairs.csv', index=False)
    print(O.to_string(float_format=lambda x: f'{x:+.3f}'))
    same_any = (Bdf[[f'{f}_t0_s{s}_maxdp' for f in ('unet', 'vit') for s in (1, 2, 3)]] == 0)
    print('identical-T0 pairs per T0 fit:', same_any.sum().to_dict())
    allsame = same_any.all(axis=1)
    sub = Bdf[allsame]
    print('pairs identical for all six T0 fits:', int(allsame.sum()), 'AOIs', sub.aoi.nunique(), 'activations', sub.activation.nunique(),
          '| gap median', sub.gap.median(), '| rho(dO, dP30)', round(spearmanr(sub.dO, sub.dP30)[0], 3),
          'rho(dO, dP7)', round(spearmanr(sub.dO, sub.dP7)[0], 3), 'rho(dO, dSM1)', round(spearmanr(sub.dO, sub.dSM1, nan_policy='omit')[0], 3))
    print('all pairs: rho(dO, dP30)', round(spearmanr(Bdf.dO, Bdf.dP30)[0], 3), 'rho(dO,dP7)', round(spearmanr(Bdf.dO, Bdf.dP7)[0], 3),
          'rho(dO,dSM1)', round(spearmanr(Bdf.dO, Bdf.dSM1, nan_policy='omit')[0], 3), '| n', len(Bdf))
    o = O[O.fit.str.contains('_t0_') == False]  # noqa: E712
    for grp in ('weak', 'intermediate', 'responding'):
        x = o[o.group == grp]
        if len(x):
            print(f'{grp}: n {len(x)}; rho(dA,dP30) {x.rho_dA_dP30.min():+.2f}..{x.rho_dA_dP30.max():+.2f} (median {x.rho_dA_dP30.median():+.2f}); '
                  f'rho(dA,dO) {x.rho_dA_dO.min():+.2f}..{x.rho_dA_dO.max():+.2f}; changed share {x.share_changed_identical.min():.2f}..{x.share_changed_identical.max():.2f}')


def parse(key):
    f, c, s = key.split('_')
    s = int(s[1:])
    if c in ('t0', 't3', 't7', 't15', 't30'):
        return f, c, s, None
    return f, 't30', s, c


if __name__ == '__main__':
    main()
