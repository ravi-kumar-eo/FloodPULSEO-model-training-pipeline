"""C1b: repeated mapping products on verified, common scored 80 m cells.

Local, read-only access to source data; writes only plot/data/repeat/. Run with
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python .../ch5_repeat_data.py
No network, model inference, reprojection, or spatial interpolation is used.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
C = BASE / 'data/conditions'
OUT = BASE / 'data/repeat'
META = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata')
WINDOWS = ('t0', 't3', 't7', 't15', 't30')
NAMES = dict(t0='T0', t3='T3', t7='T7', t15='T15', t30='T30',
             noup='T30 local', nosum='T30 daily', sumonly='T30 statistics', scratch='T30 random init')
BOUNDS = ['crs', 'bounds_minx', 'bounds_miny', 'bounds_maxx', 'bounds_maxy']
EPS = 1e-4


def write(frame, name):
    frame.to_csv(OUT / name, index=False)


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def weighted_mean(x, w):
    ok = np.isfinite(x)
    return float(ratio(np.sum(np.asarray(x)[ok] * w[ok]), w[ok].sum()))


def changed(a, b):
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def load_index():
    p = pd.read_csv(C / 'patches_test.csv.gz').sort_values('row_index').reset_index(drop=True)
    assert np.array_equal(p.row_index, np.arange(len(p)))
    assert not p.duplicated(['folder_name', 'patch_number']).any()
    cb = np.load(C / 'clusters_boot.npz')
    p['cluster'] = cb['cluster']
    assert len(p) == len(cb['cluster'])
    p['aoi'] = p.folder_name.str.split('_').str[:2].str.join('_')
    p['activation'] = p.folder_name.str.split('_').str[0]
    p['date'] = pd.to_datetime(p.folder_name.str[-8:], format='%Y%m%d')
    bounds = pd.read_csv(META / 'splits/patches.csv', usecols=['folder_name', 'patch_number'] + BOUNDS)
    p = p.merge(bounds, on=['folder_name', 'patch_number'], how='left', validate='1:1')
    assert p[BOUNDS].notna().all().all()
    assert np.allclose(p.bounds_maxx - p.bounds_minx, 2560)
    assert np.allclose(p.bounds_maxy - p.bounds_miny, 2560)
    features = pd.read_csv(C / 'product_features.csv')
    totals = p.groupby('folder_name')[['scored', 'flood']].sum()
    z = features.set_index('folder_name').loc[totals.index]
    assert np.array_equal(totals.scored, z.scored_cells)
    assert np.array_equal(totals.flood, z.flood_cells)
    return p, cb['weights'].astype(float), features


def matched_support(g, products, label):
    """Exact geometry join: patch numbers identify source rows, never locations."""
    lookups = []
    for f in products:
        h = g[g.folder_name == f]
        assert not h.duplicated(BOUNDS).any(), f'duplicate footprint in {f}'
        lookups.append({tuple(v): int(i) for i, v in zip(h.row_index, h[BOUNDS].to_numpy())})
    common = sorted(set.intersection(*(set(v) for v in lookups)))
    rr = np.array([[v[k] for k in common] for v in lookups], dtype=int)
    mask = np.all(label[rr] >= 0, axis=0)
    status = 'verified' if mask.any() else ('no_common_scored_cells' if common else 'no_exact_bound_match')
    return rr, mask, status


def align(p, label):
    audit, pair_audit, groups, all_groups, masks, pair_masks, matches = [], [], [], [], {}, {}, []
    for aoi, g in p.groupby('aoi', sort=True):
        prods = g[['folder_name', 'date']].drop_duplicates().sort_values(['date', 'folder_name'])
        if len(prods) < 2:
            continue
        assert not prods.date.duplicated().any(), 'same-date products require a separate rule'
        products = prods.folder_name.tolist()
        assert g.cluster.nunique() == 1
        cluster = int(g.cluster.iloc[0])
        rr, mask, status = matched_support(g, products, label)
        if status == 'verified':
            all_groups.append(dict(aoi=aoi, support_id=aoi, support='all_dates', products=products,
                                   rows=rr, mask=mask, cluster=cluster, kind='same_activation', pair_ids=[]))
            masks[aoi + '_rows'] = rr
            masks[aoi + '_mask'] = np.packbits(mask)
            masks[aoi + '_shape'] = np.array(mask.shape)
        retained = 0
        for j, (fa, fb) in enumerate(zip(products[:-1], products[1:])):
            rows, m, reason = matched_support(g, [fa, fb], label)
            sid = f'{aoi}_pair{j:02d}'
            renumbered = int(np.count_nonzero(p.loc[rows[0], 'patch_number'].to_numpy() !=
                                               p.loc[rows[1], 'patch_number'].to_numpy()))
            pair_id = len(groups) if reason == 'verified' else None
            pair_audit.append(dict(aoi=aoi, consecutive_index=j, support_id=sid, pair_id=pair_id,
                                   cluster=cluster, product_a=fa, product_b=fb, status=reason,
                                   matched_patches=len(rows[0]), renumbered_patches=renumbered,
                                   common_scored_cells=int(m.sum())))
            if reason != 'verified':
                continue
            retained += 1
            groups.append(dict(aoi=aoi, support_id=sid, support='pairwise', products=[fa, fb],
                               rows=rows, mask=m, cluster=cluster, kind='same_activation'))
            pair_masks[sid + '_rows'] = rows
            pair_masks[sid + '_mask'] = np.packbits(m)
            pair_masks[sid + '_shape'] = np.array(m.shape)
            for ra, rb, n in zip(*rows, m.sum((1, 2))):
                matches.append(dict(support_id=sid, pair_id=pair_id, row_a=ra, row_b=rb,
                                    patch_number_a=int(p.loc[ra, 'patch_number']),
                                    patch_number_b=int(p.loc[rb, 'patch_number']), scored_cells=int(n),
                                    **p.loc[ra, BOUNDS].to_dict()))
        audit.append(dict(aoi=aoi, n_products=len(products), n_pairs=len(products)-1, cluster=cluster,
                          retained_pairs=retained, status='verified' if retained else 'no_usable_consecutive_pair',
                          all_date_status=status, common_bound_patches=len(rr[0]), common_scored_cells=int(mask.sum()),
                          first_date=str(prods.date.min().date()), last_date=str(prods.date.max().date())))
    audit = pd.DataFrame(audit)
    assert len(audit) == 84 and audit.n_products.sum() == 243
    write(audit, 'alignment.csv')
    write(pd.DataFrame(pair_audit), 'pair_alignment.csv')
    write(pd.DataFrame(matches), 'patch_matches.csv.gz')
    np.savez_compressed(OUT / 'common_cells.npz', **masks)
    np.savez_compressed(OUT / 'pair_cells.npz', **pair_masks)
    return groups, all_groups, audit


def cross_activation(p, label):
    """Descriptive exact-footprint matches only; never use ground_unit as an AOI."""
    cross = p[p.groupby(BOUNDS).activation.transform('nunique') > 1].copy()
    write(cross[['folder_name', 'aoi', 'activation', 'patch_number', 'ground_unit', 'cluster'] + BOUNDS],
          'cross_activation_patch_matches.csv')
    groups = []
    # In this dataset there is one exact-footprint site (Olmedo); make its identity auditable.
    for unit, g in cross.groupby('ground_unit'):
        products = g[['folder_name', 'date']].drop_duplicates().sort_values('date').folder_name.tolist()
        keys = [set(map(tuple, g[g.folder_name == f][BOUNDS].to_numpy())) for f in products]
        common = sorted(set.intersection(*keys))
        if not common:
            continue
        rr = []
        for f in products:
            h = g[g.folder_name == f]
            lookup = {tuple(v): int(i) for i, v in zip(h.row_index, h[BOUNDS].to_numpy())}
            rr.append([lookup[v] for v in common])
        rr = np.asarray(rr)
        mask = np.all(label[rr] >= 0, axis=0)
        assert g.cluster.nunique() == 1
        groups.append(dict(aoi='cross_' + unit, products=products, rows=rr, mask=mask,
                           cluster=int(g.cluster.iloc[0]), kind='cross_site', support='pairwise', support_id='cross_' + unit))
    return groups


def static_checks(groups):
    records = []
    with np.load(C / 'cells_test.npz') as z:
        for name in ('elevation_m', 'hand_m', 'uda_log10', 'flowdir', 'clay_pct', 'sand_pct'):
            x = z[name]  # one 53/106 MB feature at a time
            for g in groups:
                base = x[g['rows'][0]]
                differences = [changed(base, x[r]) for r in g['rows'][1:]]
                n = sum(int(d.sum()) for d in differences)
                g.setdefault('static_same_patch', np.ones(len(base), dtype=bool))
                for difference in differences:
                    g['static_same_patch'] &= ~difference.any((1, 2))
                scored_n = sum(int((d & g['mask']).sum()) for d in differences)
                records.append(dict(aoi=g['aoi'], support_id=g['support_id'], support=g['support'], feature=name, differing_cells=n, differing_scored_cells=scored_n))
            del x
    d = pd.DataFrame(records)
    write(d, 'static_input_checks.csv')
    return d


def product_inputs(groups, p):
    ante = pd.read_csv(BASE / 'data/rq3/patch_antecedent.csv').set_index(['folder_name', 'patch_number'])
    rows = []
    for g in groups:
        weights = g['mask'].sum((1, 2)).astype(float)
        for f, rr in zip(g['products'], g['rows']):
            z = np.load(META / 'final/upstream/weather' / (f + '.npz'))
            lookup = {int(v): i for i, v in enumerate(z['patch_number'])}
            nums = p.loc[rr, 'patch_number'].to_numpy()
            ii = np.array([lookup[int(v)] for v in nums])
            lp, up, sm = (z[k][ii].astype(float) for k in ('local_precip', 'upstream_precip', 'local_sm'))
            assert lp.shape[1] == 30
            assert pd.Timestamp(str(z['dates'][-1])) == pd.Timestamp(f[-8:]) - pd.Timedelta(days=1)
            old = ante.loc[[(f, int(v)) for v in nums]]
            api = np.nansum(up * 0.9 ** np.arange(29, -1, -1), axis=1)
            assert np.allclose(api, old['api_up_0.90'], rtol=1e-6, atol=1e-5)
            assert np.allclose(sm[:, -4], old.sm_before, equal_nan=True)
            d = dict(aoi=g['aoi'], support_id=g['support_id'], folder_name=f, cluster=g['cluster'],
                     sm_day4=weighted_mean(sm[:, -4], weights), sm_last=weighted_mean(sm[:, -1], weights),
                     sm_day4_coverage=float(ratio(weights[np.isfinite(sm[:, -4])].sum(), weights.sum())))
            for t in (3, 7, 15, 30):
                for name, arr in (('up', up), ('loc', lp)):
                    d[f'precip_{name}_{t}'] = weighted_mean(np.nansum(arr[:, -t:], axis=1), weights)
                    d[f'api_{name}_{t}'] = weighted_mean(np.nansum(arr[:, -t:] * 0.9 ** np.arange(t-1, -1, -1), axis=1), weights)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    d[f'sm_mean_{t}'] = weighted_mean(np.nanmean(sm[:, -t:], axis=1), weights)
            rows.append(d)
            z.close()
    d = pd.DataFrame(rows)
    write(d, 'product_inputs.csv')
    return d.drop(columns='aoi').set_index(['support_id', 'folder_name'])


def pair_reference(groups, label, inputs):
    optical = {}
    with np.load(C / 'cells_test.npz') as z:
        for feature in ('ndvi', 'ndbi'):
            x = z[feature]
            for g in groups:
                for j in range(len(g['products']) - 1):
                    a, b = x[g['rows'][j]], x[g['rows'][j+1]]
                    key = (g['support_id'], j)
                    q = optical.setdefault(key, {})
                    delta = b.astype(float) - a.astype(float)
                    m = g['mask']; valid = np.isfinite(delta) & m
                    q[feature + '_changed'] = changed(a, b)
                    q[feature + '_abs_delta'] = float(np.mean(np.abs(delta[valid]))) if valid.any() else np.nan
                    q[feature + '_delta'] = float(np.mean(delta[valid])) if valid.any() else np.nan
                    q[feature + '_valid_cells'] = int(valid.sum())
            del x
    rows = []
    for g in groups:
        g['pair_ids'] = []
        for j, (fa, fb) in enumerate(zip(g['products'][:-1], g['products'][1:])):
            kind = 'same_activation' if fa.split('_')[0] == fb.split('_')[0] else 'different_activation'
            if g['kind'] == 'cross_site' and kind == 'same_activation':
                g['pair_ids'].append(None)
                continue
            m = g['mask']; ya, yb = label[g['rows'][j]] == 1, label[g['rows'][j+1]] == 1
            opt = optical[(g['support_id'], j)]
            oc = opt.pop('ndvi_changed') | opt.pop('ndbi_changed')
            opt['static_changed_patches'] = int((~g['static_same_patch'] & m.any((1, 2))).sum())
            opt['optical_changed_cells'] = int((oc & m).sum())
            opt['optical_changed_patches'] = int((oc.any((1, 2)) & m.any((1, 2))).sum())
            opt['optical_same_patch_cells'] = int((m & ~oc.any((1, 2))[:, None, None]).sum())
            g.setdefault('optical_same_patch', []).append(~oc.any((1, 2)))
            pa, pb = inputs.loc[(g['support_id'], fa)], inputs.loc[(g['support_id'], fb)]
            d = dict(pair_id=len(rows), support_id=g['support_id'], aoi=g['aoi'], cluster=g['cluster'], kind=kind,
                     product_a=fa, product_b=fb, date_a=fa[-8:], date_b=fb[-8:],
                     days=int((pd.Timestamp(fb[-8:])-pd.Timestamp(fa[-8:])).days),
                     cells=int(m.sum()), label_a=int((ya & m).sum()), label_b=int((yb & m).sum()),
                     label_gain=int((~ya & yb & m).sum()), label_loss=int((ya & ~yb & m).sum()), **opt)
            for key in pa.index:
                if key == 'cluster':
                    continue
                d[key + '_a'], d[key + '_b'] = pa[key], pb[key]
                d['delta_' + key] = pb[key] - pa[key]
                if key.startswith(('precip_', 'api_')):
                    d['dlog1p_' + key] = np.log1p(pb[key]) - np.log1p(pa[key])
            d['dlog_label'] = np.log(d['label_b']/d['cells'] + EPS) - np.log(d['label_a']/d['cells'] + EPS)
            d['delta_label_share'] = (d['label_b'] - d['label_a']) / d['cells']
            d['label_sign'] = int(np.sign(d['label_b'] - d['label_a']))
            g['pair_ids'].append(len(rows))
            rows.append(d)
    d = pd.DataFrame(rows)
    write(d, 'pairs_reference.csv')
    return d


def fit_tables(groups, p, label, pairs):
    products, changes, within, sanity, validation = [], [], [], [], []
    with np.load(C / 'preds_test_bits.npz') as z:
        keys = [k for k in z.files if k != 'shape']
        assert len(keys) == 38
        assert tuple(z['shape']) == label.shape
        for fit in keys:
            fam, config, seed = fit.split('_'); seed = int(seed[1:])
            pred = np.unpackbits(z[fit], count=label.size).reshape(label.shape).astype(bool)
            # Recheck every patch against the cached evaluator counts; chunk to limit temporary memory.
            for start in range(0, len(p), 2048):
                sl = slice(start, start+2048); a, y = pred[sl], label[sl]
                for name, got in [('tp', (a & (y == 1)).sum((1, 2))),
                                  ('fp', (a & (y == 0)).sum((1, 2))),
                                  ('fn', (~a & (y == 1)).sum((1, 2)))]:
                    assert np.array_equal(got, p.iloc[sl][fit + '_' + name])
            validation.append(dict(fit=fit, all_patch_counts_match=True))
            ident = dict(fit=fit, family=fam, config=config, name=NAMES[config], seed=seed)
            for g in groups:
                m = g['mask']; nn = int(m.sum()); qs = []; ys = []
                for f, rr in zip(g['products'], g['rows']):
                    q, y = pred[rr], label[rr] == 1
                    tp = int((q & y & m).sum()); fp = int((q & ~y & m).sum()); fn = int((~q & y & m).sum())
                    nf, nq = tp + fn, tp + fp
                    qs.append(nq); ys.append(nf)
                    if g['support'] == 'all_dates' or g['kind'] == 'cross_site':
                        products.append(dict(**ident, aoi=g['aoi'], kind=g['kind'], cluster=g['cluster'],
                                         folder_name=f, date=f[-8:], cells=nn, label_cells=nf, predicted_cells=nq,
                                         label_share=nf/nn, predicted_share=nq/nn, tp=tp, fp=fp, fn=fn,
                                         iou=float(ratio(tp, tp+fp+fn)), pod=float(ratio(tp, nf)),
                                         pofd=float(ratio(fp, nn-nf))))
                if len(qs) >= 3 and g['support'] == 'all_dates':
                    within.append(dict(**ident, aoi=g['aoi'], cluster=g['cluster'], n_dates=len(qs),
                                       cells=nn, n_patches=len(m), rho=rho(np.asarray(ys), np.asarray(qs)),
                                       constant_label=len(set(ys)) == 1, constant_prediction=len(set(qs)) == 1))
                for j, pair_id in enumerate(g['pair_ids']):
                    if pair_id is None:
                        continue
                    a, b = pred[g['rows'][j]], pred[g['rows'][j+1]]
                    ya, yb = label[g['rows'][j]] == 1, label[g['rows'][j+1]] == 1
                    pg, pl = ~a & b & m, a & ~b & m
                    lg, ll = ~ya & yb & m, ya & ~yb & m
                    ng, nl = int(pg.sum()), int(pl.sum())
                    d = dict(**ident, pair_id=pair_id, aoi=g['aoi'], cluster=g['cluster'],
                             predicted_a=qs[j], predicted_b=qs[j+1], predicted_gain=ng, predicted_loss=nl,
                             gain_tp=int((pg & lg).sum()), loss_tp=int((pl & ll).sum()),
                             predicted_flip=ng+nl, predicted_union=int(((a | b) & m).sum()),
                             pred_sign=int(np.sign(qs[j+1]-qs[j])),
                             dlog_pred=np.log(qs[j+1]/nn + EPS)-np.log(qs[j]/nn + EPS),
                             delta_pred_share=(qs[j+1]-qs[j])/nn)
                    changes.append(d)
                    if config == 't0':
                        # Same-patch condition accounts for the spatial receptive field of T0.
                        opt_same = g['optical_same_patch'][j]
                        same = m & opt_same[:, None, None]
                        all_same = same & g['static_same_patch'][:, None, None]
                        sanity.append(dict(**ident, pair_id=pair_id, aoi=g['aoi'], cluster=g['cluster'],
                                           cells=nn, flip=ng+nl, same_optical_patch_cells=int(same.sum()),
                                           flip_same_optical_patch=int(((a != b) & same).sum()),
                                           same_static_optical_patch_cells=int(all_same.sum()),
                                           flip_same_static_optical_patch=int(((a != b) & all_same).sum())))
            del pred
            print('checked and measured', fit, flush=True)
    pr, ch, wh = pd.DataFrame(products), pd.DataFrame(changes), pd.DataFrame(within)
    ch = ch.merge(pairs.drop(columns=['aoi', 'cluster']), on='pair_id', validate='m:1')
    write(pr, 'products.csv.gz'); write(ch, 'pairs.csv.gz'); write(wh, 'within_aoi_spearman.csv')
    write(pd.DataFrame(sanity), 't0_sanity.csv'); write(pd.DataFrame(validation), 'prediction_validation.csv')
    return pr, ch, wh, pd.DataFrame(sanity)


class Resampler:
    """Saved thesis draws over all 61 clusters; then LOCO over represented clusters."""
    def __init__(self, weights, present):
        self.present = np.asarray(sorted(present), int)
        loco = np.ones((len(present), weights.shape[1]))
        loco[np.arange(len(present)), self.present] = 0
        self.w = np.vstack([np.ones(weights.shape[1]), weights, loco])
        self.nb = len(weights); self.nc = weights.shape[1]

    def total(self, v, cluster):
        return self.w @ np.bincount(cluster, weights=np.asarray(v, float), minlength=self.nc)

    def mean(self, v, cluster):
        v = np.asarray(v, float); ok = np.isfinite(v)
        return ratio(self.total(v[ok], cluster[ok]), self.total(np.ones(ok.sum()), cluster[ok]))

    def fraction(self, num, den, cluster):
        return ratio(self.total(num, cluster), self.total(den, cluster))

    def regression(self, x, y, cluster, corr=False):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y); x, y, cl = x[ok], y[ok], cluster[ok]
        n = self.total(np.ones(len(x)), cl)
        sx, sy = self.total(x, cl), self.total(y, cl)
        xx = self.total(x*x, cl)-ratio(sx*sx, n)
        xy = self.total(x*y, cl)-ratio(sx*sy, n)
        xx = np.where(xx > 1e-12, xx, np.nan)
        if not corr:
            return ratio(xy, xx)
        yy = self.total(y*y, cl)-ratio(sy*sy, n)
        return ratio(xy, np.sqrt(xx*np.maximum(yy, 0)))


def fit_metrics(pr, ch, wh, rs):
    cl = ch.cluster.to_numpy(); pc = pr.cluster.to_numpy(); wc = wh.cluster.to_numpy()
    one = np.ones(len(ch)); ls, ps = ch.label_sign.to_numpy(), ch.pred_sign.to_numpy()
    q = {}
    q['direction'] = rs.fraction((ls == ps) & (ls != 0), ls != 0, cl)
    q['direction_all_exact'] = rs.mean(ls == ps, cl)
    q['pred_tie_share'] = rs.mean(ps == 0, cl)
    q['direction_non_tied'] = rs.fraction((ls == ps) & (ls != 0), (ls != 0) & (ps != 0), cl)
    material_label = np.abs(ch.delta_label_share.to_numpy()) >= .001
    material_pred_sign = np.where(np.abs(ch.delta_pred_share.to_numpy()) >= .001, ps, 0)
    q['direction_0p1pp_deadband'] = rs.fraction((ls == material_pred_sign) & material_label, material_label, cl)
    aoi_direction = ch.assign(agreement=np.where(ls != 0, (ls == ps).astype(float), np.nan))
    aoi_direction = aoi_direction.groupby(['aoi', 'cluster']).agreement.mean().reset_index()
    q['direction_aoi_mean'] = rs.mean(aoi_direction.agreement, aoi_direction.cluster.to_numpy())
    for sign, title in [(1, 'expansion'), (-1, 'recession')]:
        q['direction_' + title] = rs.fraction((ls == ps) & (ls == sign), ls == sign, cl)
    q['elasticity'] = rs.regression(ch.dlog_label, ch.dlog_pred, cl)
    # No pseudocount sensitivity on strictly positive shares at both dates for label and prediction.
    ok = (ch[['label_a', 'label_b', 'predicted_a', 'predicted_b']] > 0).all(axis=1)
    a = ch.loc[ok]
    q['elasticity_positive_only'] = rs.regression(np.log(a.label_b/a.label_a), np.log(a.predicted_b/a.predicted_a), cl[ok])
    q['elasticity_half_cell'] = rs.regression(np.log((ch.label_b+.5)/(ch.label_a+.5)),
                                             np.log((ch.predicted_b+.5)/(ch.predicted_a+.5)), cl)
    q['delta_share_slope'] = rs.regression(ch.delta_label_share, ch.delta_pred_share, cl)
    for mode in ('gain', 'loss'):
        q[mode + '_pod'] = rs.fraction(ch[mode+'_tp'], ch['label_'+mode], cl)
        q[mode + '_precision'] = rs.fraction(ch[mode+'_tp'], ch['predicted_'+mode], cl)
    q['change_pod'] = rs.fraction(ch.gain_tp+ch.loss_tp, ch.label_gain+ch.label_loss, cl)
    q['change_precision'] = rs.fraction(ch.gain_tp+ch.loss_tp, ch.predicted_flip, cl)
    q['flip_share'] = rs.fraction(ch.predicted_flip, ch.cells, cl)
    q['flip_predicted_union'] = rs.fraction(ch.predicted_flip, ch.predicted_union, cl)
    q['label_flip_share'] = rs.fraction(ch.label_gain+ch.label_loss, ch.cells, cl)
    q['within_rho'] = rs.mean(wh.rho, wc)
    q['within_rho_constant_prediction_zero'] = rs.mean(
        np.where(wh.constant_prediction & ~wh.constant_label, 0, wh.rho), wc)
    q['label_share'] = rs.fraction(pr.label_cells, pr.cells, pc)
    q['predicted_share'] = rs.fraction(pr.predicted_cells, pr.cells, pc)
    q['optical_r_flip'] = rs.regression(ch.optical_changed_cells/ch.cells, ch.predicted_flip/ch.cells, cl, corr=True)
    q['optical_r_ndvi_abs_flip'] = rs.regression(ch.ndvi_abs_delta, ch.predicted_flip/ch.cells, cl, corr=True)
    q['optical_r_ndbi_abs_flip'] = rs.regression(ch.ndbi_abs_delta, ch.predicted_flip/ch.cells, cl, corr=True)
    for name, num, den in [('iou', pr.tp, pr.tp+pr.fp+pr.fn), ('pod', pr.tp, pr.tp+pr.fn),
                           ('pofd', pr.fp, pr.cells-pr.label_cells)]:
        q[name] = rs.fraction(num, den, pc)
        q['product_macro_' + name] = rs.mean(pr[name], pc)
    # One-predictor associations, not causal input effects. Same spatial support at both dates.
    config = ch.config.iloc[0]
    t = int(config[1:]) if config in WINDOWS and config != 't0' else 30
    variables = {f'{k}_window': 'dlog1p_' + k + '_' + str(t)
                 for k in ('api_up', 'api_loc', 'precip_up', 'precip_loc')}
    variables.update(api_up_fixed30='dlog1p_api_up_30', precip_up_fixed30='dlog1p_precip_up_30',
                     sm_mean_window=f'delta_sm_mean_{t}', sm_day4='delta_sm_day4', sm_last='delta_sm_last')
    for key, column in variables.items():
        x = ch[column].to_numpy(float)
        q['input_r_' + key] = rs.regression(x, ch.delta_pred_share, cl, corr=True)
        # Percentage-point response per observed SD of transformed input difference (fixed scaling).
        q['input_slope_' + key] = rs.regression(x, ch.delta_pred_share, cl) * np.nanstd(x) * 100
    return q


def summarise(pr, ch, wh, weights):
    all_summary, per_seed, contrast_rows, loco_rows, counts = [], [], [], [], []
    same = ch[ch.kind == 'same_activation']
    pair_ref = same.drop_duplicates('pair_id')
    scopes = {'same_activation': pair_ref.pair_id.tolist(),
              'optical_unchanged': pair_ref.loc[pair_ref.optical_changed_patches == 0, 'pair_id'].tolist(),
              'optical_changed': pair_ref.loc[pair_ref.optical_changed_patches > 0, 'pair_id'].tolist(),
              'static_unchanged': pair_ref.loc[pair_ref.static_changed_patches == 0, 'pair_id'].tolist(),
              'excluding_seveso_discrepancy': pair_ref.loc[pair_ref.aoi != 'EMSR843_AOI01', 'pair_id'].tolist()}
    cache = {}
    for scope, pair_ids in scopes.items():
        sub = same[same.pair_id.isin(pair_ids)]
        if sub.empty:
            continue
        represented = sorted(sub.cluster.unique()); rs = Resampler(weights, represented)
        pp = sub.drop_duplicates('pair_id')
        counts.append(dict(scope=scope, n_aois=pp.aoi.nunique(), n_pairs=len(pp), n_clusters=len(represented),
                           n_products=len(set(pp.product_a) | set(pp.product_b)),
                           label_ties=int((pp.label_sign == 0).sum()), label_expansions=int((pp.label_sign == 1).sum()),
                           label_recessions=int((pp.label_sign == -1).sum()),
                           n_cells_pairs=int(pp.cells.sum()), n_label_gain=int(pp.label_gain.sum()),
                           n_label_loss=int(pp.label_loss.sum()),
                           n_all_date_aois=pr.loc[(pr.kind == 'same_activation') & pr.aoi.isin(pp.aoi), 'aoi'].nunique(),
                           n_rho_eligible_aois=wh.loc[wh.aoi.isin(pp.aoi), 'aoi'].nunique(),
                           n_rho_eligible_clusters=wh.loc[wh.aoi.isin(pp.aoi), 'cluster'].nunique()))
        fit_metrics_cache = {}
        for fit, s in sub.groupby('fit'):
            ps = pr[(pr.fit == fit) & pr.aoi.isin(s.aoi) & (pr.kind == 'same_activation')]
            ws = wh[(wh.fit == fit) & wh.aoi.isin(s.aoi)]
            metrics = fit_metrics(ps, s, ws, rs)
            fit_metrics_cache[fit] = metrics
            for m, v in metrics.items():
                per_seed.append(dict(scope=scope, fit=fit, family=s.family.iloc[0], config=s.config.iloc[0],
                                     seed=int(s.seed.iloc[0]), metric=m, estimate=v[0], n_aois=s.aoi.nunique(),
                                     n_pairs=len(s), n_clusters=s.cluster.nunique(),
                                     defined_aoi_rho=int(ws.rho.notna().sum()) if m == 'within_rho' else np.nan,
                                     defined_rho_clusters=int(ws.loc[ws.rho.notna(), 'cluster'].nunique()) if m == 'within_rho' else np.nan,
                                     metric_n_pairs=int((s.label_sign != 0).sum()) if m == 'direction' else
                                     int((np.abs(s.delta_label_share) >= .001).sum()) if m == 'direction_0p1pp_deadband' else
                                     int((s[['label_a', 'label_b', 'predicted_a', 'predicted_b']] > 0).all(axis=1).sum())
                                     if m == 'elasticity_positive_only' else len(s)))
        for (family, config), s in sub.groupby(['family', 'config']):
            fits = sorted(s.fit.unique())
            for metric in fit_metrics_cache[fits[0]]:
                matrix = np.stack([fit_metrics_cache[f][metric] for f in fits])
                # Undefined seed-specific metrics remain undefined, not silently dropped.
                v = np.mean(matrix, axis=0)
                cache[(scope, family, config, metric)] = v
                lo, hi = interval(v[1:1+rs.nb])
                all_summary.append(dict(scope=scope, family=family, config=config, name=NAMES[config], metric=metric,
                                        estimate=v[0], lo=lo, hi=hi, n_seeds=len(fits), n_aois=s.aoi.nunique(),
                                        n_pairs=len(s)//len(fits), n_clusters=s.cluster.nunique(),
                                        valid_boot=int(np.isfinite(v[1:1+rs.nb]).sum()),
                                        loco_min=nanmin(v[1+rs.nb:]), loco_max=nanmax(v[1+rs.nb:]),
                                        seed_min=nanmin(matrix[:, 0]), seed_max=nanmax(matrix[:, 0])))
        for family in ('unet', 'vit'):
            comparisons = [(w, 't0') for w in WINDOWS[1:]] + [('t7', 't3'), ('t15', 't7'), ('t30', 't15'), ('t30', 't7')]
            comparisons += [(v, 't30') for v in ('noup', 'nosum', 'sumonly', 'scratch')]
            for a, b in comparisons:
                seeds = (1,) if a not in WINDOWS else (1, 2, 3)
                for metric in fit_metrics_cache[f'{family}_{a}_s1']:
                    per = np.stack([fit_metrics_cache[f'{family}_{a}_s{s}'][metric] -
                                    fit_metrics_cache[f'{family}_{b}_s{s}'][metric] for s in seeds])
                    v = per.mean(axis=0); lo, hi = interval(v[1:1+rs.nb]); loc = v[1+rs.nb:]
                    contrast_rows.append(dict(scope=scope, family=family, contrast=a+'-'+b, metric=metric,
                                              estimate=v[0], lo=lo, hi=hi, n_seeds=len(seeds),
                                              n_aois=pp.aoi.nunique(), n_pairs=len(pp), n_clusters=len(represented),
                                              loco_min=nanmin(loc), loco_max=nanmax(loc),
                                              seed1=per[0, 0], seed2=per[1, 0] if len(seeds)>1 else np.nan,
                                              seed3=per[2, 0] if len(seeds)>1 else np.nan,
                                              seeds_same_sign=bool(np.all(np.sign(per[:, 0]) == np.sign(v[0])))))
                    if scope == 'same_activation':
                        for cluster, value in zip(represented, loc):
                            loco_rows.append(dict(family=family, contrast=a+'-'+b, metric=metric,
                                                  omitted_cluster=cluster, estimate=value))
        print('bootstrap complete', scope, 'pairs', len(pp), 'clusters', len(represented), flush=True)
    write(pd.DataFrame(all_summary), 'summary.csv'); write(pd.DataFrame(per_seed), 'per_seed.csv')
    write(pd.DataFrame(contrast_rows), 'contrasts.csv'); write(pd.DataFrame(loco_rows), 'leave_one_cluster_out.csv')
    write(pd.DataFrame(counts), 'sample_counts.csv')
    # Raw sign contingency tables explicitly preserve both kinds of ties.
    signs = ch.groupby(['kind', 'family', 'config', 'seed', 'label_sign', 'pred_sign']).size().rename('n').reset_index()
    write(signs, 'direction_contingency.csv')
    return pd.DataFrame(all_summary), pd.DataFrame(counts)


def residual_input_audit(groups, p, sanity):
    """Inspect native local rasters and saved probabilities for unexplained T0 flips.

    Native inputs are compared only where the summary-level sanity check fails.
    This is an audit, not a claim to have rerun inference or proved its exact cause.
    """
    import rasterio
    root = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/patches')
    runs = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
    residual = sanity[sanity.flip_same_static_optical_patch > 0]
    raw, probs = [], []
    for pair_id, s in residual.groupby('pair_id'):
        g = next(g for g in groups if pair_id in g['pair_ids'])
        j = g['pair_ids'].index(pair_id)
        ra, rb = g['rows'][j:j+2]
        for rowa, rowb in zip(ra, rb):
            pa, pb = p.loc[rowa], p.loc[rowb]
            for suffix, folder, bands in [('input_10m', 'patches_tif', [3, 4]),
                                           ('input_80m', 'patches_tif', [1, 2, 3, 4, 5]),
                                           ('input_160m', 'input_160m_soilgrids_bolzano', [1, 2])]:
                paths = [root / folder / v.activation / v.folder_name / f'patch_{int(v.patch_number):04d}_{suffix}.tif'
                         for v in (pa, pb)]
                record = dict(pair_id=pair_id, aoi=g['aoi'], patch_number=int(pa.patch_number), input=suffix,
                              available=all(f.exists() for f in paths), values=0, different_values=np.nan,
                              source_a=str(paths[0]), source_b=str(paths[1]))
                if record['available']:
                    with rasterio.open(paths[0]) as z:
                        a = z.read(bands)
                    with rasterio.open(paths[1]) as z:
                        b = z.read(bands)
                    record.update(values=a.size, different_values=int(changed(a, b).sum()))
                raw.append(record)
        for _, fit in s.iterrows():
            path = runs / f'final_t0/seed{fit.seed}/{fit.family}/inference_final'
            z = np.load(path / 'preds/test__prob.f16.npy', mmap_mode='r')
            a, b = z[ra].astype(float), z[rb].astype(float)
            threshold = json.loads((path / 'threshold.json').read_text())['threshold']
            audit_mask = g['mask'] & (g['static_same_patch'] & g['optical_same_patch'][j])[:, None, None]
            flip = ((a >= threshold) != (b >= threshold)) & audit_mask
            probs.append(dict(pair_id=pair_id, fit=fit.fit, threshold=threshold, scored_flips=int(flip.sum()),
                              max_abs_probability_difference=float(np.abs(b-a).max()),
                              max_flipped_probability_distance_to_threshold=float(np.max(np.abs(np.r_[a[flip], b[flip]]-threshold))),
                              minimum_flipped_probability=float(np.min(np.r_[a[flip], b[flip]])),
                              maximum_flipped_probability=float(np.max(np.r_[a[flip], b[flip]]))))
    write(pd.DataFrame(raw), 't0_native_input_audit.csv')
    write(pd.DataFrame(probs), 't0_probability_audit.csv')
    return raw, probs


def reporting_tables(summary, pr, ch, weights):
    main = summary[summary.scope == 'same_activation']
    table = []
    for fam in ('unet', 'vit'):
        for config, name in NAMES.items():
            d = main[(main.family == fam) & (main.config == config)].set_index('metric')
            row = dict(family=fam, configuration=name, n_seeds=int(d.n_seeds.iloc[0]),
                       n_aois=int(d.n_aois.iloc[0]), n_pairs=int(d.n_pairs.iloc[0]), n_clusters=int(d.n_clusters.iloc[0]))
            wh = pd.read_csv(OUT / 'within_aoi_spearman.csv')
            wh = wh[(wh.family == fam) & (wh.config == config)]
            row['rho_eligible_aois'] = wh.aoi.nunique()
            row['rho_eligible_clusters'] = wh.cluster.nunique()
            for seed, w in wh.groupby('seed'):
                row[f'rho_defined_aois_seed{seed}'] = int(w.rho.notna().sum())
                row[f'rho_defined_clusters_seed{seed}'] = w.loc[w.rho.notna(), 'cluster'].nunique()
            for metric in ('direction', 'elasticity', 'gain_pod', 'loss_pod', 'gain_precision', 'loss_precision',
                           'change_pod', 'change_precision', 'flip_share', 'flip_predicted_union', 'within_rho',
                           'iou', 'pod', 'pofd', 'label_share', 'predicted_share'):
                for stat in ('estimate', 'lo', 'hi'):
                    row[metric+'_'+stat] = d.loc[metric, stat]
            table.append(row)
    write(pd.DataFrame(table), 'main_table.csv')
    cross = ch[ch.kind == 'different_activation'].copy()
    for mode in ('gain', 'loss'):
        cross[mode+'_pod'] = ratio(cross[mode+'_tp'], cross['label_'+mode])
        cross[mode+'_precision'] = ratio(cross[mode+'_tp'], cross['predicted_'+mode])
    cross['direction'] = np.where(cross.label_sign != 0, cross.label_sign == cross.pred_sign, np.nan)
    cross['change_pod'] = ratio(cross.gain_tp+cross.loss_tp, cross.label_gain+cross.label_loss)
    cross['change_precision'] = ratio(cross.gain_tp+cross.loss_tp, cross.predicted_flip)
    cross['flip_share'] = ratio(cross.predicted_flip, cross.cells)
    write(cross, 'cross_activation_descriptive.csv')
    # Product-level seed means retain n_seeds; no invented population interval for one product/cluster.
    numeric = ['label_share', 'predicted_share', 'iou', 'pod', 'pofd']
    grouping = pr.groupby(['aoi', 'kind', 'cluster', 'folder_name', 'family', 'config'])
    prod = grouping[numeric].mean().join(grouping.seed.nunique().rename('n_seeds'))
    prod = prod.join(grouping[['iou', 'pod', 'pofd']].count().add_prefix('n_defined_')).reset_index()
    write(prod, 'product_seed_means.csv')


def interval(v):
    return tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).any() else (np.nan, np.nan)


def nanmin(v):
    return float(np.nanmin(v)) if np.isfinite(v).any() else np.nan


def nanmax(v):
    return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan


def validate_outputs():
    """Independent row-weighted bootstrap oracle and transition conservation checks."""
    import hashlib
    from PIL import Image
    pairs = pd.read_csv(OUT / 'pairs.csv.gz')
    prod = pd.read_csv(OUT / 'products.csv.gz')
    summary = pd.read_csv(OUT / 'summary.csv')
    same = pairs[pairs.kind == 'same_activation']
    assert same.fit.nunique() == 38 and len(same) == 38*140
    assert same.aoi.nunique() == 76 and same.cluster.nunique() == 28
    ref = same.drop_duplicates('pair_id')
    assert ref.cells.sum() == 21849768
    assert ref.label_gain.sum() == 261634 and ref.label_loss.sum() == 335901
    assert len(set(ref.product_a) | set(ref.product_b)) == 218
    # Independently validate exact geometry and cell masks from the saved source rows.
    p, _, _ = load_index()
    label = np.load(C / 'cells_test.npz')['label']
    pa = pd.read_csv(OUT / 'pair_alignment.csv')
    assert len(pa) == 159 and (pa.status == 'verified').sum() == 140
    with np.load(OUT / 'pair_cells.npz') as saved:
        for r in pa.itertuples():
            ordered = p.loc[p.aoi == r.aoi, ['folder_name', 'date']].drop_duplicates().sort_values('date')
            assert ordered.folder_name.tolist()[r.consecutive_index:r.consecutive_index+2] == [r.product_a, r.product_b]
            joined = p[p.folder_name == r.product_a].merge(p[p.folder_name == r.product_b], on=BOUNDS, validate='1:1')
            assert len(joined) == r.matched_patches
            if r.status != 'verified':
                assert joined.empty or not ((label[joined.row_index_x] >= 0) & (label[joined.row_index_y] >= 0)).any()
                continue
            rr = saved[r.support_id + '_rows']
            assert set(zip(*rr)) == set(zip(joined.row_index_x, joined.row_index_y))
            shape = tuple(saved[r.support_id + '_shape'])
            mask = np.unpackbits(saved[r.support_id + '_mask'], count=int(np.prod(shape))).reshape(shape).astype(bool)
            assert np.array_equal(mask, (label[rr[0]] >= 0) & (label[rr[1]] >= 0))
            assert int(mask.sum()) == r.common_scored_cells
    with np.load(OUT / 'common_cells.npz') as saved:
        for key in saved.files:
            if not key.endswith('_rows'):
                continue
            aoi = key[:-5]; rr = saved[key]
            assert len(rr) == p.loc[p.aoi == aoi, 'folder_name'].nunique()
            for row in rr[1:]:
                assert np.array_equal(p.loc[rr[0], BOUNDS].to_numpy(), p.loc[row, BOUNDS].to_numpy())
            shape = tuple(saved[aoi + '_shape'])
            mask = np.unpackbits(saved[aoi + '_mask'], count=int(np.prod(shape))).reshape(shape).astype(bool)
            assert np.array_equal(mask, np.all(label[rr] >= 0, axis=0))
    del label
    dn = ref[ref.aoi == 'EMSR848_AOI03'].sort_values('date_a')
    assert dn.cells.tolist() == [463061, 463363, 463252, 462675]
    assert dn.date_b.tolist() == [20251101, 20251104, 20251106, 20251108]
    audit = pd.read_csv(OUT / 'alignment.csv')
    n_all_products = audit.loc[audit.all_date_status == 'verified', 'n_products'].sum()
    assert len(prod[prod.kind == 'same_activation']) == 38*n_all_products
    assert np.array_equal(pairs.predicted_b-pairs.predicted_a, pairs.predicted_gain-pairs.predicted_loss)
    assert np.array_equal(pairs.label_b-pairs.label_a, pairs.label_gain-pairs.label_loss)
    assert (pairs.gain_tp <= pairs[['predicted_gain', 'label_gain']].min(axis=1)).all()
    assert (pairs.loss_tp <= pairs[['predicted_loss', 'label_loss']].min(axis=1)).all()
    assert (pairs.predicted_flip <= pairs.predicted_union).all()
    assert (pairs.predicted_union <= pairs.cells).all()
    cb = np.load(C / 'clusters_boot.npz')
    w = np.vstack([np.ones(61), cb['weights']])
    rng = np.random.default_rng(int(cb['boot_seed']))
    expected = np.stack([np.bincount(rng.integers(0, 61, 61), minlength=61) for _ in range(4000)])
    assert np.array_equal(expected, cb['weights'])
    independent = {}
    for fit, d in same.groupby('fit'):
        pw = w[:, d.cluster.to_numpy()]
        def fraction(num, den):
            return ratio(pw @ np.asarray(num, float), pw @ np.asarray(den, float))
        x, y = d.dlog_label.to_numpy(), d.dlog_pred.to_numpy()
        n = pw.sum(axis=1); sx, sy = pw@x, pw@y
        slope = ratio(pw@(x*y)-sx*sy/n, pw@(x*x)-sx*sx/n)
        ps = prod[(prod.fit == fit) & (prod.kind == 'same_activation')]
        wp = w[:, ps.cluster.to_numpy()]
        independent[fit] = dict(
            direction=fraction((d.label_sign == d.pred_sign) & (d.label_sign != 0), d.label_sign != 0),
            elasticity=slope,
            change_pod=fraction(d.gain_tp+d.loss_tp, d.label_gain+d.label_loss),
            change_precision=fraction(d.gain_tp+d.loss_tp, d.predicted_flip),
            flip_share=fraction(d.predicted_flip, d.cells),
            iou=ratio(wp@ps.tp.to_numpy(), wp@(ps.tp+ps.fp+ps.fn).to_numpy()))
    max_error = 0
    for (fam, config), d in same.groupby(['family', 'config']):
        fits = sorted(d.fit.unique())
        for metric in independent[fits[0]]:
            v = np.mean([independent[f][metric] for f in fits], axis=0)
            expected = np.r_[v[0], np.nanpercentile(v[1:], [2.5, 97.5])]
            q = summary[(summary.scope == 'same_activation') & (summary.family == fam) &
                        (summary.config == config) & (summary.metric == metric)].iloc[0]
            actual = q[['estimate', 'lo', 'hi']].to_numpy(float)
            max_error = max(max_error, float(np.max(np.abs(expected-actual))))
            assert np.allclose(expected, actual, rtol=1e-10, atol=1e-12)
    manifest = pd.read_csv(OUT / 'map_manifest.csv')
    assert len(manifest) == 17 and (manifest.candidate == 'main').sum() == 1
    pngs = [Path(f) for f in manifest.png] + [BASE / 'png/5_3' / ('fig5_repeat_summary_'+s+'.png') for s in ('area', 'cells')]
    images = []
    for f in pngs:
        with Image.open(f) as im:
            assert im.format == 'PNG'
            images.append(dict(path=str(f), width=im.width, height=im.height))
            im.verify()
    result = dict(all_38_fits_present=True, transition_conservation=True,
                  exact_geometry_and_pair_masks_verified=True, original_adjacency_verified=True,
                  all_date_geometry_and_masks_verified=True, danang_first_five_dates_verified=True,
                  saved_bootstrap_draws_reproduced=True, independent_bootstrap_metrics=6,
                  independent_bootstrap_configurations=18, max_absolute_bootstrap_error=max_error,
                  png_count=len(images), image_integrity=images,
                  note='Image-integrity verification complements manual visual inspection; it does not certify appearance.')
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    # Inventory contains only the explicitly permitted task artifacts.
    files = sorted(OUT.iterdir()) + [Path(__file__), Path(__file__).with_name('fig5_repeat_summary.py'),
                                    Path(__file__).with_name('fig5_repeat_maps.py')] + pngs
    inventory = []
    for f in files:
        if f.name == 'file_inventory.csv' or not f.is_file():
            continue
        h = hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        inventory.append(dict(path=str(f), bytes=f.stat().st_size, sha256=h.hexdigest()))
    write(pd.DataFrame(inventory), 'file_inventory.csv')
    print('Validated 38 fits, 18 configurations, six independent bootstrap metrics and 19 PNGs; maximum error', max_error)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p, weights, features = load_index()
    label = np.load(C / 'cells_test.npz')['label']
    assert np.array_equal((label >= 0).sum((1, 2)), p.scored)
    assert np.array_equal((label == 1).sum((1, 2)), p.flood)
    groups, all_groups, audit = align(p, label)
    groups += cross_activation(p, label)
    static = static_checks(groups + all_groups)
    inputs = product_inputs(groups, p)
    pairs = pair_reference(groups, label, inputs)
    pr, ch, wh, sanity = fit_tables(groups + all_groups, p, label, pairs)
    raw_audit, probability_audit = residual_input_audit(groups, p, sanity)
    summary, counts = summarise(pr, ch, wh, weights)
    reporting_tables(summary, pr, ch, weights)
    checks = dict(initial_aois=84, initial_products=243, initial_pairs=159,
                  retained_aois=int((audit.status == 'verified').sum()),
                  dropped_aois=int((audit.status != 'verified').sum()),
                  drop_reasons=audit[audit.status != 'verified'].status.value_counts().to_dict(),
                  all_38_fits_match_patch_counts=True,
                  within_key_different_activation_pairs=0,
                  exact_cross_activation_pairs=int((pairs.kind == 'different_activation').sum()),
                  cross_activation_inference='Descriptive only: one site, one cluster; no population interval.',
                  static_inputs_equal=not bool(static.differing_cells.any()),
                  t0_flips_on_unchanged_optical_patches=int(sanity.flip_same_optical_patch.sum()),
                  t0_flips_on_unchanged_static_optical_patches=int(sanity.flip_same_static_optical_patch.sum()),
                  static_difference_aois=sorted(static.loc[static.differing_cells > 0, 'aoi'].unique().tolist()),
                  t0_residual_native_input_audit=dict(patch_input_comparisons=len(raw_audit),
                      available=sum(r['available'] for r in raw_audit),
                      different_values=sum(r['different_values'] for r in raw_audit if r['available']),
                      interpretation='Small near-threshold discrepancy at Seveso; exact inference cause unverified. Native local rasters and saved probabilities audited; exclusion sensitivity reported.'),
                  n_all_date_aois=int((audit.all_date_status == 'verified').sum()),
                  n_aoi_at_least_3=int(((audit.all_date_status == 'verified') & (audit.n_products >= 3)).sum()),
                  n_atlas_aois=int((audit.n_products >= 4).sum()))
    (OUT / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    methodology = dict(
        configuration_names=NAMES,
        configuration_sources=['Chapter 4 Table 4.4 and sections 4.4.3, 4.5.3', 'rq3_common.ABLATIONS', 'ch5_data.run_dir'],
        spatial_support='Match exact CRS and all four bounds, never patch number. Each original consecutive pair uses its own intersection of scored cells on bound-matched patches. An unmatched date is never bridged. pair_alignment.csv records all 159 candidates; pair_cells.npz and patch_matches.csv.gz expose retained support.',
        all_date_support='Separate exact-bound and scored-cell intersection across ALL dates of each AOI for within-AOI Spearman (at least 3 dates) and descriptive product scores. common_cells.npz stores these masks; main_table.csv reports eligible and defined rho counts per seed. Product scores and rho do not use the 76-AOI pair sample.',
        static_verification='Full saved arrays of elevation, HAND, log UDA, flow direction, clay and sand compared on all common patches, including unscored cells. Differences are recorded rather than assumed absent: recovered Calais/Frankenfels have differing saved clay/sand summaries.',
        aggregation='Product and pair records for every fit. Point estimates are arithmetic means of seed-specific metrics. Cell metrics pool counts; direction and regression weight each consecutive pair equally; within-AOI rho averages defined AOI correlations.',
        uncertainty='Saved 4000 x 61 thesis cluster multiplicities. Resample whole clusters, recompute each seed metric, then average seeds. 95% percentile intervals. Paired differences use same draws. Ablations compare only with seed-1 T30. Undefined replicates excluded; valid counts saved.',
        direction='Primary: same nonzero sign on label-changing pairs. Predicted ties fail. Exact integer cell counts define ties. Label ties excluded from primary, included in direction_all_exact; direction_non_tied excludes both tie types. Full contingency saved.',
        direction_sensitivity='Additional 0.1 percentage-point deadband applied to both changes, conditioning on material label change; equal-AOI averaging; exclusion of pairs with differing saved static inputs, and exclusion of the Seveso T0 discrepancy. These are sensitivities, not selected alternative primary definitions.',
        elasticity='OLS with intercept: delta natural log(predicted share + 1e-4) on delta natural log(labelled share + 1e-4). Sensitivities: strictly positive raw shares, half-cell pseudocount, and linear share changes.',
        cell_changes='gain=0 to 1, loss=1 to 0. A correct change requires same cell and direction. Combined change POD=(gain TP+loss TP)/(label gains+losses); precision denominator predicted gains+losses. flip_share denominator all common scored cells; flip_predicted_union denominator union of predicted flood at either date.',
        optical_sanity='NDVI/NDBI are float16 summaries of valid 10 m pixels within 80 m cells, not exact model tensors. Equality of summaries does not prove identical 10 m inputs. The residual audit additionally requires unchanged saved static summaries. Same-optical-patch checks include the whole patch to allow spatial receptive fields. Missing-to-present changes count as optical changes.',
        rho_sensitivity='Ordinary Spearman is undefined for constant prediction; defined counts and clusters saved per seed. Additional sensitivity assigns zero to constant predictions when labels vary, without calling that value an ordinary Spearman correlation.',
        inputs='Patch summaries weighted by common scored cells, identical spatial weights at both dates. API k=0.90 truncated to each input window; local/upstream precipitation sums and soil moisture. Weather dates verified to end on day -1; API30 and day -4 SM checked against patch_antecedent.csv. Invalid SM excluded with coverage recorded.',
        input_response='Univariate Pearson correlations and OLS response in predicted-share percentage points per sample SD of delta log1p(API/precipitation) or delta SM; bootstrap recomputes slopes, using fixed original-sample SD for readable units. Observational associations, not causal effects. T0 uses 30-day summaries as negative-control associations; SM day -4 is not an input of T3. Fixed-30-day versions enable same-exposure comparisons across lengths.',
        optical_strata='Pair metrics use the stated pair subset. Product and within-AOI metrics in strata describe all dates of participating AOIs and should not be interpreted as pair-restricted product scores.',
        cross_activation='AOI key contains activation. Separately match exact CRS and patch bounds across activations; only the Olmedo shared footprint qualifies. Never equate spatial dependence blocks with AOIs.',
        maps='Seed-majority maps for T0/T7/T30; quantitative results remain seed means. All products displayed on native grids at a fixed geographic extent, with individual-product map scores saved only in map_metrics.csv. Pair support and breaks are documented in map_manifest.csv and pair_alignment.csv, not inferred from full-footprint maps. No registration/resampling. Main selection ranks non-Da-Nang sequences with at least 4 consecutively matched dates by cumulative absolute pairwise labelled-area change; selection and caption metadata saved in map_selection.csv/map_manifest.csv.',
        limitations=['Common-cell selection reduces spatial coverage.', 'Intervals describe test-cluster sampling, not a population of random training seeds.', 'Single-seed ablations lack training-variation intervals.', 'Mapping changes can reflect sensor, label type and delineation differences as well as water dynamics.', 'Input associations are confounded by weather timing, changing optical composites and flood evolution.', 'Recovered Calais/Frankenfels pairs have differing saved clay/sand summaries; static-unchanged sensitivity provided.'])
    (OUT / 'methodology.json').write_text(json.dumps(methodology, indent=2) + '\n')
    print(json.dumps(checks, indent=2), flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true', help='validate existing outputs independently; write inventory')
    args = parser.parse_args()
    validate_outputs() if args.validate else main()
