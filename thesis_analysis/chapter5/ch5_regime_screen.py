"""RQ2 numbers: basin-seen against basin-unseen test ground (generalisation to basins absent from training).
Runs where the condition tables are (Bolzano: /home/ravi/shared_data/global_flood_dataset/ravi/ch5_conditions/):

    python3 ch5_regime_screen.py <conditions dir>

Inputs: cells_test.npz, preds_test_bits.npz, fits.json, patches_test.csv.gz (ch5_cells_build.py), clusters_boot.npz,
product_features.csv. A spatial block (ground_unit) carries one basin regime; the single partial block is left out.
Outputs in the same directory:
  rq2_regime_fits.csv     per fit and regime: TP, FP, FN, TN, IoU, POD, POFD, area bias, mean IoU over the
                          flood-positive blocks (all 38 fits, so RQ3 can reuse it)
  rq2_regime_diff.csv     T0, seed means per family: unseen - seen of IoU, block-mean IoU, POD, POFD, with 95% cluster
                          bootstrap intervals (same 4,000 x 61 draws as the rest of the chapter)
  rq2_composition.csv     per regime: blocks, products, patches, cells, flooded fraction (pooled and median block),
                          flooded area per product, low-ground share (HAND < 2 m), mean slope, share of patches whose
                          maximum upstream drainage area exceeds 1,000 km2, permanent-water fraction, NDVI >= 0.6 share,
                          mappings per block, continent, climate and reference-resolution shares
  rq2_bands.csv           T0 seed means: POD and POFD per regime within classes of block flooded fraction and of
                          cell HAND, with unseen - seen and its interval, cells and blocks per class
  rq2_random.csv          composition check: blocks dealt at random to the two regimes (same counts) 4,000 times;
                          per T0 fit the observed and random unseen/seen ratios of pooled IoU, block-mean IoU, POD, POFD
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
FAM = ('unet', 'vit')
SEEDS = (1, 2, 3)
T0 = {f: [f'{f}_t0_s{s}' for s in SEEDS] for f in FAM}
N_RANDOM, RANDOM_SEED = 4000, 20261013


def main():
    cells = np.load(D / 'cells_test.npz')
    bits = np.load(D / 'preds_test_bits.npz')
    fits = json.loads((D / 'fits.json').read_text())
    cb = np.load(D / 'clusters_boot.npz')
    pt = pd.read_csv(D / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'ground_unit', 'basin_regime',
                                                          'continent', 'climate', 'slope_deg_mean', 'uda_log10_max'])
    Y = cells['label']
    n = len(pt)
    scored, flood, dry = Y >= 0, Y == 1, Y == 0
    W = cb['weights'].astype(float)
    K = W.shape[1]
    clus = cb['cluster'].astype(np.int64)
    blocks, bidx = np.unique(pt.ground_unit, return_inverse=True)
    B = len(blocks)
    breg = pt.groupby(bidx).basin_regime.first().reindex(range(B)).to_numpy()
    reg_code = {'seen': 0, 'unseen': 1}
    rb = np.array([reg_code.get(r, -1) for r in breg])               # regime of each block (-1: partial)
    rrow = rb[bidx]                                                   # regime of each row
    bclus = pd.Series(clus).groupby(bidx).first().reindex(range(B)).to_numpy()
    P = {k: np.unpackbits(bits[k])[:Y.size].reshape(Y.shape).astype(bool) for k in fits}

    def counts_rows(k):                                              # per row TP, FP, FN, TN on scored cells
        p = P[k]
        return np.stack([(p & flood).sum((1, 2)), (p & dry).sum((1, 2)), (~p & flood).sum((1, 2)),
                         (~p & dry).sum((1, 2))], 1).astype(float)

    def metrics(c):                                                  # c: (..., 4) TP FP FN TN
        tp, fp, fn, tn = (c[..., i] for i in range(4))
        with np.errstate(invalid='ignore', divide='ignore'):
            return {'iou': tp / (tp + fp + fn), 'pod': tp / (tp + fn), 'pofd': fp / (fp + tn),
                    'area_bias': (tp + fp) / (tp + fn)}

    # 1. per fit and regime
    rows, row_counts = [], {}
    for k in fits:
        c = counts_rows(k)
        row_counts[k] = c
        cb_ = np.zeros((B, 4)); np.add.at(cb_, bidx, c)
        biou = metrics(cb_)['iou']
        bpos = (cb_[:, 0] + cb_[:, 2]) > 0
        for name, code in reg_code.items():
            sel = rrow == code
            m = metrics(c[sel].sum(0))
            rows.append({'fit': k, 'regime': name, **dict(zip(['tp', 'fp', 'fn', 'tn'], c[sel].sum(0))), **m,
                         'block_mean_iou': np.nanmean(biou[(rb == code) & bpos]), 'blocks_flood': int(((rb == code) & bpos).sum())})
    pd.DataFrame(rows).to_csv(D / 'rq2_regime_fits.csv', index=False)

    # 2. T0 differences unseen - seen with cluster bootstrap (rows -> clusters)
    def per_cluster(c, code):
        out = np.zeros((K, 4)); np.add.at(out, clus[rrow == code], c[rrow == code]); return out
    diff_rows = []
    for metric in ('iou', 'pod', 'pofd', 'area_bias', 'block_mean_iou'):
        for f in FAM:
            full, reps = [], []
            for k in T0[f]:
                c = row_counts[k]
                if metric == 'block_mean_iou':
                    cb_ = np.zeros((B, 4)); np.add.at(cb_, bidx, c)
                    biou = metrics(cb_)['iou']; bpos = (cb_[:, 0] + cb_[:, 2]) > 0
                    vals = {code: np.where((rb == code) & bpos, biou, np.nan) for code in (0, 1)}
                    full.append(np.nanmean(vals[1]) - np.nanmean(vals[0]))
                    r = []
                    for code in (0, 1):
                        v, ok = vals[code], ~np.isnan(vals[code])
                        s_ = np.bincount(bclus[ok], v[ok], minlength=K); n_ = np.bincount(bclus[ok], minlength=K)
                        r.append((W @ s_) / (W @ n_))
                    reps.append(r[1] - r[0])
                else:
                    full.append(metrics(c[rrow == 1].sum(0))[metric] - metrics(c[rrow == 0].sum(0))[metric])
                    reps.append(metrics(W @ per_cluster(c, 1))[metric] - metrics(W @ per_cluster(c, 0))[metric])
            rep = np.mean(reps, 0)
            lo, hi = np.nanpercentile(rep, [2.5, 97.5])
            diff_rows.append({'family': f, 'metric': metric, 'unseen_minus_seen': np.mean(full), 'lo': lo, 'hi': hi,
                              'per_seed': ' '.join(f'{x:+.4f}' for x in full)})
    pd.DataFrame(diff_rows).to_csv(D / 'rq2_regime_diff.csv', index=False)

    # 3. composition of the regimes
    pf = pd.read_csv(D / 'product_features.csv').set_index('folder_name')
    hand = cells['hand_m'].astype(np.float32); ndvi = cells['ndvi'].astype(np.float32)
    comp = []
    for name, code in reg_code.items():
        sel = rrow == code
        sc, fl = scored[sel], flood[sel]
        prods = pt.folder_name[sel].unique()
        blk = np.where(rb == code)[0]
        bf = np.bincount(bidx[sel], flood[sel].sum((1, 2)), minlength=B)[blk]
        bs = np.bincount(bidx[sel], scored[sel].sum((1, 2)), minlength=B)[blk]
        nprod_blk = pt[sel].groupby('ground_unit').folder_name.nunique()
        h, v = hand[sel][sc], ndvi[sel][sc]
        comp.append({'regime': name, 'blocks': len(blk), 'blocks_flood': int((bf > 0).sum()), 'products': len(prods),
                     'patches': int(sel.sum()), 'scored_cells': int(sc.sum()), 'flood_cells': int(fl.sum()),
                     'flood_fraction_pooled': fl.sum() / sc.sum(),
                     'flood_fraction_block_median': float(np.median(bf / np.maximum(bs, 1))),
                     'flooded_area_product_median_km2': float(pf.reindex(prods).area_km2.median()),
                     'low_ground_share_hand_lt2': float(np.nanmean(h < 2)), 'hand_median_m': float(np.nanmedian(h)),
                     'ndvi_ge_0.6_share': float(np.nanmean(v >= 0.6)),
                     'patch_mean_slope_median_deg': float(pt.slope_deg_mean[sel].median()),
                     'patches_large_river_share': float((pt.uda_log10_max[sel] >= np.log10(1001)).mean()),
                     'permanent_water_fraction': float((Y[sel] == -1).mean()),
                     'mappings_per_block_mean': float(nprod_blk.mean()),
                     **{f'continent_{c}': float((pt.continent[sel] == c).mean()) for c in sorted(pt.continent.dropna().unique())},
                     **{f'climate_{c}': float((pt.climate[sel] == c).mean()) for c in sorted(pt.climate.dropna().unique())},
                     **{f'resolution_{c}': float(pf.reindex(prods).resolution_class.eq(c).mean())
                        for c in ('very-high', 'high', 'medium')}})
    pd.DataFrame(comp).to_csv(D / 'rq2_composition.csv', index=False)

    # 4. like with like: block flooded-fraction classes and cell HAND classes
    bff = np.zeros(B)
    allf = np.bincount(bidx, flood.sum((1, 2)), minlength=B); alls = np.bincount(bidx, scored.sum((1, 2)), minlength=B)
    bff = allf / np.maximum(alls, 1)
    ff_edges, ff_labels = [0.005, 0.01, 0.03, 0.1], ['<0.5%', '0.5-1%', '1-3%', '3-10%', '>=10%']
    ffc_row = np.digitize(bff, ff_edges)[bidx]
    hand_edges, hand_labels = [1, 2, 5, 10], ['<1 m', '1-2 m', '2-5 m', '5-10 m', '>=10 m']
    hc = np.digitize(np.nan_to_num(hand, nan=-1), hand_edges); hc[~np.isfinite(hand)] = -1
    band_rows = []
    for feat, labels, cls_cell in (('block_flood_fraction', ff_labels, np.broadcast_to(ffc_row[:, None, None], Y.shape)),
                                   ('cell_hand', hand_labels, hc)):
        for j, lab in enumerate(labels):
            inc = scored & (cls_cell == j)
            r = {'feature': feat, 'class': lab}
            for name, code in reg_code.items():
                selr = (rrow == code)[:, None, None] & inc
                r[f'{name}_cells'] = int(selr.sum()); r[f'{name}_flood_cells'] = int((selr & flood).sum())
                blk = np.unique(bidx[(selr & flood).any((1, 2))]); r[f'{name}_blocks_flood'] = len(blk)
            for metric in ('pod', 'pofd'):
                for f in FAM:
                    full, reps = {0: [], 1: []}, []
                    rep_pair = []
                    for k in T0[f]:
                        p = P[k]
                        tp = (p & flood & inc).sum((1, 2)); fp = (p & dry & inc).sum((1, 2))
                        fn = (~p & flood & inc).sum((1, 2)); tn = (~p & dry & inc).sum((1, 2))
                        c = np.stack([tp, fp, fn, tn], 1).astype(float)
                        vals = []
                        for code in (0, 1):
                            m = metrics(c[rrow == code].sum(0))[metric]; full[code].append(m)
                            vals.append(metrics(W @ per_cluster(c, code))[metric])
                        rep_pair.append(vals[1] - vals[0])
                    r[f'{f}_{metric}_seen'] = np.mean(full[0]); r[f'{f}_{metric}_unseen'] = np.mean(full[1])
                    rep = np.mean(rep_pair, 0)
                    r[f'{f}_{metric}_diff'] = r[f'{f}_{metric}_unseen'] - r[f'{f}_{metric}_seen']
                    r[f'{f}_{metric}_lo'], r[f'{f}_{metric}_hi'] = np.nanpercentile(rep, [2.5, 97.5])
            band_rows.append(r)
            print(feat, lab, {k: r[k] for k in r if k.endswith('cells') or k.endswith('flood')}, flush=True)
    pd.DataFrame(band_rows).to_csv(D / 'rq2_bands.csv', index=False)

    # 5. composition check: blocks dealt at random, same numbers per regime
    rng = np.random.default_rng(RANDOM_SEED)
    keep = np.where(rb >= 0)[0]
    n_unseen = int((rb[keep] == 1).sum())
    rand_rows = []
    blk_counts = {}
    for k in [k for f in FAM for k in T0[f]]:
        cb_ = np.zeros((B, 4)); np.add.at(cb_, bidx, row_counts[k]); blk_counts[k] = cb_
    obs = {}
    perm = np.array([rng.permutation(keep)[:n_unseen] for _ in range(N_RANDOM)])
    for k, cb_ in blk_counts.items():
        biou = metrics(cb_)['iou']; bpos = (cb_[:, 0] + cb_[:, 2]) > 0
        def ratios(unseen_idx):
            u = np.zeros(B, bool); u[unseen_idx] = True; s_ = np.zeros(B, bool); s_[keep] = True; s_ &= ~u
            mu, ms = metrics(cb_[u].sum(0)), metrics(cb_[s_].sum(0))
            return [mu['iou'] / ms['iou'], np.nanmean(biou[u & bpos]) / np.nanmean(biou[s_ & bpos]),
                    mu['pod'] / ms['pod'], mu['pofd'] / ms['pofd']]
        o = ratios(np.where(rb == 1)[0])
        R = np.array([ratios(ix) for ix in perm])
        for j, name in enumerate(('iou', 'block_mean_iou', 'pod', 'pofd')):
            rand_rows.append({'fit': k, 'ratio': name, 'observed': o[j], 'random_median': np.median(R[:, j]),
                              'random_lo': np.percentile(R[:, j], 2.5), 'random_hi': np.percentile(R[:, j], 97.5),
                              'share_random_ge_observed': float((R[:, j] >= o[j]).mean())})
        print(k, 'done random', flush=True)
    pd.DataFrame(rand_rows).to_csv(D / 'rq2_random.csv', index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
