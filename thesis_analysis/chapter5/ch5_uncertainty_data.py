"""Source data of Section 5.5 (uncertainty and error analysis): U-Net and ViT without (T0) and with (T30) the 30-day
input history, seeds 1-3, on the test subset; the seed consistency of every comparison of Sections 5.1, 5.3 and 5.4.

Test subset, full rows, scored cells (label >= 0), each fit at its validation-selected global threshold. Cell
predictions: plot/data/conditions/preds_test_bits.npz; probabilities: runs_final/*/inference_final/preds. Spatial
measures (distances, connected patches) are computed on the mosaic of each mapping product, its test patches placed at
their positions on its 2,560 m patch grid (event_maps.layout), with chessboard distances (one cell = 80 m, diagonal
neighbours included). Writes chapters/chapter5/plot/data/uncertainty/:
  consistency.csv  ViT minus U-Net (T0, T30; seed-matched) and T3, T7, T15, T30 minus T0 (per family; paired by seed):
                   micro and macro IoU change of each seed and of the seed mean with its 95% cluster bootstrap interval,
                   and for ViT minus U-Net the share of the nine cross-seed pairs in which ViT scores higher
  variance.csv     split of the variation of the product IoU (394 flood-positive products) of T0 and T30 into
                   product, family, seed within family, product x family and product x seed within family (sums of
                   squares of the balanced design, shares of the total)
  votes.csv        scored cells and reference flood cells by the number of seeds (0-3) that predict flood
  calibration.csv  scored cells, reference flood cells and summed probability in 20 probability bins, per fit
  brier.csv        Brier score, Brier score of the constant flooded share, expected calibration error, per fit
  errors.csv       false alarms and missed flood cells by type, per fit (types and order in TYPES_FP, TYPES_FN)
  tolerance.csv    predicted cells within d cells of reference flood and reference flood cells within d cells of a
                   predicted cell, d = 0-4, per fit
  components.csv   false-alarm cells by the size of their connected patch (8-connected), per fit

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_uncertainty_data.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402
import event_maps as em  # noqa: E402
from ch5_rq4_data import boot_measures  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'uncertainty'
RQ4 = Path(__file__).resolve().parents[1] / 'data' / 'rq4'
WINDOWS = ('t0', 't30')
FITS = [(f, w, s) for w in WINDOWS for f in cd.FAMILIES for s in cd.SEEDS]
D_MAX = 4                                    # tolerance distances 0-4 cells (0-320 m)
FAR = 99                                     # distance where a product has no cell of the target set
LOW_HAND = 2.0                               # m; low ground (Section 5.3.2: 66-69% of the T0 false alarms)
DENSE_NDVI = 0.6                             # dense vegetation, upper NDVI class of Figure 5.10
TYPES_FP = ('edge', 'near', 'pw', 'low', 'other')   # <= 1 cell from reference flood; 2-3 cells; next to permanent
#                                              water or mixed cells; HAND below 2 m; elsewhere (first match wins)
TYPES_FN = ('edge', 'near', 'pw', 'veg', 'other')   # <= 1 cell from predicted flood; 2-3 cells; next to permanent
#                                              water or mixed cells; NDVI >= 0.6; elsewhere (first match wins)
SIZES = ((1, 1), (2, 9), (10, 99), (100, 999), (1000, 10 ** 9))


def bits(family, window, seed):
    B = np.load(em.C / 'preds_test_bits.npz')
    shape = tuple(B['shape'])
    return np.unpackbits(B[f'{family}_{window}_s{seed}'])[:int(np.prod(shape))].astype(bool).reshape(shape)


# ------------------------------------------------------------------------------------------- seed consistency
def consistency():
    ts = cd.test_set()
    M = {(f, w, s): boot_measures(cd.counts(f, w, s)) for f in cd.FAMILIES for w in WINDOWS for s in cd.SEEDS}
    out = []
    for w in WINDOWS:                          # ViT minus U-Net, seed-matched
        per = {}
        for s in cd.SEEDS:
            (pu, ru, qu), (pv, rv, qv) = M[('unet', w, s)], M[('vit', w, s)]
            mi = (pv['iou'] - pu['iou'], rv['iou'] - ru['iou'])
            vu, vv = qu['iou'].to_numpy(), qv['iou'].to_numpy()
            ma = (float(np.nanmean(vv) - np.nanmean(vu)),
                  cd.boot_mean(vv, ts.product_cluster) - cd.boot_mean(vu, ts.product_cluster))
            per[s] = {'micro_iou': mi, 'macro_iou': ma}
        pairs = {k: np.mean([M[('vit', w, a)][0]['iou'] > M[('unet', w, b)][0]['iou']
                             for a in cd.SEEDS for b in cd.SEEDS]) if k == 'micro_iou' else
                 np.mean([np.nanmean(M[('vit', w, a)][2]['iou']) > np.nanmean(M[('unet', w, b)][2]['iou'])
                          for a in cd.SEEDS for b in cd.SEEDS]) for k in ('micro_iou', 'macro_iou')}
        for k in ('micro_iou', 'macro_iou'):
            for s in (*cd.SEEDS, 'mean'):
                if s == 'mean':
                    pt = np.mean([per[x][k][0] for x in cd.SEEDS])
                    rp = np.mean([per[x][k][1] for x in cd.SEEDS], axis=0)
                else:
                    pt, rp = per[s][k]
                lo, hi = cd.interval(rp)
                out.append({'comparison': f'vit-unet {w}', 'family': 'both', 'seed': s, 'measure': k, 'change': pt,
                            'lo': lo, 'hi': hi, 'pairs_higher': pairs[k] if s == 'mean' else np.nan})
    ch = pd.read_csv(RQ4 / 'changes.csv')
    ch = ch[(ch['from'] == 't0') & ch.measure.isin(['micro_iou', 'macro_iou'])]
    for _, r in ch.iterrows():                 # every input-history length minus T0, paired by seed (Section 5.4)
        out.append({'comparison': f"{r['to']}-t0", 'family': r.family, 'seed': r.seed, 'measure': r.measure,
                    'change': r.change, 'lo': r.lo, 'hi': r.hi, 'pairs_higher': np.nan})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- variance split
def variance():
    pr = pd.read_csv(RQ4 / 'products.csv')
    out = []
    for w in WINDOWS:
        t = pr[pr.window == w].pivot_table(index='folder_name', columns=['family', 'seed'], values='iou')
        Y = t.to_numpy().reshape(len(t), 2, 3)          # product x family x seed
        g = Y.mean()
        p_ = Y.mean(axis=(1, 2))
        f_ = Y.mean(axis=(0, 2))
        fs = Y.mean(axis=0)                             # family x seed
        pf = Y.mean(axis=2)                             # product x family
        n_p, n_f, n_s = Y.shape
        ss = {'product': n_f * n_s * ((p_ - g) ** 2).sum(),
              'family': n_p * n_s * ((f_ - g) ** 2).sum(),
              'seed within family': n_p * ((fs - f_[:, None]) ** 2).sum(),
              'product x family': n_s * ((pf - p_[:, None] - f_[None, :] + g) ** 2).sum(),
              'product x seed within family': ((Y - pf[:, :, None] - fs[None] + f_[None, :, None]) ** 2).sum()}
        tot = ((Y - g) ** 2).sum()
        assert abs(sum(ss.values()) - tot) < 1e-8 * tot
        out += [{'window': w, 'source': k, 'ss': v, 'share': v / tot} for k, v in ss.items()]
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- votes and calibration
def votes(lab):
    keep, flood = lab >= 0, lab == 1
    out = []
    for w in WINDOWS:
        for f in cd.FAMILIES:
            v = sum(bits(f, w, s).astype(np.uint8) for s in cd.SEEDS)
            n = np.bincount(v[keep], minlength=4)
            k = np.bincount(v[flood], minlength=4)
            out += [{'family': f, 'window': w, 'votes': i, 'cells': int(n[i]), 'flood': int(k[i])} for i in range(4)]
    return pd.DataFrame(out)


def calibration():
    rows, brier = [], []
    edges = np.linspace(0, 1, 21)
    for f, w, s in FITS:
        ev = cd.run_dir(f, w, s) / 'inference_final/preds'
        P = np.load(ev / 'test__prob.f16.npy').astype(np.float32)
        Y = np.load(ev / 'test__labels.i8.npy')
        keep = Y >= 0
        p, y = P[keep], (Y[keep] == 1).astype(np.float32)
        del P
        b = np.clip(np.digitize(p, edges[1:-1]), 0, 19)
        n = np.bincount(b, minlength=20)
        k = np.bincount(b, y, minlength=20)
        ps_ = np.bincount(b, p, minlength=20)
        rows += [{'family': f, 'window': w, 'seed': s, 'bin': i, 'lo': edges[i], 'hi': edges[i + 1],
                  'cells': int(n[i]), 'flood': float(k[i]), 'psum': float(ps_[i])} for i in range(20)]
        base = y.mean()
        bs = float(np.mean((p - y) ** 2))
        ece = float(np.sum(np.abs(ps_ - k)) / len(p))
        brier.append({'family': f, 'window': w, 'seed': s, 'brier': bs, 'brier_climatology': float(base * (1 - base)),
                      'brier_skill': 1 - bs / (base * (1 - base)), 'ece': ece, 'mean_probability': float(p.mean()),
                      'flooded_share': float(base)})
        print(f'{f} {w} s{s}: Brier {bs:.4f}  ECE {ece:.4f}', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(brier)


# ------------------------------------------------------------------------------------------- spatial error analysis
def dist_to(mask):
    """Chessboard distance (cells) of every cell to the nearest True cell of mask; FAR where mask is empty."""
    if not mask.any():
        return np.full(mask.shape, FAR, dtype=np.int32)
    return ndimage.distance_transform_cdt(~mask, metric='chessboard').astype(np.int32)


def spatial(lab, hand, ndvi):
    p, _, _, _ = em.data()
    products = p.folder_name.unique()
    preds = {k: bits(*k) for k in FITS}
    err = {k: {('fp', t): 0 for t in TYPES_FP} | {('fn', t): 0 for t in TYPES_FN} for k in FITS}
    tol = {k: np.zeros((D_MAX + 1, 4), dtype=np.int64) for k in FITS}     # pred, pred_near, ref, ref_near
    comp = {k: np.zeros(len(SIZES), dtype=np.int64) for k in FITS}
    eight = np.ones((3, 3), dtype=int)
    t0 = time.time()
    for i, prod in enumerate(products):
        rr, rows, cols = em.layout(prod)
        H, W = (rows.max() + 1), (cols.max() + 1)

        def mos(a, fill):
            m = np.full((H, W, 32, 32), fill, dtype=a.dtype)
            m[rows, cols] = a[rr]
            return m.transpose(0, 2, 1, 3).reshape(H * 32, W * 32)

        L = mos(lab, np.int8(-2))                       # -2: grid position without a test patch
        scored, ref = L >= 0, L == 1
        d_ref, d_pw = dist_to(ref), dist_to(L == -1)
        hd, nd = mos(hand, np.float16(np.nan)), mos(ndvi, np.float16(np.nan))
        for k in FITS:
            pr = mos(preds[k], False) & scored
            fp, fn = pr & ~ref, ref & ~pr
            d_pr = dist_to(pr)
            left = fp.copy()
            for t, m in (('edge', d_ref <= 1), ('near', d_ref <= 3), ('pw', d_pw <= 1), ('low', hd < LOW_HAND),
                         ('other', np.ones_like(fp))):
                hit = left & m
                err[k][('fp', t)] += int(hit.sum())
                left &= ~hit
            left = fn.copy()
            for t, m in (('edge', d_pr <= 1), ('near', d_pr <= 3), ('pw', d_pw <= 1), ('veg', nd >= DENSE_NDVI),
                         ('other', np.ones_like(fn))):
                hit = left & m
                err[k][('fn', t)] += int(hit.sum())
                left &= ~hit
            for d in range(D_MAX + 1):
                tol[k][d] += (int(pr.sum()), int((pr & (d_ref <= d)).sum()), int(ref.sum()),
                              int((ref & (d_pr <= d)).sum()))
            lbl, n = ndimage.label(fp, structure=eight)
            if n:
                size = np.bincount(lbl.ravel())[1:]
                for j, (a, b) in enumerate(SIZES):
                    sel = (size >= a) & (size <= b)
                    comp[k][j] += int(size[sel].sum())
        if i % 50 == 0:
            print(f'{i + 1}/{len(products)} products, {time.time() - t0:.0f} s', flush=True)
    e = pd.DataFrame([{'family': k[0], 'window': k[1], 'seed': k[2], 'kind': kind, 'type': t, 'cells': v}
                      for k in FITS for (kind, t), v in err[k].items()])
    tl = pd.DataFrame([{'family': k[0], 'window': k[1], 'seed': k[2], 'd': d, 'pred': a, 'pred_near': b, 'ref': c,
                        'ref_near': x} for k in FITS for d, (a, b, c, x) in enumerate(tol[k])])
    cp = pd.DataFrame([{'family': k[0], 'window': k[1], 'seed': k[2], 'size_min': a, 'size_max': b, 'cells': v}
                       for k in FITS for (a, b), v in zip(SIZES, comp[k])])
    return e, tl, cp


def check(errors, tol):
    """Error types add up to the FP and FN counts of each fit, and d = 0 reproduces its precision and recall."""
    fits = pd.read_csv(RQ4 / 'fits.csv').set_index(['family', 'window', 'seed'])
    for (f, w, s), g in errors.groupby(['family', 'window', 'seed']):
        r = fits.loc[(f, w, s)]
        fp, fn = g[g.kind == 'fp'].cells.sum(), g[g.kind == 'fn'].cells.sum()
        if (fp, fn) != (r.fp, r.fn):
            raise SystemExit(f'{f} {w} {s}: error types {fp, fn} differ from FP/FN {r.fp, r.fn}')
        t = tol[(tol.family == f) & (tol.window == w) & (tol.seed == s) & (tol.d == 0)].iloc[0]
        if (t.pred_near, t.pred, t.ref_near) != (r.tp, r.tp + r.fp, r.tp):
            raise SystemExit(f'{f} {w} {s}: tolerance d = 0 differs from TP/FP')
    print('checks passed: error types sum to FP and FN; d = 0 equals TP', flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    consistency().to_csv(OUT / 'consistency.csv', index=False)
    print('consistency done', flush=True)
    variance().to_csv(OUT / 'variance.csv', index=False)
    c = np.load(em.C / 'cells_test.npz')
    lab = c['label']
    votes(lab).to_csv(OUT / 'votes.csv', index=False)
    print('votes done', flush=True)
    cal, br = calibration()
    cal.to_csv(OUT / 'calibration.csv', index=False)
    br.to_csv(OUT / 'brier.csv', index=False)
    e, tl, cp = spatial(lab, c['hand_m'], c['ndvi'])
    check(e, tl)
    e.to_csv(OUT / 'errors.csv', index=False)
    tl.to_csv(OUT / 'tolerance.csv', index=False)
    cp.to_csv(OUT / 'components.csv', index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
