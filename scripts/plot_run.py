#!/usr/bin/env python3
"""Five figures per final run (runs_final), the set that scripts/plot_run.py made for the old runs.

    python scripts/plot_run.py <run_dir> [<run_dir> ...]
    python scripts/plot_run.py --all          # every finished fit under runs_final

Reads saved outputs only: training_log.csv, run_manifest.json and inference_final/ (metrics.csv, threshold.json,
patches.csv, preds/test__prob|labels|tau|index). Writes <run>/plots/:
  fig1_loss.png                 train and validation segmentation loss, the selected epoch marked
  fig2_test.png                 test, flood-only precision / recall / IoU / AUPRC, basin-seen and basin-unseen
  fig3_threshold_head_test.png  global threshold against the per-patch threshold head, flood-only IoU
  fig4_per_product_test.png     distribution over flood-positive mapping products (macro)
  fig5_micro_macro_test.png     pooled (micro) against the mean over mapping products (macro)
  per_product_test.csv          the numbers behind fig4 and fig5

Differences from the old runs: one checkpoint is evaluated (the epoch with the highest validation flood-only IoU,
run_manifest.json selection), so there is one figure of each kind; "event" is now the mapping product; the regimes
are basin-seen and basin-unseen (the one partial ground unit, 377 patches, is in neither panel).
metrics.csv holds no AUPRC per regime and no threshold-head scores per regime, so both are recomputed from the
saved probabilities with the evaluator's rules (counts: p >= threshold on cells with label >= 0; AUPRC: Keras
AUC(curve='PR'), 400 thresholds pooled, 200 per product, reproduced in numpy with exact integer counts). Checks
against metrics.csv: global and head TP/FP/FN of the whole test subset (exact), pooled and per-product IoU per regime
(1e-9), mean per-product AUPRC of the whole test subset (5e-4).
The pooled AUPRC of metrics.csv is NOT used: evaluate.py passes all ~49 M test cells to one Keras AUC update, whose
float32 counters lose counts above 2**24 (12 Oct 2026, final_t30/seed1/unet: TF in one call 0.2763 = metrics.csv;
TF in 4 M-cell batches 0.2738 = this script; false positives at the lowest threshold 30,223,098 in TF against
46,499,674 counted). The script prints both values for every run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.ticker import MaxNLocator                          # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / 'runs_final'
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
# seaborn 'colorblind' (Wong 2011), as in scripts/plot_run.py
CB = {'blue': '#0173B2', 'orange': '#DE8F05', 'vermillion': '#D55E00'}
TRAIN, VAL = CB['blue'], CB['orange']
PAIR = (CB['blue'], CB['orange'])
COHORTC = (CB['blue'], CB['vermillion'])
REGIMES = (('seen', 'basin-seen'), ('unseen', 'basin-unseen'))
FAM = {'unet': 'U-Net', 'vit': 'ViT'}
CONFIG = {'t3': 'T3', 't7': 'T7', 't15': 'T15', 'noup': 'T30 local', 'nosum': 'T30 daily',
          'sumonly': 'T30 statistics'}
METS = (('precision', 'precision'), ('recall', 'recall'), ('iou', 'IoU'), ('auprc', 'AUPRC'))


def title_of(d: Path) -> str:
    """External model name (Chapter 4 names, Ravi's ablation names), e.g. 'U-Net-T30 local, seed 1'."""
    stage, seed, name = d.parent.parent.name, d.parent.name.replace('seed', ''), d.name
    fam, _, rest = name.partition('_')
    cfg = {'final_t0': 'T0', 'final_t30': 'T30', 'final_scratch': 'T30 random init'}.get(stage, CONFIG.get(rest, rest))
    return f'{FAM[fam]}-{cfg}, seed {seed}'


def pr_auc(p: np.ndarray, y: np.ndarray, n: int) -> float:
    """Keras AUC(curve='PR', summation_method='interpolation', num_thresholds=n) on flood-only cells:
    a cell is predicted positive when p > threshold (Davis & Goadrich interpolation)."""
    if not y.any():
        return float('nan')
    eps = 1e-7
    thr = np.array([0.0 - eps] + [(i + 1) / (n - 1) for i in range(n - 2)] + [1.0 + eps], np.float32)
    pos, neg = np.sort(p[y]), np.sort(p[~y])
    tp = (len(pos) - np.searchsorted(pos, thr, side='right')).astype(float)
    fp = (len(neg) - np.searchsorted(neg, thr, side='right')).astype(float)
    fn = len(pos) - tp
    dtp = tp[:-1] - tp[1:]
    P = tp + fp
    dp = np.maximum(P[:-1] - P[1:], 0)
    slope = np.divide(dtp, dp, out=np.zeros_like(dtp), where=dp != 0)
    intercept = tp[1:] - slope * P[1:]
    ratio = np.ones_like(dtp)
    ok = (P[:-1] > 0) & (P[1:] > 0)
    ratio[ok] = P[:-1][ok] / P[1:][ok]
    num = slope * (dtp + intercept * np.log(ratio))
    den = np.maximum(tp[1:] + fn[1:], 0)
    return float(np.divide(num, den, out=np.zeros_like(num), where=den != 0).sum())


def safe(a, b):
    return a / b if b else float('nan')


def compute(d: Path):
    """Per-product table and per-regime pooled numbers, checked against metrics.csv."""
    ev = d / 'inference_final'
    m = pd.read_csv(ev / 'metrics.csv')
    row = lambda c, rule='global': m[(m.cohort == c) & (m.rows == 'full') & (m.rule == rule)].iloc[0]
    ix = pd.read_csv(ev / 'preds/test__index.csv')
    P = np.load(ev / 'preds/test__prob.f16.npy').astype(np.float32)
    Y = np.load(ev / 'preds/test__labels.i8.npy')
    TAU = np.load(ev / 'preds/test__tau.f32.npy')
    thr = np.float32(json.loads((ev / 'threshold.json').read_text())['threshold'])
    keep, truth = Y >= 0, Y == 1
    ax = (1, 2)
    out = {}
    for rule, t in (('global', thr), ('head', TAU.reshape(-1, 1, 1))):
        pred = (P >= t) & keep
        out[rule] = ((pred & truth).sum(ax), (pred & ~truth).sum(ax), (~pred & truth).sum(ax))
        tot = tuple(int(x.sum()) for x in out[rule])
        r = row('test', rule)
        if tot != (int(r.tp), int(r.fp), int(r.fn)):
            raise SystemExit(f'{d}: {rule} TP/FP/FN {tot} differ from metrics.csv {(r.tp, r.fp, r.fn)}')
    g = ix[['folder_name', 'ground_unit', 'basin_regime']].copy()
    g['scored'], g['flood'] = keep.sum(ax), truth.sum(ax)
    for rule, (tp, fp, fn) in out.items():
        pre = '' if rule == 'global' else 'head_'
        g[pre + 'tp'], g[pre + 'fp'], g[pre + 'fn'] = tp, fp, fn
    # per-product within-product AUPRC (200 thresholds), flood-positive products only
    pp = g.groupby('folder_name').agg(regime=('basin_regime', 'first'), ground_unit=('ground_unit', 'first'),
                                      n_patches=('scored', 'size'), scored=('scored', 'sum'), flood=('flood', 'sum'),
                                      tp=('tp', 'sum'), fp=('fp', 'sum'), fn=('fn', 'sum'), head_tp=('head_tp', 'sum'),
                                      head_fp=('head_fp', 'sum'), head_fn=('head_fn', 'sum'))
    if g.groupby('folder_name').basin_regime.nunique().max() > 1:
        raise SystemExit(f'{d}: a mapping product spans two regimes')
    pp = pp[pp.flood > 0].copy()
    idx = g.groupby('folder_name').indices
    pp['auprc'] = [pr_auc(P[idx[f]][keep[idx[f]]], truth[idx[f]][keep[idx[f]]], 200) for f in pp.index]
    pp['precision'] = [safe(a, a + b) for a, b in zip(pp.tp, pp.fp)]
    pp['recall'] = pp.tp / (pp.tp + pp.fn)
    pp['iou'] = pp.tp / (pp.tp + pp.fp + pp.fn)
    pp['head_iou'] = pp.head_tp / (pp.head_tp + pp.head_fp + pp.head_fn)
    # checks against the evaluator
    if abs(pp.auprc.mean() - row('test').product_mean_auprc) > 5e-4:
        raise SystemExit(f'{d}: mean per-product AUPRC {pp.auprc.mean():.4f} differs from metrics.csv '
                         f'{row("test").product_mean_auprc:.4f}')
    au = pr_auc(P[keep], truth[keep], 400)
    print(f'  pooled test AUPRC: exact {au:.4f}, metrics.csv {row("test").pooled_auprc:.4f} '
          f'(difference {row("test").pooled_auprc - au:+.4f})', flush=True)
    reg = {}
    for key, lab in REGIMES:
        r = row(f'test:{key}')
        sel = (g.basin_regime == key).to_numpy()
        T, F, N = (int(x[sel].sum()) for x in out['global'])
        hT, hF, hN = (int(x[sel].sum()) for x in out['head'])
        prod = pp[pp.regime == key]
        if abs(T / (T + F + N) - r.pooled_iou) > 1e-9 or abs(prod.iou.mean() - r.product_mean_iou) > 1e-9:
            raise SystemExit(f'{d}: {lab} IoU differs from metrics.csv')
        reg[key] = {'label': lab, 'n_patches': int(r.n_patches), 'base_rate': g.flood[sel].sum() / g.scored[sel].sum(),
                    'micro': {'precision': r.pooled_precision, 'recall': r.pooled_recall, 'iou': r.pooled_iou,
                              'auprc': pr_auc(P[sel][keep[sel]], truth[sel][keep[sel]], 400)},
                    'head_iou': hT / (hT + hF + hN), 'products': prod}
    return reg, pp, float(thr)


def fig_loss(d, title, man, out):
    log = pd.read_csv(d / 'training_log.csv')
    ep = log.epoch.to_numpy() + (1 if log.epoch.min() == 0 else 0)
    sel = man['selection']
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ep, log.seg_loss, color=TRAIN, lw=1.6, marker='o', ms=3, label='train')
    ax.plot(ep, log.val_seg_loss, color=VAL, lw=1.6, marker='s', ms=3, label='validation')
    e = int(sel['selected_epoch'])
    v = float(log.val_seg_loss[ep == e].iloc[0])
    ax.axvline(e, color='k', ls=':', lw=1)
    ax.annotate(f'selected: epoch {e}\nval flood IoU {sel["selected_value"]:.4f}', (e, v),
                textcoords='offset points', xytext=(12, 16), fontsize=9)
    ax.set_xlabel('epoch')
    ax.set_ylabel('segmentation loss')
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    ax.set_title(f'{title} — training and validation loss')
    fig.tight_layout()
    fig.savefig(out / 'fig1_loss.png')
    plt.close(fig)


def fig_test(reg, title, sub, thr, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for j, (key, _) in enumerate(REGIMES):
        ax, r = axes[j], reg[key]
        vals = [r['micro'][k] for k, _ in METS]
        ax.bar(range(4), vals, .6, color=COHORTC[j])
        for i, v in enumerate(vals):
            ax.text(i, v + .018, f'{v:.4f}', ha='center', fontsize=11)
        ax.set_xticks(range(4))
        ax.set_xticklabels([n for _, n in METS], fontsize=11)
        ax.set_ylim(0, 1.10)
        ax.grid(alpha=.3, axis='y')
        ax.set_title(f"{r['label']}   (n={r['n_patches']:,} patches)\n"
                     f"flood base rate {100 * r['base_rate']:.2f}%   |   threshold {thr:.2f} from validation", fontsize=11.5)
        if j == 0:
            ax.set_ylabel('flood-only score', fontsize=11)
    fig.suptitle(f'{title} — TEST, FLOOD-ONLY | {sub}', fontsize=13, y=.98)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / 'fig2_test.png')
    plt.close(fig)


def fig_head(reg, title, sub, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    gl = [reg[k]['micro']['iou'] for k, _ in REGIMES]
    pt = [reg[k]['head_iou'] for k, _ in REGIMES]
    x, w = np.arange(len(REGIMES)), .38
    ax.bar(x - w / 2, gl, w, label='global threshold (from validation)', color=PAIR[0])
    ax.bar(x + w / 2, pt, w, label='per-patch threshold head', color=PAIR[1])
    for i, (a, b) in enumerate(zip(gl, pt)):
        ax.text(i - w / 2, a + .002, f'{a:.4f}', ha='center', fontsize=8.5)
        ax.text(i + w / 2, b + .002, f'{b:.4f}', ha='center', fontsize=8.5)
        ax.text(i, max(a, b) + .010, f'{(b - a) / a * 100:+.0f}%', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([reg[k]['label'] for k, _ in REGIMES])
    ax.set_ylim(0, max(max(gl), max(pt)) * 1.28)
    ax.set_ylabel('flood-only IoU')
    ax.set_title(sub, pad=12, fontsize=10)
    ax.grid(alpha=.3, axis='y')
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f'{title} — threshold head vs global threshold (TEST, flood-only)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / 'fig3_threshold_head_test.png')
    plt.close(fig)


def fig_products(reg, title, sub, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for k, (mk, mn) in enumerate(METS):
        ax = axes[k]
        data = [reg[key]['products'][mk].dropna().to_numpy() for key, _ in REGIMES]
        labs = [f"{reg[key]['label']}\n(n={len(v)} products)" for (key, _), v in zip(REGIMES, data)]
        bp = ax.boxplot(data, tick_labels=labs, showmeans=True, widths=.55, patch_artist=True)
        for b, cl in zip(bp['boxes'], COHORTC):
            b.set_facecolor(cl)
            b.set_alpha(.65)
        for i, v in enumerate(data):
            if len(v):
                ax.scatter(rng.normal(i + 1, 0.045, len(v)), v, s=5, c='k', alpha=.25, zorder=3)
                ax.text(i + 1, 0.965, f'mean {np.mean(v):.3f}', ha='center', fontsize=8.5,
                        transform=ax.get_xaxis_transform(), bbox=dict(fc='white', ec='0.7', alpha=.85, pad=1.5))
        ax.set_title(f'{mn}  — MACRO (per mapping product)', fontsize=11)
        ax.grid(alpha=.3, axis='y')
        ax.set_ylim(-0.02, 1.02)
        if k == 0:
            ax.set_ylabel('flood-only score')
    fig.suptitle(f'{title} — TEST, per-product distribution | {sub}', fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, .94])
    fig.savefig(out / 'fig4_per_product_test.png')
    plt.close(fig)


def fig_micro_macro(reg, title, sub, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for k, (mk, mn) in enumerate(METS):
        ax, x, w = axes[k], np.arange(len(REGIMES)), .36
        micro = [reg[key]['micro'][mk] for key, _ in REGIMES]
        macro = [float(reg[key]['products'][mk].dropna().mean()) for key, _ in REGIMES]
        ax.bar(x - w / 2, micro, w, label='MICRO (pixel-pooled)', color=PAIR[0])
        ax.bar(x + w / 2, macro, w, label='MACRO (mean of mapping products)', color=PAIR[1])
        for i, (a, b) in enumerate(zip(micro, macro)):
            ax.text(i - w / 2, a + .006, f'{a:.3f}', ha='center', fontsize=8.5)
            ax.text(i + w / 2, b + .006, f'{b:.3f}', ha='center', fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([reg[key]['label'] for key, _ in REGIMES], fontsize=9)
        ax.set_title(mn, fontsize=11)
        ax.grid(alpha=.3, axis='y')
        ax.set_ylim(0, max(max(micro), max(macro)) * 1.30)
        if k == 0:
            ax.set_ylabel('flood-only score')
            ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f'{title} — TEST, MICRO vs MACRO | {sub}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(out / 'fig5_micro_macro_test.png')
    plt.close(fig)


def plot_run(d: Path):
    d = d.resolve()
    title = title_of(d)
    man = json.loads((d / 'run_manifest.json').read_text())
    sub = f"selected checkpoint (epoch {man['selection']['selected_epoch']}, highest validation flood-only IoU)"
    reg, pp, thr = compute(d)
    out = d / 'plots'
    out.mkdir(exist_ok=True)
    fig_loss(d, title, man, out)
    fig_test(reg, title, sub, thr, out)
    fig_head(reg, title, sub, out)
    fig_products(reg, title, sub, out)
    fig_micro_macro(reg, title, sub, out)
    pp.drop(columns=['scored']).to_csv(out / 'per_product_test.csv', float_format='%.6f')
    print(f'{title}: {", ".join(p.name for p in sorted(out.iterdir()))}', flush=True)


def finished():
    return sorted(p.parent.parent for p in RUNS.glob('final_*/seed*/*/inference_final/metrics.csv')
                  if (p.parent.parent / 'COMPLETE').exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', type=Path)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    for d in (finished() if a.all else a.runs):
        plot_run(d)


if __name__ == '__main__':
    main()
