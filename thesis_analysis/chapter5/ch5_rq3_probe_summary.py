"""Summary of the encoder probes of the T30 networks (exploratory, Section 5.4.3): one row per fit with a probe.npz
(experiments/src/scripts/final/encoder_probe.py, copied from Bolzano to
experiments/runs_final/final_t30/seed<s>/<family>/inference_probe/ and, for T3, T7 and T15 (Section 5.4),
experiments/runs_final/final_window/seed<s>/<family>_t<T>/inference_probe/). Each probe holds 2,000 test patches drawn
with seed 0, the same patches for every fit; the antecedent descriptors (catchment area, API, soil moisture) are taken
from the 30-day inputs of the first T30 probe, so that they are the same for every input-history length.

Columns of data/rq3/probes.csv:
  A  rep_median, rep_p95       relative change of the hydroclimatic output of the network (ViT: tokens after the gated
                               cross-attention; U-Net: bottleneck after the temporal collapse) when the daily inputs are
                               replaced by their mean over the 2,000 patches, day by day
     dprob_mean, dprob_p95     mean |p - p_neutral| over the scored cells of a patch, mean and 95th percentile
     dprob_top5_share          share of the summed |p - p_neutral| held by the 5% of patches with the largest change
     rho_<x>                   Spearman correlation of the per-patch |p - p_neutral| with the upstream catchment area,
                               the API of the upstream precipitation (k = 0.9), the soil moisture of day -4 and the
                               number of reference flood cells
     gate_ratio_median         ViT only: ||g * cross-attention|| / ||trunk tokens||
  B  att_last7_rel             mean attention on days -7 to -1 (all days when T < 7) relative to a uniform weighting
     att_rowspread_median      median over patches of the largest over the smallest daily weight of a patch
     att_rowspread_iqr         interquartile range of that ratio over patches (0 = the same weighting for every patch)
     rho_att_precip            median over patches of the Spearman correlation between the daily weight and the daily
                               local plus upstream precipitation
  D  occl_total                summed over the 30 days of the mean |p - p_true| when one day's precipitation is removed
     occl_day1_share           share of that sum from day -1; occl_last7_share from days -7 to -1 (1 when T <= 7)

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_rq3_probe_summary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final/final_t30')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq3' / 'probes.csv'
K = 0.9 ** np.arange(29, -1, -1)                    # weight 1 on day -1 (index 29), 0.9 per day back in time


def summary(z, family, window, seed, W30):
    W, dp, rep, att, o = z['weather'], z['dprob_ref'], z['rep_change'], z['att'], z['occl_dprob']
    api = (W30[:, :, 3] * K).sum(1)
    top = dp >= np.percentile(dp, 95)
    spread = att.max(1) / att.min(1)
    P = W[:, :, 0] + W[:, :, 3]
    ok = (P.std(1) > 0) & (att.std(1) > 0)
    r = {'family': family, 'window': window, 'seed': seed, 'patches': len(dp),
         'rep_median': np.median(rep), 'rep_p95': np.percentile(rep, 95),
         'dprob_mean': dp.mean(), 'dprob_p95': np.percentile(dp, 95), 'dprob_top5_share': dp[top].sum() / dp.sum(),
         'rho_area': spearmanr(W30[:, -1, 4], dp).correlation, 'rho_api': spearmanr(api, dp).correlation,
         'rho_sm': spearmanr(W30[:, -4, 1], dp).correlation, 'rho_flood': spearmanr(z['flood'], dp).correlation,
         'gate_ratio_median': float(np.median(z['gate_ratio'])) if z['gate_ratio'].size else np.nan,
         'att_last7_rel': att[:, -7:].mean() * att.shape[1],
         'att_rowspread_median': np.median(spread), 'att_rowspread_iqr': np.subtract(*np.percentile(spread, [75, 25])),
         'rho_att_precip': np.median([spearmanr(att[i], P[i]).correlation for i in np.flatnonzero(ok)]) if ok.any()
         else np.nan,
         'occl_total': o.mean(1).sum(), 'occl_day1_share': o[-1].sum() / o.sum(), 'occl_last7_share': o[-7:].sum() / o.sum()}
    return r


def main():
    rows, ref, W30 = [], None, None
    fits = [(f, 't30', s, RUNS / f'seed{s}/{f}/inference_probe/probe.npz') for f in ('unet', 'vit') for s in (1, 2, 3)]
    fits += [(f, f't{T}', s, RUNS.parent / f'final_window/seed{s}/{f}_t{T}/inference_probe/probe.npz')
             for f in ('unet', 'vit') for T in (3, 7, 15) for s in (1, 2, 3)]
    for family, window, seed, f in fits:
        if not f.exists():
            print('missing', f)
            continue
        z = np.load(f)
        key = (z['folder_name'], z['patch_number'])
        if ref is None:
            ref = key
        elif not ((key[0] == ref[0]).all() and (key[1] == ref[1]).all()):
            raise SystemExit(f'{f}: probe patches differ from the first probe')
        if W30 is None:
            if window != 't30':
                raise SystemExit('the first probe found must be a T30 probe (30-day descriptors)')
            W30 = z['weather']
        rows.append(summary(z, family, window, seed, W30))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(df.set_index(['family', 'window', 'seed']).T.round(5).to_string())


if __name__ == '__main__':
    main()
