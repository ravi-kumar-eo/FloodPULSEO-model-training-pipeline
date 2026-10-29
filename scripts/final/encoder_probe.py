#!/usr/bin/env python3
"""Inside the hydroclimatic encoders of a T30 fit (thesis Section 5.3.7, exploratory).

    python encoder_probe.py --run-dir <runs_final>/final_t30/seed1/vit [--n 2000] [--batch 64]

Loads the fit exactly as scripts/final/evaluate.py does and runs a fixed random sample of test rows (seed 0) through
the network with recording hooks on its own layers (nothing in the network changes):
  A  representation change: the hydroclimatic output of the network (ViT: output of the gated cross-attention block,
     the tokens the head reads; U-Net: the bottleneck after the temporal collapse) with the true daily inputs and with
     a neutral input (the mean daily inputs of the sample, day by day), as ||rep(w) - rep(w_ref)|| / ||rep(w)|| per
     row, and the mean |p(w) - p(w_ref)| over the scored cells. ViT also: ||g * CrossAttn|| / ||trunk tokens||.
  B  attention over the 30 days: ViT, the cross-attention scores of the image tokens over the day tokens (mean over
     heads and image tokens); U-Net, the L-TAE masks (mean over heads and the 4 x 4 bottleneck cells). Index 0 is
     day -30, index 29 day -1.
  C  encoder output per row: ViT, the day tokens after the day transformer, mean over the days; U-Net, the day vectors
     of the patch weather encoder, mean over the days.
  D  removal of single days: for each day, the local and upstream precipitation of that day set to zero; mean
     |p - p_true| over the scored cells and the change in predicted flood cells (at the fit's own threshold).
Writes <run-dir>/inference_probe/probe.npz (rows, folder_name, patch_number, weather, and the arrays above).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import tensorflow as tf
import yaml

import evaluate as ev                                                    # noqa: E402
from flood.paths import METADATA                                         # noqa: E402
from flood.data.dataset_final import FinalFloodPatches                    # noqa: E402

CAP: dict = {}


def hook(layer, name, keep='out'):
    """Record the output (or the first input) of a layer's call; the computation is unchanged."""
    orig = layer.call

    def wrapped(*a, **k):
        out = orig(*a, **k)
        CAP[name] = a[0] if keep == 'in' else out
        return out
    layer.call = wrapped


def hook_vit_interaction(L):
    """Same computation as GatedCrossAttention.call, keeping the attention scores and the gated term."""
    def wrapped(x, w, training=False):
        h, sc = L.att(L.nq(x), L.nk(w), training=training, return_attention_scores=True)
        gh = tf.cast(L.g, x.dtype) * h
        out = x + gh
        if L.ffn_mult:
            out = out + tf.cast(L.g2, x.dtype) * L.ff(L.nf(out))
        CAP.update(att=sc, trunk=x, gated=gh, rep=out)
        return out
    L.call = wrapped


def run(model, fp, rows, wx, batch, family):
    """Forward pass of `rows` with daily inputs wx (len(rows), 30, 5); returns P and the recorded arrays."""
    P, rec = [], {k: [] for k in ('rep', 'att', 'emb', 'ratio')}
    for s in range(0, len(rows), batch):
        idx = rows[s:s + batch]
        data = [fp.load(int(i)) for i in idx]
        x = [tf.convert_to_tensor(np.stack([d[j] for d in data])) for j in range(3)]
        x.append(tf.convert_to_tensor(wx[s:s + batch]))
        out = model(tuple(x), training=False)
        p = out[0] if isinstance(out, (tuple, list)) else out
        P.append(np.asarray(tf.cast(p, tf.float32))[..., 0])
        if family == 'vit':
            rec['rep'].append(np.asarray(CAP['rep'], np.float32))
            rec['att'].append(np.asarray(tf.reduce_mean(CAP['att'], axis=[1, 2]), np.float32))     # (b, T)
            rec['emb'].append(np.asarray(tf.reduce_mean(CAP['wtok'], axis=1), np.float32))
            g = tf.norm(tf.reshape(CAP['gated'], [len(idx), -1]), axis=1)
            t = tf.norm(tf.reshape(CAP['trunk'], [len(idx), -1]), axis=1)
            rec['ratio'].append(np.asarray(g / t, np.float32))
        else:
            rec['rep'].append(np.asarray(CAP['bott'], np.float32))
            rec['att'].append(np.asarray(tf.reduce_mean(CAP['masks'], axis=[1, 3, 4]), np.float32))  # (b, T)
            rec['emb'].append(np.asarray(tf.reduce_mean(CAP['wx'], axis=1), np.float32))
    return np.concatenate(P), {k: np.concatenate(v) if v else None for k, v in rec.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    run_dir = a.run_dir
    man = json.loads((run_dir / 'run_manifest.json').read_text())
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    family = man['family']
    if cfg.get('weather_mode') != 'upstream' or not int(cfg.get('temporal_days', 0)):
        raise SystemExit('needs an upstream-mode fit with dynamic hydroclimatic variables')
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA)
    dp = ev.data_paths(cfg, md)
    if ev.md5(dp['version']) != man['data_version_md5'] or ev.md5(dp['split'] / 'patches.csv') != man['split_md5']:
        raise SystemExit('REFUSED: data version or split differs from the one this fit trained on')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    thr = float(json.loads((run_dir / 'inference_final' / 'threshold.json').read_text())['threshold'])
    model = ev.build_model(ev.FAMILIES[family][0], cfg)
    w = run_dir / 'checkpoints' / man.get('selection', {}).get('checkpoint', 'best_loss.weights.h5')
    model((tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2)),
           tf.zeros((1,) + ev.weather_shape(cfg))), training=False)
    model.load_weights(str(w))
    net = getattr(model, 'backbone', model)              # FloodModelFinal wraps the network
    if family == 'vit':
        hook_vit_interaction(net.interaction)
        hook(net.day_encoder, 'wtok')
    else:
        hook(net.ltae, 'masks')
        hook(net.ups[0], 'bott', keep='in')
        hook(net.patch_enc, 'wx')

    kw = ev.final_dataset_kwargs(cfg, dp)
    kw.pop('mono_only')
    if not Path(kw['patches_dir']).exists():           # fits trained on AWS record the AWS patch folder
        kw['patches_dir'] = os.environ.get('FLOOD_PATCHES_DIR',
                                           '/home/ravi/shared_data/global_flood_dataset/pipeline/data/patches/patches_tif')
        print(f"patches_dir of the config not found; using {kw['patches_dir']}", flush=True)
    fp = FinalFloodPatches(cohort='test', allow_test=True, **kw)
    rows = np.sort(np.random.default_rng(0).choice(len(fp.df), size=min(a.n, len(fp.df)), replace=False))
    fp.enable_eval_cache(workers=a.workers, rows=rows)
    W = fp._wx[rows, -fp.temporal_days:].copy()          # the network receives the last T days (FinalFloodPatches.weather)
    lab = np.stack([fp.load(int(i))[-1][..., 0] for i in rows]).astype(np.int8)
    keep = lab >= 0
    t0 = time.time()

    P, rec = run(model, fp, rows, W, a.batch, family)
    Wref = np.repeat(W.mean(0, keepdims=True), len(W), axis=0)
    Pr, recr = run(model, fp, rows, Wref, a.batch, family)
    diff = (rec['rep'] - recr['rep']).reshape(len(rows), -1)
    rep_change = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(rec['rep'].reshape(len(rows), -1), axis=1), 1e-12)
    dprob_ref = np.where(keep, np.abs(P - Pr), 0).reshape(len(rows), -1).sum(1) / np.maximum(keep.reshape(len(rows), -1).sum(1), 1)
    print(f'A done ({time.time() - t0:.0f}s): median representation change {np.median(rep_change):.4f}, '
          f'mean |dp| vs neutral input {dprob_ref.mean():.5f}', flush=True)

    T = W.shape[1]
    occl_dprob, occl_dpred = np.zeros((T, len(rows)), np.float32), np.zeros((T, len(rows)), np.float32)
    pred0 = ((P >= thr) & keep).reshape(len(rows), -1).sum(1)
    for t in range(T):
        Wt = W.copy()
        Wt[:, t, 0] = 0.0
        Wt[:, t, 3] = 0.0
        Pt, _ = run(model, fp, rows, Wt, a.batch, family)
        occl_dprob[t] = np.where(keep, np.abs(Pt - P), 0).reshape(len(rows), -1).sum(1) / np.maximum(keep.reshape(len(rows), -1).sum(1), 1)
        occl_dpred[t] = ((Pt >= thr) & keep).reshape(len(rows), -1).sum(1) - pred0
        print(f'D day {t - T} done ({time.time() - t0:.0f}s): mean |dp| {occl_dprob[t].mean():.6f}', flush=True)

    out = run_dir / 'inference_probe'
    out.mkdir(exist_ok=True)
    df = fp.df.reset_index(drop=True)
    np.savez_compressed(out / 'probe.npz', rows=rows, folder_name=df.folder_name.to_numpy()[rows].astype(str),
                        patch_number=df.patch_number.to_numpy()[rows], weather=W, threshold=thr,
                        rep_change=rep_change, dprob_ref=dprob_ref, att=rec['att'], emb=rec['emb'],
                        gate_ratio=rec['ratio'] if rec['ratio'] is not None else np.array([]),
                        occl_dprob=occl_dprob, occl_dpred=occl_dpred, pred0=pred0,
                        flood=(lab == 1).reshape(len(rows), -1).sum(1), scored=keep.reshape(len(rows), -1).sum(1))
    for s in fp._shm.values():
        s.close(); s.unlink()
    print(f'wrote {out}/probe.npz ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
