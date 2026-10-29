#!/usr/bin/env python3
"""Day attention of a fit with dynamic hydroclimatic variables on ALL test rows (one forward pass, nothing changed in
the network). Model loading and hooks as in encoder_probe.py (part B): ViT, the cross-attention scores of the image
tokens over the day tokens (mean over heads and image tokens); U-Net, the L-TAE masks (mean over heads and the 4 x 4
bottleneck cells). Index 0 is the oldest day, index T-1 day -1.

    python attention_all.py --run-dir <runs_final>/final_t30/seed1/vit [--batch 64] [--threads 16]

Writes <run-dir>/inference_attention_all/attention.npz: rows, folder_name, patch_number, weather (N, T, 5) as the
network receives it, att (N, T), flood and scored cells of the 80 m label, predicted flood cells at the fit's threshold.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import yaml

import evaluate as ev                                                    # noqa: E402
from flood.paths import METADATA                                         # noqa: E402
from flood.data.dataset_final import FinalFloodPatches                    # noqa: E402
from encoder_probe import CAP, hook, hook_vit_interaction                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    run_dir = a.run_dir
    man = json.loads((run_dir / 'run_manifest.json').read_text())
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    family = man['family']
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
    net = getattr(model, 'backbone', model)
    if family == 'vit':
        hook_vit_interaction(net.interaction)
    else:
        hook(net.ltae, 'masks')

    kw = ev.final_dataset_kwargs(cfg, dp)
    kw.pop('mono_only')
    if not Path(kw['patches_dir']).exists():
        kw['patches_dir'] = os.environ['FLOOD_PATCHES_DIR']
    fp = FinalFloodPatches(cohort='test', allow_test=True, **kw)
    rows = np.arange(len(fp.df))
    W = fp._wx[rows, -fp.temporal_days:].astype(np.float32).copy()
    pool = ThreadPoolExecutor(a.threads)
    load = lambda idx: list(pool.map(lambda i: fp.load(int(i)), idx))
    att, flood, scored, pred = [], [], [], []
    t0 = time.time()
    batches = [rows[s:s + a.batch] for s in range(0, len(rows), a.batch)]
    nxt = pool.submit(load, batches[0])
    for b, idx in enumerate(batches):
        data = nxt.result()
        if b + 1 < len(batches):
            nxt = pool.submit(load, batches[b + 1])
        x = [tf.convert_to_tensor(np.stack([d[j] for d in data])) for j in range(3)]
        x.append(tf.convert_to_tensor(W[idx[0]:idx[-1] + 1]))
        out = model(tuple(x), training=False)
        p = np.asarray(tf.cast(out[0] if isinstance(out, (tuple, list)) else out, tf.float32))[..., 0]
        lab = np.stack([d[-1][..., 0] for d in data]).astype(np.int8)
        keep = lab >= 0
        if family == 'vit':
            att.append(np.asarray(tf.reduce_mean(CAP['att'], axis=[1, 2]), np.float32))
        else:
            att.append(np.asarray(tf.reduce_mean(CAP['masks'], axis=[1, 3, 4]), np.float32))
        flood.append((lab == 1).reshape(len(idx), -1).sum(1)); scored.append(keep.reshape(len(idx), -1).sum(1))
        pred.append(((p >= thr) & keep).reshape(len(idx), -1).sum(1))
        if b % 100 == 0:
            print(f'{b * a.batch:,}/{len(rows):,} rows ({time.time() - t0:.0f}s)', flush=True)
    out = run_dir / 'inference_attention_all'
    out.mkdir(exist_ok=True)
    df = fp.df.reset_index(drop=True)
    np.savez_compressed(out / 'attention.npz', rows=rows, folder_name=df.folder_name.to_numpy().astype(str),
                        patch_number=df.patch_number.to_numpy(), weather=W, att=np.concatenate(att),
                        flood=np.concatenate(flood), scored=np.concatenate(scored), pred=np.concatenate(pred),
                        threshold=thr)
    print(f'wrote {out}/attention.npz, {len(rows):,} rows ({time.time() - t0:.0f}s)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
