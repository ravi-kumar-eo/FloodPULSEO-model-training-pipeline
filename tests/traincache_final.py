#!/usr/bin/env python3
"""Training RAM cache: bit-identical to the disk path, and how much faster per row.

    python traincache_final.py --budget-gb 0.5
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml

from flood.paths import METADATA                                             # noqa: E402
from flood.data.dataset_final import FinalFloodPatches, load_transforms        # noqa: E402
from flood.training.core_final import data_paths, resolve_patches_dir          # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--budget-gb', type=float, default=0.5)
    a = ap.parse_args()
    D = Path(__file__).resolve().parents[1] / 'configs' / 'final'
    cfg = yaml.safe_load(open(D / '_common.yaml')); cfg.update(yaml.safe_load(open(D / 't30_unet_s1.yaml')))
    cfg['patches_dir'] = resolve_patches_dir(cfg)
    dp = data_paths(cfg, Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA))
    fp = FinalFloodPatches(cohort='train', patches_csv=dp['split'] / 'patches.csv', patches_dir=cfg['patches_dir'],
                           labels_dir=dp['labels'], flood_flags_csv=dp['split'] / 'flood80_flags.csv',
                           temporal_days=30, transforms=load_transforms(dp['transforms']),
                           aoi_weather_dir=dp['aoi_weather'])
    fp.preload_labels(verbose=False)
    rows = fp.train_cache_rows(a.budget_gb, seed=1)
    print(f'budget {a.budget_gb} GB -> {len(rows):,} rows, all flood-bearing: {bool(fp.has_flood[rows].all())}')
    fp.enable_eval_cache(workers=24, rows=rows, tag='traincache')
    ok = True
    for i in rows[:200]:
        c = fp.load(int(i)); u = fp._load_uncached(int(i))
        ok &= all(np.array_equal(x, y) for x, y in zip(c, u))
    print('cached == disk on 200 cached rows:', ok)
    other = np.setdiff1d(np.arange(len(fp)), rows)[:100]
    ok2 = all(int(i) not in fp._cache_pos for i in other)
    print('uncached rows still read from disk:', ok2)
    t = time.perf_counter(); [fp.load(int(i)) for i in rows[:300]]; tc = (time.perf_counter() - t) / 300
    t = time.perf_counter(); [fp.load(int(i)) for i in other[:100]]; tu = (time.perf_counter() - t) / 100
    print(f'per row, one thread: cached {1000 * tc:.2f} ms, disk {1000 * tu:.2f} ms ({tu / tc:.0f}x)')
    for s in fp._shm.values():
        s.close(); s.unlink()
    print('RESULT:', 'PASS' if ok and ok2 else 'FAIL')


if __name__ == '__main__':
    main()
