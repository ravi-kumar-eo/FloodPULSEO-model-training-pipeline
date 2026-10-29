"""Section 5.8 — Training time of every training run (Ravi, 20 Oct 2026: training time in its own subsection).

Reads run_manifest.json of the 38 final training runs (experiments/runs_final/final_{t0,t30,window,ablation,scratch}):
elapsed hours of the run, epochs run, GPU and host, parameters, and writes one row per run with the minutes per epoch
(hours x 60 / epochs run). Times are wall-clock times of one GPU and include validation after every epoch.

Output  plot/data/training/time.csv

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_training_time_data.py
"""
import json
from pathlib import Path

import pandas as pd

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'training' / 'time.csv'


def main():
    rows = []
    for stage in ('final_t0', 'final_t30', 'final_window', 'final_ablation', 'final_scratch'):
        for m in sorted((RUNS / stage).glob('seed*/*/run_manifest.json')):
            j = json.loads(m.read_text())
            env, sel = j.get('environment', {}), j.get('selection', {})
            name = m.parent.name                                   # e.g. unet, vit_t30, unet_t3, vit_noup
            config = {'final_t0': 't0', 'final_t30': 't30', 'final_scratch': 'scratch'}.get(
                stage, name.split('_', 1)[1] if '_' in name else 't0')
            rows.append({'stage': stage, 'family': j['family'], 'config': config, 'seed': j['seed'],
                         'host': env.get('compute') or env.get('host'), 'gpu': env.get('gpu', '').split(',')[0],
                         'hours': j['hours'], 'epochs_run': sel.get('epochs_run'),
                         'params': j.get('params'), 'steps_per_epoch': j.get('steps_per_epoch')})
    d = pd.DataFrame(rows)
    d['min_per_epoch'] = d.hours * 60 / d.epochs_run
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.to_string(index=False))
    print('total hours', round(d.hours.sum(), 1), len(d))


if __name__ == '__main__':
    main()
