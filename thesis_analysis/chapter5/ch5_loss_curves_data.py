"""Loss curves of all 38 fits (Ravi, 19 Oct 2026: loss curves in the thesis, T0 in Section 5.1, T30 in Section 5.3,
the other lengths and the ablations in Appendix A).

Reads, read-only, every fit's training_log.csv (one row per epoch, 1-based) and run_manifest.json under
experiments/runs_final/ and writes plot/data/training/:
  curves.csv  config, family, seed, epoch, train_loss (seg_loss, the training segmentation loss of Equation 4.5),
              val_loss (val_seg_loss, the same loss on the validation subset, the early-stopping metric) and val_iou
              (val_flood_iou_best, the pooled flood-only validation IoU at the best of 49 thresholds, the selection
              metric of Section 4.5.3)
  runs.csv    config, family, seed, epochs_run, selected_epoch, lowest_val_loss_epoch, warmup_epochs, hours
Checks: the selected epoch is the epoch of the highest val_iou after the warm-up, the lowest-loss epoch that of the
lowest val_loss after the warm-up, and the number of log rows equals epochs_run, all as stored in the manifest.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_loss_curves_data.py
"""
import json
from pathlib import Path

import pandas as pd

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'training'
FAMILIES = ('unet', 'vit')


def fit_dirs():
    """(config, family, seed, run directory) of the 38 fits of Table 4.4."""
    for s in (1, 2, 3):
        for f in FAMILIES:
            yield 't0', f, s, RUNS / 'final_t0' / f'seed{s}' / f
            yield 't30', f, s, RUNS / 'final_t30' / f'seed{s}' / f
            for w in ('t3', 't7', 't15'):
                yield w, f, s, RUNS / 'final_window' / f'seed{s}' / f'{f}_{w}'
    for f in FAMILIES:
        for a in ('noup', 'nosum', 'sumonly'):
            yield a, f, 1, RUNS / 'final_ablation' / 'seed1' / f'{f}_{a}'
        yield 'scratch', f, 1, RUNS / 'final_scratch' / 'seed1' / f


def main():
    curves, runs = [], []
    for config, f, s, d in fit_dirs():
        log = pd.read_csv(d / 'training_log.csv')
        man = json.loads((d / 'run_manifest.json').read_text())
        sel = man['selection']
        warm = int(man.get('warmup_epochs', 0) or 0)
        # the warm-up gate of Section 4.5.3: ViT T0 and ViT random init only
        if 'warmup_epochs' not in man:
            warm = 5 if (f == 'vit' and config in ('t0', 'scratch')) else 0
        eligible = log[log.epoch > warm]
        assert len(log) == sel['epochs_run'], (d, len(log), sel['epochs_run'])
        assert int(eligible.loc[eligible.val_flood_iou_best.idxmax(), 'epoch']) == sel['selected_epoch'], d
        assert int(eligible.loc[eligible.val_seg_loss.idxmin(), 'epoch']) == sel['lowest_val_seg_loss_epoch'], d
        curves.append(pd.DataFrame({'config': config, 'family': f, 'seed': s, 'epoch': log.epoch,
                                    'train_loss': log.seg_loss, 'val_loss': log.val_seg_loss,
                                    'val_iou': log.val_flood_iou_best}))
        runs.append(dict(config=config, family=f, seed=s, epochs_run=sel['epochs_run'],
                         selected_epoch=sel['selected_epoch'], lowest_val_loss_epoch=sel['lowest_val_seg_loss_epoch'],
                         warmup_epochs=warm, hours=man.get('hours')))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(curves, ignore_index=True).to_csv(OUT / 'curves.csv', index=False)
    r = pd.DataFrame(runs)
    r.to_csv(OUT / 'runs.csv', index=False)
    assert len(r) == 38, len(r)
    print(r.to_string(index=False))


if __name__ == '__main__':
    main()
