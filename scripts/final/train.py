#!/usr/bin/env python3
"""Entry point of the FINAL programme (METHOD_PLAN_RERUN.md v3): one script, both families.

    python train.py unet final_t0     --config ../../configs/final/t0_unet_s1.yaml --runs-dir <runs_final>
    python train.py vit  final_t30    --config ../../configs/final/t30_vit_s1.yaml --runs-dir <runs_final>
    python train.py unet final_window --config ../../configs/final/t7_unet_s1.yaml --runs-dir <runs_final>

Family and stage are declared on the command line AND in the config; core_final refuses a
mismatch, a temporal_days outside the stage, or a weather fit without AOI weather / U-TAE.
Optimisers and schedules are the tested ones of exp1_terrain: U-Net SGD + Nesterov with
nnU-Net poly decay; ViT AdamW with warm-up + linear decay.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # src/ on the path

import yaml                                                    # noqa: E402
from tensorflow import keras                                   # noqa: E402
from flood.training.core_final import run                      # noqa: E402
from flood.callbacks import PolyDecay, WarmupPolyDecay         # noqa: E402


def unet_backbone(cfg):
    from flood.models.static.unet import ResidualUNet
    return ResidualUNet(cfg)


def vit_backbone(cfg):
    from flood.models.static.vit import PlainViT
    return PlainViT(cfg)


def unet_optimizer(cfg):
    return keras.optimizers.SGD(learning_rate=float(cfg.get('learning_rate', 0.01)),
                                momentum=float(cfg.get('momentum', 0.99)), nesterov=True, clipnorm=1.0)


def vit_optimizer(cfg):
    return keras.optimizers.AdamW(learning_rate=float(cfg.get('learning_rate', 1.2e-4)),
                                  weight_decay=float(cfg.get('weight_decay', 0.05)), clipnorm=1.0)


def unet_lr(cfg):
    return PolyDecay(total_epochs=int(cfg.get('epochs', 100)), peak_lr=float(cfg.get('learning_rate', 0.01)),
                     power=float(cfg.get('poly_power', 0.9)))


def vit_lr(cfg):
    return WarmupPolyDecay(warmup_epochs=int(cfg.get('warmup_epochs', 5)), total_epochs=int(cfg.get('epochs', 100)),
                           peak_lr=float(cfg.get('learning_rate', 1.2e-4)), min_lr=float(cfg.get('min_lr', 1e-7)))


FAMILIES = {'unet': (unet_backbone, unet_optimizer, unet_lr), 'vit': (vit_backbone, vit_optimizer, vit_lr)}

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in FAMILIES:
        sys.exit(__doc__)
    family, stage, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    cfg_path = rest[rest.index('--config') + 1]
    declared = yaml.safe_load(open(cfg_path)).get('stage')
    if declared != stage:
        sys.exit(f'config declares stage {declared!r}, command line says {stage!r}')
    sys.exit(run(*FAMILIES[family], stage=stage, family=family, argv=rest))
