# FloodPULSEO-model-training-pipeline

Training, evaluation and analysis code of the PhD thesis *Predicting Flood Extent with Artificial Intelligence from Land Surface Characteristics and Daily Hydroclimatic Inputs* (Ravi Kumar, Sapienza University of Rome, 2026).

A residual U-Net and a Vision Transformer (ViT) predict binary flood extent at 80 m from land surface characteristics (T0) and from land surface characteristics with input histories of 3, 7, 15 and 30 days of precipitation and soil moisture (T3 to T30). The networks are trained on the FloodPULSEO dataset. The code that builds the dataset is in [FloodPULSEO-pipeline](https://github.com/ravi-kumar-eo/FloodPULSEO-pipeline).

## Contents

| Folder | Content |
|---|---|
| `flood/` | Python package: patch loading, residual U-Net, Vision Transformer, hydroclimatic encoders (day encoder, temporal convolutions, U-TAE masks, gated cross-attention), patch-level threshold head, losses, metrics and training loop |
| `configs/` | Configurations of the 38 training runs of the thesis: T0, T3, T7, T15 and T30 for both architectures and seeds 1–3, and the four T30 ablations of seed 1; `_common.yaml` holds the settings shared by all runs |
| `scripts/` | Training (`train.py`), evaluation (`evaluate.py`), training curves (`plot_run.py`), perturbed hydroclimatic inputs (`perturb_eval.py`), encoder probes (`encoder_probe.py`), attention weights (`attention_all.py`), block-removal evaluation (`block_removal_eval.py`) and the analysis of the patch-level threshold head (`cache_head_features.py`, `head_sweep.py`) |
| `build/` | Builders of the spatial blocks, the training, validation and test split, the 80 m flood labels, the input transforms and the upstream precipitation of each patch |
| `tests/` | Regression tests and data checks run before training |
| `baselines/gff/` | Comparison with the Global Flood Forecasting (GFF) models at expert-labelled sites |
| `baselines/lisflood_fp/` | Comparison with the LISFLOOD-FP event footprints of Scarpellini et al. (2026) |
| `thesis_analysis/chapter5/` | Scripts that compute the results of Chapter 5 and draw its figures |

## Requirements

The thesis runs used Python 3.12, TensorFlow 2.21 and Keras 3.15 on one NVIDIA RTX PRO 6000 GPU with 96 GB of memory.

```bash
pip install -r requirements.txt
```

## Data

The FloodPULSEO dataset is deposited on the Harvard Dataverse, [doi:10.7910/DVN/A7X9JD](https://doi.org/10.7910/DVN/A7X9JD) (the DOI resolves after the publication of the record).

- `patches_dir` in `configs/_common.yaml` points to the folder of the patch GeoTIFFs.
- The environment variable `FLOOD_METADATA_DIR` points to the metadata folder. Without it, the code uses `metadata/` next to the folder of this repository (`flood/paths.py`).
- The scripts in `build/` write the tables of `metadata/final/` (split, 80 m labels, hydroclimatic inputs, input transforms) from the dataset.

## Training and evaluation

```bash
# T0: land surface characteristics only
python scripts/train.py unet final_t0 --config configs/t0_unet_s1.yaml --runs-dir runs_final
# T30: starts from the T0 checkpoint of the same architecture and seed
python scripts/train.py vit final_t30 --config configs/t30_vit_s1.yaml --runs-dir runs_final
# T3, T7, T15
python scripts/train.py unet final_window --config configs/t7_unet_s1.yaml --runs-dir runs_final

# evaluation at the validation-selected global threshold
python scripts/evaluate.py --run-dir runs_final/final_t30/seed1/unet --with-test
```

Each run folder stores its configuration, a copy of the code it ran (`code/`, `code_manifest.json`), the training log and a run manifest with the data version, the parent checkpoint and the software environment.

## Citation

Kumar, R. (2026). *Predicting Flood Extent with Artificial Intelligence from Land Surface Characteristics and Daily Hydroclimatic Inputs*. PhD thesis, Sapienza University of Rome.

## License

MIT License, see [LICENSE](LICENSE).
