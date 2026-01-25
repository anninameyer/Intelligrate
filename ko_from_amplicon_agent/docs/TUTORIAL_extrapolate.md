# intelligrate.extrapolate Tutorial

## What extrapolate does (short)
`intelligrate.extrapolate` learns a KNN mapping from ASV k-mer profiles (X) to KO profiles (Y) using paired
amplicon–shotgun samples. It embeds X with CLR + SVD, optionally learns a diagonal metric, predicts in CLR
(Optionally with a latent SVD for Y), then converts to TSS. OOD diagnostics report nearest-neighbor distances
to the training set.

This tutorial covers installation, input formats, training, full fitting, prediction, and a full parameter
reference for the YAML config.

## Installation
Recommended: create a clean Python environment (venv or conda) with Python >= 3.10.
This package is pure Python and should work in any standard environment that can install the dependencies.

For end users (when published):
```
pip install intelligrate
```

For development from the repo:
```
pip install -e .
```

## Input data format
All input tables are TSV with sample IDs in the first column and features as headers.

Required files (see `data/` for examples):
- `X_kmers.tsv`: ASV k-mer features for paired samples (rows = samples, columns = k-mers)
- `X_kmers_full.tsv`: ASV k-mers for all samples (paired + unpaired)
- `Y_kos.tsv`: KO profiles (TPM or raw counts) for the paired samples
- `ko_to_superclass.tsv`: two-column mapping (KO -> pathway/superclass)

Optional:
- `picrust2_kos.tsv`: PICRUSt2 KO predictions for paired samples (for benchmarking)

## 1) Train (nested CV)
`make score` uses `configs/default.yaml`. This is the **fast** default; change `cv.outer_splits` and
`cv.inner_splits` to trade runtime for stability.

Why: training uses **nested CV** to select hyperparameters without leakage and produce robust out‑of‑fold
predictions for evaluation.

```
make score
# or
python -m intelligrate.extrapolate.train --config configs/default.yaml
```

Outputs written to `results/`:
- `oof_clr.tsv` and `oof_tss.tsv`: OOF predictions in CLR and TSS
- `folds.tsv`: per-fold best params + fold metrics
- `summary.json` and `summary.tsv`: overall metrics (includes PICRUSt2 benchmark when provided)
- `grid_results.tsv`: only when a grid sweep is used (see below)

Timestamped copies are also written for reproducibility.

Interpretation:
- Higher `OBJECTIVE_DM_SPEARMAN_MEAN` means predicted sample–sample relationships more closely match truth.
- `model_dm_union` is the KO‑union Spearman score; compare against `picrust2_dm_union` when available.

### Beginner-friendly example (explicit steps)
1) Start from the default config:
```
cp configs/default.yaml configs/my_run.yaml
```
2) Run training with your config:
```
python -m intelligrate.extrapolate.train --config configs/my_run.yaml
```
3) Check outputs in `results/` (`oof_clr.tsv`, `oof_tss.tsv`, `folds.tsv`, `summary.json`, `summary.tsv`).

### Hyperparameter tuning (grids)
All numerical config values can be swept via the optional `grid` section. Any entry there becomes a
hyperparameter sweep (cartesian product). The training script runs each combination and keeps the best
`OBJECTIVE_DM_SPEARMAN_MEAN` result as the final outputs.

Example grid:
```
grid:
  model:
    y_detect_threshold: [1000.0, 2000.0, 3000.0]
    neigh_k_grid: [[16, 20], [24, 28]]
  embed:
    n_components: [64, 128]
  cv:
    outer_splits: [5, 10]
```

### Extended example (grid + manual overrides)
This example shows how to (a) change a non-grid parameter (e.g., `pseudocount_y`) and (b) run a grid
for other values.

1) Copy the default config:
```
cp configs/default.yaml configs/my_grid.yaml
```
2) Edit `configs/my_grid.yaml` and update:
```
model:
  pseudocount_y: 0.000001

grid:
  model:
    y_detect_threshold: [1000.0, 2000.0, 3000.0]
    tau_mult_grid: [[0.5, 1.0], [2.0, 4.0]]
  embed:
    n_components: [64, 128]
```
3) Run:
```
python -m intelligrate.extrapolate.train --config configs/my_grid.yaml
```

Notes:
- Any value in `grid` can be a list. The trainer will try all combinations.
- Non-grid values (like `pseudocount_y` above) are treated as fixed for that entire sweep.

## 2) Full fit (final model)
Fit a final model on the paired data and save it with joblib.

Why: this trains a single deployable model on all paired samples, using fixed hyperparameters.

Step A: fit and save the embedding (one-time). For consistency with training, reuse the same parameters
from config:
```
python - <<'PY'
import joblib
import pandas as pd
from intelligrate.extrapolate.embedding import fit_x_embedding_svd_clr

X_full = pd.read_csv('data/X_kmers_full.tsv', sep='\t', index_col=0)
embed = fit_x_embedding_svd_clr(
    X_full,
    min_prev_x_abs=14,
    pseudocount_x=0.5,
    n_components=128,
    seed=0,
)
joblib.dump(embed, 'results/embed.joblib')
PY
```

Step B: fit the final model:
```
python -m intelligrate.extrapolate.full_fit \
  --x data/X_kmers.tsv \
  --y data/Y_kos.tsv \
  --embed-path results/embed.joblib \
  --model-out results/model.joblib \
  --neigh-k 12 \
  --tau-mult 2.0 \
  --lam 0.0 \
  --y-latent-k 10 \
  --use-metric-learning \
  --metric-ridge 2.5 \
  --metric-max-pairs 5000 \
  --ood-shrink \
  --ood-lam-base 0.15 \
  --ood-lam-cap 0.80
```

## 3) Full predict (new samples)
Predict KOs for any k-mer table (paired or unpaired). The command writes:
- `*.clr.tsv` (CLR predictions)
- `*.tss.tsv` (TSS predictions)
- `*.diag.tsv` (OOD NN distance diagnostics)

Why: use the trained model to extrapolate KOs to samples without shotgun data and inspect OOD diagnostics
to gauge how far predictions are from the training manifold.

Interpretation:
- Larger `ood_nn_min` suggests a sample is farther from the training manifold and predictions may be less reliable.

```
python -m intelligrate.extrapolate.full_predict \
  --model results/model.joblib \
  --x data/X_kmers_full.tsv \
  --out-prefix results/pred
```

### Optional evaluation on paired subset
If ground truth KOs exist for a subset, pass `--y-truth` to compute the same metrics as `train`:
```
python -m intelligrate.extrapolate.full_predict \
  --model results/model.joblib \
  --x data/X_kmers.tsv \
  --y-truth data/Y_kos.tsv \
  --out-prefix results/pred_paired
```
This writes `results/pred_paired.metrics.tsv`.

### Leakage-free evaluation on paired samples (recommended)
**Important:** the full-fit model is trained on *all paired samples*. Evaluating those same paired samples
with full-fit predictions is optimistic (leakage).

To get leakage-free paired predictions **with fixed hyperparameters**, run a single CV pass with fixed
params and replace the paired rows in your full prediction table. This is implemented in the API helper:
`fixed_param_oof_knn_on_embedding` (see the notebook).

Notes:
- Set `outer_splits = len(X)` to approximate leave-one-out (slow).
- The default `outer_splits` gives a faster, still leakage-free estimate.
- The notebook writes `oof_fixed_tss.tsv` and then **overwrites `pred_full.*`** so paired rows are OOF,
  while unpaired rows remain full-fit predictions.

## PICRUSt2 comparison
If `picrust2_kos.tsv` is provided in the config, `train` reports the same metrics for PICRUSt2 and logs:
- `picrust2_dm_union`
- `model_dm_union`
- `delta_union = model_dm_union - picrust2_dm_union`

Why: this provides a baseline comparison against PICRUSt2 under identical metrics.

## Config reference (all parameters)
Below is a concise reference for **every parameter in `configs/default.yaml`**.

### data
- `x_full`: filename for all ASV k-mer samples
- `x`: filename for paired ASV k-mers (subset of `x_full`)
- `y`: filename for paired KO table
- `picrust2`: optional PICRUSt2 KO table (paired samples)
- `ko_to_superclass`: KO -> pathway/superclass mapping

### cv
- `outer_splits`: outer CV folds (higher = slower, more stable)
- `inner_splits`: inner CV folds for hyperparameter selection
- `seed`: random seed for CV and SVDs
- `informed_splits`: use KMeans-informed splits (prevents target leakage by clustering X only)

### embed
- `min_prev_x_abs`: minimum prevalence for k-mers kept in embedding
- `pseudocount_x`: pseudocount for CLR in X embedding
- `n_components`: SVD components for X embedding

### model
- `min_prev_y_abs`: minimum prevalence for KO features retained per fold
- `y_detect_threshold`: absolute count threshold used in prevalence filtering
- `pseudocount_y`: pseudocount for CLR in Y
- `neigh_k_grid`: KNN neighborhood sizes (grid)
- `tau_mult_grid`: kernel width multiplier (grid)
- `lam_grid`: shrinkage toward global mean (grid)
- `y_latent_k_grid`: number of Y latent SVD components (grid; 0 = no latent)
- `use_metric_learning`: whether to learn diagonal metric in embedding space
- `metric_max_pairs`: max pairs for metric learning sampling
- `metric_ridge_grid`: ridge values for metric learning (grid)
- `ood_shrink`: enable OOD shrinkage
- `ood_shrink_inner`: also apply OOD shrinkage during inner CV
- `ood_lam_base`: base shrinkage coefficient for OOD
- `ood_lam_cap`: cap for OOD shrinkage
- `ood_tau_inflate`: inflate tau based on OOD distances

### objective
- `w_dm`: weight for primary objective (Aitchison DM Spearman)
- `w_wclr`: weight for weighted CLR MSE (optional)
- `w_pw_rmse`: weight for pathway RMSE (optional)
- `w_softf1`: weight for thresholded F1 (optional)
- `w_jsd`: weight for JSD (optional)

### prf
- `prf_thresh`: threshold for presence/absence metrics in TSS
- `prf_weight`: weighting scheme (`binary`, `truth_abundance`, `pred_abundance`)

### metrics (optional reporting only)
- `compute_wclr`: compute weighted CLR MSE
- `compute_jsd`: compute Jensen-Shannon divergence
- `compute_pathway_rmse`: compute pathway RMSE (overall)
- `pathway_rmse_per_group`: compute per-pathway RMSE table
- `pathway_rmse_log1p`: log1p transform for pathway RMSE

### score
- `min_prev_y_abs`: prevalence filter for the optional global OOF check
- `y_detect_threshold`: detection threshold for that check
- `pseudocount_y`: CLR pseudocount for that check

### grid (optional)
Any numeric config value can be swept via:
```
grid:
  <section>:
    <parameter>: [values...]
```
The trainer runs the cartesian product of all grid values and keeps the best run.

## Notes for beginners
- KO-union evaluation compares predictions to truth over the union of KOs, then computes CLR and Aitchison distances.
- The OOD diagnostics in `*.diag.tsv` help identify samples that are far from the training manifold.
- Use the default config for quick iteration; increase `cv.outer_splits` and `cv.inner_splits` for more robust scoring.
