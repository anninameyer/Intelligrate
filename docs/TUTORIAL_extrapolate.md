# intelligrate.extrapolate Tutorial

See also:
- `../README.md` (project overview + quickstart)
- `TUTORIAL_subset.md` (sample subsetting workflow)
- `notebooks/02_extrapolate_train_evaluate_full_fit_predict_HF_sourdough.ipynb` (hands-on sourdough run)

This tutorial explains the extrapolate workflow in two parallel ways:
- [Python API workflow](#python-api-workflow-recommended)
- [CLI workflow](#cli-workflow-same-actions)

The API and CLI expose the same core actions:
1. `train`: leakage-aware nested CV, hyperparameter selection, OOF predictions, and metrics.
2. `fixed-param-sweep`: optional stability check for fixed hyperparameter choices.
3. `full-fit`: fit one final model on all paired samples.
4. `full-predict`: predict target profiles for samples with only starting-layer features.

---

## What Extrapolate Does

![Intelligrate overview](assets/extrapolate.png)

`intelligrate.extrapolate` learns a model that maps a starting data layer `X` to a target data
layer `Y`. In the example notebooks, `X` is an amplicon-derived k-mer or ASV table and `Y` is a
shotgun-derived KO, EC, or pathway table.

The workflow:
1. embeds `X` with CLR + SVD,
2. learns a k-nearest-neighbor model in embedded `X` space,
3. predicts target profiles in CLR space,
4. converts predictions back to TSS/relative-abundance space,
5. optionally evaluates predictions against paired truth or a PICRUSt2 baseline.

---

## Inputs And Outputs

Example datasets live under `data/` in the GitHub repository. The pip package contains the library;
notebooks and example data are downloaded from GitHub separately.

Recommended layout for the sourdough example:

```text
your_working_folder/
  02_extrapolate_train_evaluate_full_fit_predict_HF_sourdough.ipynb
  data/HF_sourdough/
    X_kmers.tsv
    X_kmers_full.tsv
    Y_kos.tsv
    picrust2_kos.tsv          # optional baseline comparison
    ko_to_superclass.tsv      # optional pathway/superclass summaries
  results/
```

Required input tables:
- `X_kmers.tsv` or `X_ASVs.tsv`: paired starting-layer table, samples x features.
- `X_kmers_full.tsv` or `X_ASVs_full.tsv`: full starting-layer table, samples x features.
- `Y_kos.tsv`, `Y_ecs.tsv`, or `Y_pwys.tsv`: paired target table, samples x features.

Optional input tables:
- `picrust2_kos.tsv` or `picrust2_ecs.tsv`: baseline predictions for comparison only.
- `ko_to_superclass.tsv` or `pwy_to_superclass.tsv`: feature-to-group mapping for pathway/superclass summaries only.

PICRUSt2 and mapping files are not required for training, fitting, or prediction.

Main outputs:
- `oof_clr.tsv`, `oof_tss.tsv`: out-of-fold predictions from nested CV.
- `folds.tsv`: per-fold metrics and selected parameters.
- `summary.json`, `summary.tsv`: overall training/evaluation summary.
- `model.joblib`: final fitted model from `full-fit`.
- `pred.clr.tsv`, `pred.tss.tsv`, `pred.diag.tsv`: full-prediction outputs.
- `pred.metrics.tsv`: optional evaluation metrics if truth is provided during prediction.

---

## Installation

Use Python 3.10-3.12 on macOS, Linux, or Windows.

Using conda or mamba:

```bash
conda create -n intelligrate python=3.11
conda activate intelligrate
pip install intelligrate
```

Using `venv` on macOS/Linux:

```bash
python -m venv intelligrate-env
source intelligrate-env/bin/activate
python -m pip install --upgrade pip
pip install intelligrate
```

Using `venv` on Windows PowerShell:

```powershell
py -m venv intelligrate-env
.\intelligrate-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install intelligrate
```

Verify the install:

```bash
python -c "import intelligrate; print('intelligrate import OK')"
intelligrate --help
intelligrate extrapolate --help
intelligrate extrapolate write-config --help
```

For notebooks, install the notebook extras in the same environment:

```bash
pip install "intelligrate[notebooks]"
```

Clone the repository only if you want to modify Intelligrate itself:

```bash
git clone https://github.com/anninameyer/Intelligrate.git
cd Intelligrate
pip install -e ".[dev]"
```

---

## Python API Workflow (Recommended)

The API workflow is useful in notebooks and scripts because you can inspect intermediate objects,
edit parameters directly, and write outputs wherever you want.

### Setup

```python
from pathlib import Path
import joblib
import pandas as pd

from intelligrate.extrapolate import (
    evaluate_paired_subset,
    fit_final_model,
    fit_x_embedding_svd_clr,
    predict_final_model,
    run_fixed_param_sweep_explicit,
    run_training_config,
    save_model,
)

project_dir = Path.cwd()
data_dir = project_dir / "data" / "HF_sourdough"
results_dir = project_dir / "results" / "HF_sourdough"
results_dir.mkdir(parents=True, exist_ok=True)

X_full = pd.read_csv(data_dir / "X_kmers_full.tsv", sep="\t", index_col=0)
X = pd.read_csv(data_dir / "X_kmers.tsv", sep="\t", index_col=0)
Y = pd.read_csv(data_dir / "Y_kos.tsv", sep="\t", index_col=0)
```

### Action 1: Train

`train` runs nested CV. Use it to estimate performance without leakage and to get OOF predictions.

```python
cfg = {
    "data": {
        "x_full": "HF_sourdough/X_kmers_full.tsv",
        "x": "HF_sourdough/X_kmers.tsv",
        "y": "HF_sourdough/Y_kos.tsv",
        "picrust2": "HF_sourdough/picrust2_kos.tsv",  # optional
        "ko_to_superclass": "HF_sourdough/ko_to_superclass.tsv",  # optional
    },
    "cv": {"outer_splits": 5, "inner_splits": 3, "seed": 0, "informed_splits": False},
    "embed": {"min_prev_x_abs": 30, "pseudocount_x": 0.5, "n_components": 128},
    "model": {
        "min_prev_y_abs": 1,
        "y_detect_threshold": 1000.0,
        "pseudocount_y": 0.5 / 1e6,
        "neigh_k_grid": [20, 24, 28, 32],
        "tau_mult_grid": [0.5, 1.0, 2.0, 4.0],
        "lam_grid": [0.0],
        "y_latent_k_grid": [0, 10, 20],
        "use_metric_learning": True,
        "metric_max_pairs": 2000,
        "metric_ridge_grid": [1.0, 2.5, 5.0],
        "tau_scale_k_nn": 5,
        "ood_shrink": True,
        "ood_shrink_inner": True,
        "ood_lam_base": 0.7,
        "ood_lam_cap": 0.5,
        "ood_tau_inflate": False,
    },
    "objective": {"w_dm": 1.0, "w_wclr": 0.0, "w_pw_rmse": 0.0, "w_softf1": 0.0, "w_jsd": 0.0},
    "prf": {"prf_thresh": 1.0e-6, "prf_weight": "binary"},
    "metrics": {
        "compute_wclr": False,
        "compute_jsd": False,
        "compute_pathway_rmse": False,
        "pathway_rmse_per_group": False,
        "pathway_rmse_log1p": True,
    },
    "score": {"min_prev_y_abs": 1, "y_detect_threshold": 1.0, "pseudocount_y": 0.5 / 1e6},
}

payload = run_training_config(cfg, data_dir=project_dir / "data", out_dir=results_dir / "train")
summary = payload["run"]
summary["model_dm_union_strict"]
```

Important returned objects:
- `payload["oof_clr"]`: OOF CLR predictions.
- `payload["oof_tss"]`: OOF TSS predictions.
- `payload["folds"]`: per-fold metrics and selected parameters.
- `payload["run"]`: summary metrics.

The same tables are written under `results/HF_sourdough/train/`.

### Action 2: Fixed-Parameter Sweep

This optional action evaluates fixed hyperparameter combinations with OOF predictions. Use it when
you want one stable parameter set before fitting the final model.

```python
embed = fit_x_embedding_svd_clr(
    X_full,
    min_prev_x_abs=30,
    pseudocount_x=0.5,
    n_components=128,
    seed=0,
)

sweep_df = run_fixed_param_sweep_explicit(
    X_full=X_full,
    X=X,
    Y=Y,
    ko_to_superclass=None,
    out_path=results_dir / "fixed_param_sweep.tsv",
    cv_cfg=cfg["cv"],
    embed_cfg=cfg["embed"],
    model_cfg=cfg["model"],
    prf_cfg=cfg["prf"],
    metrics_cfg=cfg["metrics"],
    sweep_cfg={
        "neigh_k": [20, 24, 28],
        "tau_mult": [0.5, 1.0],
        "y_latent_k": [10, 20],
        "metric_ridge": [1.0, 2.5],
    },
    embed=embed,
)

sweep_df.head()
```

Use the top-ranked row as a candidate final parameter set.

### Action 3: Full Fit

`full-fit` trains one final deployable model on all paired samples.

```python
model = fit_final_model(
    X_train=X,
    Y_train_tpm=Y,
    embed=embed,
    min_prev_y_abs=1,
    y_detect_threshold=1000.0,
    pseudocount_y=0.5 / 1e6,
    neigh_k=24,
    tau_mult=1.0,
    lam=0.0,
    y_latent_k=10,
    use_metric_learning=True,
    metric_ridge=2.5,
    metric_max_pairs=2000,
    tau_scale_k_nn=5,
    ood_shrink=True,
    ood_lam_base=0.7,
    ood_lam_cap=0.5,
    seed=0,
)

save_model(model, results_dir / "model.joblib")
joblib.dump(embed, results_dir / "embed.joblib")
```

### Action 4: Full Predict

`full-predict` predicts target profiles for any compatible `X` table.

```python
pred_clr, pred_tss, pred_diag = predict_final_model(X_full, model)

pred_clr.to_csv(results_dir / "pred.clr.tsv", sep="\t")
pred_tss.to_csv(results_dir / "pred.tss.tsv", sep="\t")
pred_diag.to_csv(results_dir / "pred.diag.tsv", sep="\t")
```

Optional paired evaluation:

```python
metrics = evaluate_paired_subset(
    truth_tpm=Y,
    pred_tss=pred_tss.loc[Y.index],
    pseudocount=0.5 / 1e6,
    detect_threshold=1000.0,
    prf_thresh=1.0e-6,
    prf_weight="binary",
)

pd.DataFrame([metrics]).to_csv(results_dir / "pred.metrics.tsv", sep="\t", index=False)
metrics["dm_union_strict"]
```

---

## CLI Workflow (Same Actions)

Use the CLI when you prefer file-based runs. The CLI is organized hierarchically:

```bash
intelligrate extrapolate --help
```

### Write A Config Template

The config-driven actions use YAML. Start from the installed template:

```bash
intelligrate extrapolate write-config --out configs/default.yaml
```

Edit `configs/default.yaml` so the `data:` paths point to your downloaded example data or your own
tables. Paths in `data:` are interpreted relative to `data/` for `train` and
`fixed-param-sweep`.

### Action 1: Train

```bash
intelligrate extrapolate train --config configs/default.yaml
```

Outputs are written under `results/`:
- `oof_clr.tsv`
- `oof_tss.tsv`
- `folds.tsv`
- `summary.json`
- `summary.tsv`

### Action 2: Fixed-Parameter Sweep

Edit the `fixed_param_sweep:` block in `configs/default.yaml`, then run:

```bash
intelligrate extrapolate fixed-param-sweep \
  --config configs/default.yaml \
  --out results/fixed_param_sweep.tsv
```

The output TSV has one row per parameter combination and is sorted by `dm_union_strict`.

### Action 3: Full Fit

First fit and save the X embedding:

```bash
python - <<'PY'
from pathlib import Path
import joblib
import pandas as pd
from intelligrate.extrapolate import fit_x_embedding_svd_clr

Path("results").mkdir(exist_ok=True)
X_full = pd.read_csv("data/HF_sourdough/X_kmers_full.tsv", sep="\t", index_col=0)
embed = fit_x_embedding_svd_clr(
    X_full,
    min_prev_x_abs=30,
    pseudocount_x=0.5,
    n_components=128,
    seed=0,
)
joblib.dump(embed, "results/embed.joblib")
PY
```

Then fit the final model:

```bash
intelligrate extrapolate full-fit \
  --x data/HF_sourdough/X_kmers.tsv \
  --y data/HF_sourdough/Y_kos.tsv \
  --embed-path results/embed.joblib \
  --model-out results/model.joblib \
  --min-prev-y-abs 1 \
  --y-detect-threshold 1000.0 \
  --pseudocount-y 0.0000005 \
  --neigh-k 24 \
  --tau-mult 1.0 \
  --lam 0.0 \
  --y-latent-k 10 \
  --use-metric-learning \
  --metric-ridge 2.5 \
  --metric-max-pairs 2000 \
  --tau-scale-k-nn 5 \
  --ood-shrink \
  --ood-lam-base 0.7 \
  --ood-lam-cap 0.5
```

### Action 4: Full Predict

Predict the target layer for all samples:

```bash
intelligrate extrapolate full-predict \
  --model results/model.joblib \
  --x data/HF_sourdough/X_kmers_full.tsv \
  --out-prefix results/pred
```

Outputs:
- `results/pred.clr.tsv`
- `results/pred.tss.tsv`
- `results/pred.diag.tsv`

Optional evaluation when `Y` truth is available for a subset:

```bash
intelligrate extrapolate full-predict \
  --model results/model.joblib \
  --x data/HF_sourdough/X_kmers.tsv \
  --y-truth data/HF_sourdough/Y_kos.tsv \
  --out-prefix results/pred_paired
```

This also writes `results/pred_paired.metrics.tsv`.

---

## How To Interpret Metrics

Primary metric:
- `model_dm_union_strict`: Spearman correlation between sample-sample Aitchison distance matrices
  for truth and prediction, computed on the union of target features after detection thresholding.

Useful related metrics:
- `model_dm_union_raw`: same distance-matrix metric without detection thresholding.
- `model_bray_union_strict`: Bray-Curtis distance-matrix Spearman on TSS predictions.
- `model_procrustes_aitchison_strict`: Procrustes correlation in Aitchison/CLR space.
- `model_soft_precision`, `model_soft_recall`, `model_soft_f1`: thresholded feature detection metrics.
- `picrust2_*`: same metrics for PICRUSt2 if an optional PICRUSt2 table is provided.
- `delta_union`: model `dm_union_strict` minus PICRUSt2 `dm_union_strict`.

Metric set definitions:
- `union_raw`: truth and prediction feature union, missing features filled with zero, no detection threshold.
- `union_strict`: truth and prediction feature union, missing features filled with zero, detection threshold applied.
- `intersection`: features present in both truth and prediction.

Leakage note: full-fit predictions on the same paired samples used for training are optimistic.
Use `train` OOF predictions or fixed-parameter OOF predictions when you need leakage-free paired
evaluation.

---

## Parameter Guide

Training and sweep parameters:
- `outer_splits`, `inner_splits`: more CV folds are slower but can be more stable.
- `min_prev_x_abs`: higher values keep fewer starting-layer features.
- `n_components`: SVD dimensions for the `X` embedding.
- `min_prev_y_abs`: minimum prevalence for target features.
- `y_detect_threshold`: detection threshold before target prevalence filtering and strict metrics.
- `neigh_k`: k-nearest-neighbor neighborhood size.
- `tau_mult`: kernel bandwidth multiplier.
- `lam`: shrinkage toward the training-set mean.
- `y_latent_k`: target-layer latent dimensions; `0` predicts CLR directly.
- `use_metric_learning`, `metric_ridge`, `metric_max_pairs`: supervised diagonal metric learning options.
- `ood_shrink`, `ood_lam_base`, `ood_lam_cap`: shrink predictions for samples far from the training set.
- `prf_thresh`, `prf_weight`: threshold and weighting scheme for precision/recall/F1.

Optional metric toggles:
- `compute_wclr`: weighted CLR MSE.
- `compute_jsd`: Jensen-Shannon divergence.
- `compute_pathway_rmse`: grouped/pathway RMSE; requires a mapping file.
- `pathway_rmse_per_group`: per-group RMSE table; requires a mapping file.

---

## Config Reference

Run:

```bash
intelligrate extrapolate write-config --out configs/default.yaml
```

Then inspect and edit:

```yaml
data:
  x_full: "HF_sourdough/X_kmers_full.tsv"
  x: "HF_sourdough/X_kmers.tsv"
  y: "HF_sourdough/Y_kos.tsv"
  picrust2: "HF_sourdough/picrust2_kos.tsv"              # optional
  ko_to_superclass: "HF_sourdough/ko_to_superclass.tsv"  # optional
```

The full template also includes `cv`, `embed`, `model`, `objective`, `prf`, `metrics`, `score`,
and `fixed_param_sweep` sections.

See command-specific help for all CLI arguments:

```bash
intelligrate extrapolate train --help
intelligrate extrapolate fixed-param-sweep --help
intelligrate extrapolate full-fit --help
intelligrate extrapolate full-predict --help
```

See also:
- `../README.md`
- `TUTORIAL_subset.md`
