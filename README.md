# intelligrate

**Design representative data subsets** (e.g., for shotgun sequencing) **and extrapolate the follow-up results back to the full cohort** (e.g., predict KO profiles from amplicon k-mers) — one Python package, two workflows.

## What Intelligrate does

### 1) `subset`
Select a *diversity- and metadata-balanced* subset of samples that still represents your full dataset well.

**Typical use case:** choose samples for, e.g., more cost- and/or time-expensive follow-up assays (shotgun, metabolomics, isolate screening, long-reads).


### 2) `extrapolate`
Learn a mapping from a *starting layer* (e.g., amplicon marker gene k-mers) to a *follow-up layer* (e.g., shotgun-derived KO profiles) on a subset of paired samples, then predict follow-up profiles for the full dataset.

**Typical use case:** infer KO profiles for all samples when only a subset has shotgun sequencing data.
 
 ![Intelligrate overview](docs/assets/intelligrate_overview.png)
---

## Why these two workflows belong together

1. **Sometimes, you cannot follow up everything.** `subset` helps you pick the most informative samples under cost/time/capacity constraints.  
2. **You still want cohort-wide insights.** `extrapolate` uses the paired subset to predict follow-up profiles for the full dataset.  
3. **End-to-end reproducibility.** The workflows allow you to reproducibly select representative subsets and extrapolate findings from the subsets back to the full dataset through transparent mapping-fine-tuning and evaluation of mapping performance.

---

## Table of contents
- [Install](#install)
- [Quickstart: subset](#quickstart-subset)
- [Quickstart: extrapolate](#quickstart-extrapolate)
- [Tutorials and notebooks](#tutorials-and-notebooks)
- [Input formats](#input-formats)
- [Outputs](#outputs)
- [Parameter guide](#parameter-guide)

---

## Install

### From a GitHub tag
```bash
pip install "intelligrate @ git+https://github.com/ORG/REPO.git@vX.Y.Z"
```

## Quickstart: subset
Full runnable example in the provided [Tutorials and notebooks](#tutorials-and-notebooks).

**Rationale:** pick a subset that is (i) diverse in feature space, (ii) spatially spread (optional), and (iii) balanced across key metadata.

```
python - <<'PY'
import pandas as pd
from intelligrate.subset import compute_distance_matrix, suggest_k, fit_kmedoids, ga_subset

ft = pd.read_csv("data/feature_table_rel.tsv", sep="\t", index_col=0)  # samples x features
md = pd.read_csv("data/metadata.tsv", sep="\t", index_col=0)           # samples x metadata

# 1) distances
D = compute_distance_matrix(ft, metric="bray", assume_relative=True)

# 2) (optional) pick a reasonable k via diagnostics
_ = suggest_k(D, ft, k_range=range(2, 8), gap_B=3, random_state=42)

# 3) cluster and run GA subset selection
kmed = fit_kmedoids(D, k=5, random_state=42)

selected, best_scores, fitness = ga_subset(
    cluster_df=kmed["cluster_df"],
    metadata_df=md,
    total_samples=30,
    balance_vars=["r_samp_country", "r_samp_source", "hub"],
    coord_vars=("latitude", "longitude"),
    population_size=30,
    generations=20,
    random_state=42,
)

selected.to_csv("results/subset/ga_selected_samples.tsv", sep="\t")
PY

```

## Quickstart: extrapolate
Full runnable example in the provided [Tutorials and notebooks](#tutorials-and-notebooks).

**Rationale:** fit on paired samples (X → Y), then predict Y for all samples with only X (e.g., predict KO profiles from marker gene amplicons).

```
python - <<'PY'
import joblib
import pandas as pd
from intelligrate.extrapolate.embedding import fit_x_embedding_svd_clr
from intelligrate.extrapolate.full_fit import fit_final_model, save_model
from intelligrate.extrapolate.full_predict import predict_final_model

# X_full: all samples with amplicon k-mers
# X: paired subset (same feature space), matched to Y rows
# Y: KO profiles (TPM/TSS) for paired subset
X_full = pd.read_csv("data/X_kmers_full.tsv", sep="\t", index_col=0)
X      = pd.read_csv("data/X_kmers.tsv",      sep="\t", index_col=0)
Y      = pd.read_csv("data/Y_kos.tsv",        sep="\t", index_col=0)

# 1) fit reusable X embedding on the full X feature space
embed = fit_x_embedding_svd_clr(
    X_full,
    min_prev_x_abs=14,
    pseudocount_x=0.5,
    n_components=128,
    seed=0,
)
joblib.dump(embed, "results/embed.joblib")

# 2) fit final model on paired subset
model = fit_final_model(
    X_train=X,
    Y_train_tpm=Y,
    embed=embed,
    min_prev_y_abs=1,
    y_detect_threshold=3000.0,
    pseudocount_y=0.5 / 1e6,
    neigh_k=24,
    tau_mult=1.0,
    lam=0.0,
    y_latent_k=10,
    use_metric_learning=True,
    metric_ridge=2.5,
    metric_max_pairs=5000,
    tau_scale_k_nn=10,
    ood_shrink=True,
    ood_lam_base=0.7,
    ood_lam_cap=0.5,
    seed=0,
)
save_model(model, "results/model.joblib")

# 3) predict for all samples in X_full
Yhat_clr, Yhat_tss, diag = predict_final_model(X_full, model)
Yhat_tss.to_csv("results/pred_full.tss.tsv", sep="\t")
PY

```

## Tutorials and notebooks
Tutorials:
- `docs/TUTORIAL_subset.md`
- `docs/TUTORIAL_extrapolate.md`

Example notebooks (runnable end-to-end on `data/`):
- `docs/notebooks/01_subset_kmedoids_ga_selection.ipynb`
- `docs/notebooks/02_extrapolate_train_evaluate_full_fit_predict.ipynb`

## Input table formats
All inputs are TSV with:
- rows = samples
- columns = features
- first column = sample IDs (index)

Example files in `data/`:
- `feature_table_rel.tsv`, `metadata.tsv` (subset workflow)
- `X_kmers.tsv`, `X_kmers_full.tsv` (extrapolate inputs (k-mer features))
- `Y_kos.tsv` (KO profiles for paired samples)
- `picrust2_kos.tsv` (optional baseline for comparison of KO predictions)

## Outputs
All outputs go to `results/` by default.

Subset outputs (in `results/subset/`):
- `distance.tsv`, `distance_meta.json`
- `k_diagnostics.tsv`, `k_diagnostics.png`
- `kmedoids_clusters.tsv`, `kmedoids_cluster_counts.tsv`
- `ga_selected_samples.tsv`, `ga_best_scores.tsv`, `ga_fitness_array.tsv`

Extrapolate outputs:
- `oof_clr.tsv`, `oof_tss.tsv` (OOF predictions)
- `folds.tsv` (per-fold best params + metrics)
- `summary.json`, `summary.tsv` (overall metrics)
- `pred*.clr.tsv`, `pred*.tss.tsv`, `pred*.diag.tsv` (full_predict outputs)
- `pred*.metrics.tsv` (metrics when truth is provided)

## Parameter guide
### subset (intelligrate.subset)
These are the main functions used in the subset notebook. All parameters are editable in Python.

**compute_distance_matrix(feature_table, metric, assume_relative, pseudocount)**
- `feature_table`: samples x features table
- `metric`: how to compute distance (`bray`, `jaccard`, `aitchison`)
- `assume_relative`: True if values already sum to 1 per sample
- `pseudocount`: small value added before CLR (only for Aitchison)

**suggest_k(distance_df, feature_table, k_range, gap_B, random_state, return_fig)**
- `distance_df`: sample-sample distances (square matrix)
- `feature_table`: same samples, raw features
- `k_range`: which k values to test
- `gap_B`: how many random reference datasets for the gap statistic
- `random_state`: fixed seed for reproducibility
- `return_fig`: return a diagnostic plot

**fit_kmedoids(distance_df, k, random_state)**
- `distance_df`: sample-sample distances
- `k`: number of clusters
- `random_state`: fixed seed for reproducibility

**ga_subset(cluster_df, metadata_df, total_samples, balance_vars, coord_vars, ...)**
- `cluster_df`: k-medoids clusters (with `Cluster` column)
- `metadata_df`: sample metadata table
- `total_samples`: how many samples to select
- `balance_vars`: categorical fields to balance in the subset
- `coord_vars`: latitude/longitude columns for spatial diversity
- `min_category_n`: ignore categories with fewer than this many samples
- `min_per_category`: minimum per category in the selected subset
- `grid_size`: size of lat/lon grid cells for spatial diversity
- `population_size`: GA population size (bigger = slower)
- `generations`: number of GA generations (bigger = slower)
- `random_state`: fixed seed
- `fixed_include`: list of sample IDs to force include
- `fixed_exclude`: list of sample IDs to force exclude
- `metadata_weights`: relative weights per balance variable
- `grid_weight`: how much geographic coverage matters
- `distance_weight`: how much pairwise distance matters
- `balance_weight`: how much metadata balance matters
- `balance_scale`: scaling for balance term (keeps it comparable)
- `hard_penalty_weight`: penalty for violating minimum category counts

### extrapolate (intelligrate.extrapolate)
Key steps are: embed X, train with nested CV, fit a final model, predict for all samples.

**fit_x_embedding_svd_clr(X_full, min_prev_x_abs, pseudocount_x, n_components, seed)**
- `min_prev_x_abs`: drop rare k-mers (less than this many samples)
- `pseudocount_x`: added before CLR
- `n_components`: SVD embedding size
- `seed`: random seed

**nested CV training (via train._run_once or train.main)**
Training uses these parameter groups:
- **CV**: `outer_splits`, `inner_splits`, `seed`, `informed_splits`
- **Model**:
  - `min_prev_y_abs`: drop rare KOs
  - `y_detect_threshold`: detection threshold in TSS space
  - `pseudocount_y`: added before CLR
  - `neigh_k_grid`, `tau_mult_grid`, `lam_grid`, `y_latent_k_grid`: hyperparameter grids
  - `use_metric_learning`, `metric_max_pairs`, `metric_ridge_grid`
  - `ood_shrink`, `ood_shrink_inner`, `ood_lam_base`, `ood_lam_cap`, `ood_tau_inflate`
- **Objective weights**: `w_dm`, `w_wclr`, `w_pw_rmse`, `w_softf1`, `w_jsd`
- **Precision/Recall**: `prf_thresh`, `prf_weight`
- **Optional metrics**: `compute_wclr`, `compute_jsd`, `compute_pathway_rmse`,
  `pathway_rmse_per_group`, `pathway_rmse_log1p`
- **Score**: `min_prev_y_abs`, `y_detect_threshold`, `pseudocount_y`

**fit_final_model(...)**
- `X_train`, `Y_train_tpm`, `embed`: training data + embedding
- `min_prev_y_abs`, `y_detect_threshold`, `pseudocount_y`
- `neigh_k`, `tau_mult`, `lam`, `y_latent_k`
- `use_metric_learning`, `metric_ridge`, `metric_max_pairs`
- `tau_scale_k_nn`, `ood_shrink`, `ood_lam_base`, `ood_lam_cap`, `seed`

**predict_final_model(X_new, model)**
- `X_new`: new k-mer table
- `model`: model dict produced by `fit_final_model`

**fixed_param_oof_knn_on_embedding(...)**
- Runs a single CV pass with **fixed hyperparameters** (no inner CV).
- Use for leakage‑free predictions on paired samples.
- Set `outer_splits = len(X)` to approximate leave‑one‑out (slow).

**evaluate_paired_subset(truth_tpm, pred_tss, pseudocount, detect_threshold, prf_thresh, prf_weight, ...)**
- `truth_tpm`: true KO table
- `pred_tss`: predicted KO table (TSS)
- `pseudocount`, `detect_threshold`: same logic as training
- `prf_thresh`, `prf_weight`: precision/recall configuration
- Optional: `compute_wclr`, `compute_jsd`, `compute_pathway`, `compute_per_pathway`

For full explanations and runnable examples, see the [Tutorials and notebooks](#tutorials-and-notebooks) above.
