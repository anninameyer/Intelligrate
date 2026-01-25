# KO From Amplicon

## What extrapolate does (short)
`intelligrate.extrapolate` learns a KNN-based mapping from ASV k-mer profiles to KO profiles using paired
amplicon–shotgun samples. It embeds ASV k-mers in CLR space with SVD, optionally learns a diagonal metric,
and predicts KOs in CLR (with optional latent SVD) before converting to TSS. Out-of-distribution distances
flag samples far from the training manifold.

## intelligrate package
This repository now ships a reusable Python package named `intelligrate`. The current pipeline lives under
`src/intelligrate/extrapolate/`, while the legacy `ko_from_amplicon` namespace remains as a compatibility shim.

### Quickstart
- Install (end users): `pip install intelligrate`
- Train (nested CV, OOF outputs, summary): `make score` or `python -m intelligrate.extrapolate.train --config configs/default.yaml`
- `make score_fast` is an alias of `make score` (default config is already fast)
- Full fit (produce model artifact): `python -m intelligrate.extrapolate.full_fit --x data/X_kmers.tsv --y data/Y_kos.tsv --embed-path results/embed.joblib --model-out results/model.joblib`
- Full predict (produce CLR/TSS + diagnostics): `python -m intelligrate.extrapolate.full_predict --model results/model.joblib --x data/X_kmers_full.tsv --out-prefix results/pred`

### Train outputs
`make score` and `make score_fast` write the following to `results/`:
- `oof_clr.tsv` and `oof_tss.tsv`: OOF predictions in CLR and TSS
- `folds.tsv`: per-fold best params + fold metrics
- `summary.json` and `summary.tsv`: overall metrics (including PICRUSt2 benchmark if provided)

Timestamped copies are also written for reproducibility.

For a full walkthrough (installation, data format, and workflows), see `docs/TUTORIAL.md`.

## Evaluation updates (2026-01-24)
- Added KO-union DM Spearman benchmarking for model OOF vs truth and PICRUSt2 vs truth.
- Summary JSON now includes: `model_dm_union`, `picrust2_dm_union`, `delta_union`, and `runtime_sec`.
- `make score`/`make score_fast` print union metrics when PICRUSt2 is provided.
- Added an optimistic full-fit deployment score (`full_fit_dm_union`) computed by fitting on all paired samples using mode CV params.
- Added secondary Bray–Curtis DM Spearman and Procrustes similarity (Aitchison + Bray–Curtis) for OOF, full-fit, and PICRUSt2. These now apply the same `y_detect_threshold` filtering used in training.
- Union-metric evaluation uses the legacy (no NaN fill) KO-union for the main `model_dm_union` score. Raw (unthresholded) and strict (NaN fill) union metrics are also reported alongside thresholded versions.

## Efficiency updates (2026-01-24)
- Inner CV skips secondary metrics when their weights are zero.
- Inner CV caches validation ground-truth distance matrices per split.
- Inner CV caches metric-learning transforms and latent SVDs per split.
- Supervised metric learning samples up to `max_pairs` without enumerating all pairs.
