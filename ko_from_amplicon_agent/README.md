# KO From Amplicon

## Evaluation updates (2026-01-24)
- Added KO-union DM Spearman benchmarking for model OOF vs truth and PICRUSt2 vs truth.
- Summary JSON now includes: `model_dm_union`, `picrust2_dm_union`, `delta_union`, and `runtime_sec`.
- `make score`/`make score_fast` print union metrics when PICRUSt2 is provided.
- Added an optimistic full-fit deployment score (`full_fit_dm_union`) computed by fitting on all paired samples using mode CV params.
- Added secondary Bray–Curtis DM Spearman and Procrustes similarity (Aitchison + Bray–Curtis) for OOF, full-fit, and PICRUSt2.

## Efficiency updates (2026-01-24)
- Inner CV skips secondary metrics when their weights are zero.
- Inner CV caches validation ground-truth distance matrices per split.
- Inner CV caches metric-learning transforms and latent SVDs per split.
- Supervised metric learning samples up to `max_pairs` without enumerating all pairs.
