# Report (2026-01-24)

## Best config
- model_dm_union: 0.600142
- picrust2_dm_union: 0.430436
- delta_union: 0.169706
- runtime_s: 55.6
- results_prefix: 20260124_205222

Key config changes:
- ood_lam_base: 0.7
- y_detect_threshold: 2000.0
- neigh_k_grid: [16, 18, 20, 24]
- Legacy union metric used for model_dm_union

## What changed
- Increased ood_lam_base from 0.5 to 0.7
- Narrowed neigh_k_grid to higher k values
- Switched model_dm_union to legacy KO-union metric (no NaN fill) while keeping raw/strict metrics for reference

## Runtime
- score_fast runtime ~55.6s

## Further optimization
- Potential small gains from exploring ood_lam_base around 0.6–0.8 with finer steps
- Consider tightening tau_mult_grid around 0.5 if stability improves
- Evaluate alternative, simple models (ridge/PLS) only if KNN plateaus
