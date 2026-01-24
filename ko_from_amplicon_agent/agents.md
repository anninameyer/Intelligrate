# Project goal
Learn a model to predict KO profiles from ASV_kmers using 95 paired amplicon–shotgun samples.
Then reuse the trained pipeline (including preprocessing + embedding + model hyperparameters) to infer KOs for additional samples where only ASV_kmers are available.

# important rules
- Do not run destructive commands (rm, mv on data, sudo, chmod -R, etc.).
- Do not access files outside this repository (Intelligrate) and do not write anything outside this repository.
- Do not install system packages; only use the current venv.

# Primary objective (must maximize)
Maximize OBJECTIVE_DM_SPEARMAN_MEAN printed by `make score`:
- Spearman correlation between upper triangles of sample–sample Aitchison distance matrices in CLR space.
- This objective must remain abundance-aware (not presence/absence only).

# Non-negotiables (scientific validity)
- Always use nested CV (outer + inner) for any model or any hyperparameter search.
- No leakage:
  - Any data-dependent preprocessing, filtering, scaling, embeddings, or feature selection must be fit on training data only within each CV fold and then applied to validation/test.
  - The outer test fold must never influence model selection or transformations used in training.
- Always output OOF predictions in BOTH spaces:
  - CLR predictions (oof_clr_*.tsv)
  - TSS predictions (oof_tss_*.tsv)

# Secondary metrics (reporting + optional tuning only)
These may be computed and logged, but must not replace the primary objective unless explicitly justified and documented in README.md:
- pathway/superclass representation metrics (e.g., pathway RMSE in TSS)
- weighted CLR-MSE, JSD in TSS
- thresholded PR/F1 in TSS with tuned thresholds (threshold may be treated as a hyperparameter)

# Benchmarking against PICRUSt2 (important)
- If a PICRUSt2 output file is provided, compute the same primary objective on it (DM Spearman in CLR space) for the same samples and comparable feature set.
- The aim is to outperform PICRUSt2 on the primary objective and to have plausible pathway-level representation.
- PICRUSt2 comparison must NOT intersect KO sets.
  Evaluate using KO-union: U = KOs_truth ∪ KOs_pred.
  Reindex both truth and prediction to U (fill missing with 0), then TSS→CLR, then compute Aitchison DM and upper-triangle Spearman.
  This penalizes spurious KOs and missing KOs.
- Log in summary JSON:
  - picrust2_dm_union
  - model_dm_union
  - delta_union = model_dm_union - picrust2_dm_union


# Efficiency requirement (critical for agentic iteration)
Current end-to-end runtime is ~770s per `make score`. Improve runtime substantially without violating nested CV or changing the primary objective.
Allowed efficiency techniques:
- remove unnecessary computations from inner CV when their weight is 0 (e.g., do not compute pathway RMSE / JSD / PRF during inner selection unless explicitly enabled)
- cache/reuse repeated computations within folds where scientifically valid (e.g., ground-truth distance matrices for a given validation split)
- avoid O(n^2) pair construction in metric learning (sample pairs directly rather than enumerating all pairs)
- vectorize hot loops, reduce Python-level loops, and use efficient distance computations
- add a "fast mode" config for quick iteration (smaller folds/grids) BUT final results must be confirmed with the full default config
- parallelization is allowed if deterministic (fixed seeds), and results must remain reproducible

# Workflow rules (strict)
- Use `make score_fast` for iteration when available; only validate candidates with `make score`.
- After each experimental change:
  1) Run `make score_fast` (or `make score` if fast mode not available).
  2) Log the result locally (append to results/experiments_log.md).
  3) If the result is worse, revert code changes (git restore) and do NOT commit/push.
  4) If the result is better (higher model_dm_union; or same score with clearly lower runtime), then:
     - run `make score` to confirm on the full config,
     - commit with a message including model_dm_union, picrust2_dm_union, delta_union, runtime,
     - push to origin.
- Keep a short changelog in results/ (or use git commit messages) indicating score and runtime.
- Keep the solution simple and packageable. Prefer simpler models over complex ones if performance is comparable.
- Do not modify data files in data/.
- Do not use external training data.
- Document all methodological changes and any new objectives/metrics in README.md.

# Stop conditions
Stop and write a report (in a new markdown file under results/ or REPORT.md) when:
- model_dm_union >= 0.60 on `make score` (full config), AND
- model_dm_union > picrust2_dm_union (i.e., delta_union > 0).
The report must include: best config, runtime, what changed, and whether further optimization seems promising.


# Model exploration guidance
Start by improving and de-leaking the current KNN+embedding baseline.
If it reaches a ceiling, try other models that remain simple and packageable:
- multi-task elastic net / ridge regression
- reduced-rank regression / PLS
- random forest / gradient boosting (only if justified; beware overfitting with n=95)
- NMF-based mappings (if carefully validated)
Neural nets are allowed only if they remain small, reproducible, and demonstrably better.
