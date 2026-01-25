# Project goal
Build a reusable Python package **intelligrate** with two main functionalities:
1) `intelligrate.extrapolate`: learn a model to predict KO profiles from ASV_kmers using paired amplicon–shotgun samples, then extrapolate KOs to additional samples with only ASV_kmers.
2) `intelligrate.subset`: (later) genetic-algorithm-based subsetting of samples for shotgun sequencing to optimize diversity (ASV_kmers) and metadata representation.

This repository currently focuses on (1) `intelligrate.extrapolate`. The subsetting module will be packaged later once this one is stable.

# Important rules
- Do not run destructive commands (rm, mv on data, sudo, chmod -R, etc.).
- Do not access files outside this repository (Intelligrate) and do not write anything outside this repository.
- Do not install system packages; only use the current venv.
- Do not modify files in data/.

# Package naming
- The published package name must be `intelligrate`.
- Refactor/rename existing `ko_from_amplicon` modules into `intelligrate` if needed.

# Primary objective (must maximize)
Maximize OBJECTIVE_DM_SPEARMAN_MEAN printed by `make score`:
- Spearman correlation between upper triangles of sample–sample Aitchison distance matrices in CLR space.
- Must remain abundance-aware (not presence/absence only).
- Evaluation must NOT intersect KO sets. Use KO-union evaluation:
  U = KOs_truth ∪ KOs_pred (or ∪ KOs_picrust2 where relevant).
  Reindex to U (fill missing with 0 in TSS), then compute CLR (with pseudocount), then Aitchison DM.

# Non-negotiables (scientific validity)
- Always use nested CV (outer + inner) for any model or any hyperparameter search.
- No leakage:
  - Any data-dependent preprocessing, filtering, scaling, embeddings, or feature selection must be fit on training data only within each CV fold and then applied to validation/test.
  - Exception allowed: an unsupervised embedding fit on full X (no Y) is permitted if documented and does not use targets.
  - The outer test fold must never influence model selection or transformations used in training.
- Always output OOF predictions in BOTH spaces:
  - CLR predictions (oof_clr_*.tsv)
  - TSS predictions (oof_tss_*.tsv)

# Metrics (must be available in BOTH train and full_predict evaluation)
Compute and report the same metrics for:
(A) OOF predictions from nested CV training, and
(B) full-model predictions evaluated on the paired subset (intersection of predicted samples and truth samples).
Metrics:
- Primary: CLR Aitchison DM Spearman (KO-union)
- Secondary (reporting):
  - Bray–Curtis DM Spearman (computed on TSS)
  - Procrustes correlation (ordination of true vs predicted; deterministic)
  - Thresholded Precision/Recall/F1 in TSS:
    - threshold must be configurable (treat as a hyperparameter if desired)
    - weight scheme must be configurable (binary/truth_abundance/pred_abundance)
  - Weighted CLR-MSE, JSD, pathway RMSE (optional; compute only if enabled). Pathway RMSE is biologically interesting, so we should have a toggle to compute it (overall, and specifically per pathways so users can review which pathways are predicted well or poorly).

# Benchmarking against PICRUSt2 (optional but important)
- If a PICRUSt2 output file is provided, compute the same metrics for PICRUSt2.
- Log in summary JSON:
  - picrust2_dm_union
  - model_dm_union
  - delta_union = model_dm_union - picrust2_dm_union

# Required functionality in intelligrate.extrapolate
Implement these modules and keep them stable:
1) `train`:
   - nested CV with hyperparameter grids
   - outputs: oof_clr, oof_tss, folds.csv (per-fold best params + metrics), summary.json
2) `full_fit`:
   - fit final model on all paired data using chosen hyperparameters
   - save model artifact (joblib)
3) `full_predict`:
   - predict for new X (kmer-only)
   - “always output full predicted tables (CLR + TSS) and a distance-to-training diagnostic (e.g., min NN distance in embedding space to the paired training set) to flag extrapolations far from the training manifold
   - if truth Y exists for a subset, evaluate with the same metrics as in the 'train' mode and expose these results

# Efficiency requirement
Improve runtime of `make score` substantially without violating nested CV or changing the primary objective.
Allowed techniques:
- skip computing metrics whose weights are 0 during inner CV
- cache computations within folds when scientifically valid
- avoid O(n^2) pair enumeration in metric learning: sample pairs directly
- vectorize hot loops; reduce Python-level loops
- `score_fast` config can be used for experimentation and final scoring.

# Workflow rules (strict)
- Use `make score_fast` for iteration.
- After each experimental change:
  1) Run `make score_fast`
  2) Append result to results/experiments_log.md (score + runtime + what changed)
  3) If worse: revert code changes and do NOT commit/push
  4) If better (higher model_dm_union; or same score with clearly lower runtime):
     - commit message must include model_dm_union, picrust2_dm_union, delta_union, runtime
     - push to origin
- Keep solution simple and packageable.
- Document methodological changes in README.md.

# Stop conditions
Stop and write a report (results/REPORT.md) when:
- model_dm_union >= 0.65 on `make score` (not only fast), OR
- no improvement > 0.01 after 8 meaningful attempts, OR
- runtime is reduced by >50% with no loss in score.
Report must include: best config, runtime, what changed, and whether further optimization seems promising.

# Tutorial requirement
After packaging is stable, add a tutorial (notebook or markdown) that demonstrates:
- installation of intelligrate
- what input data format is expected
- train (nested CV) on example data in data/
- full_fit on paired data
- full_predict on full X dataset
- optional: PICRUSt2 comparison
- explain for beginners which files are generated and what they mean, and how to tune hyperparameters and what are important considerations.

