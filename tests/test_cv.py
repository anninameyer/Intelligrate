from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.extrapolate.cv_knn import fixed_param_oof_knn_on_embedding
from intelligrate.extrapolate.embedding import fit_x_embedding_svd_clr
from intelligrate.extrapolate.train import _iter_grid_configs


def test_fixed_param_oof_smoke_test_outputs_both_spaces():
    n = 12
    x = pd.DataFrame(
        np.arange(n * 8, dtype=float).reshape(n, 8) + 1.0,
        index=[f"s{i}" for i in range(n)],
        columns=[f"x{i}" for i in range(8)],
    )
    y = pd.DataFrame(
        np.arange(n * 6, dtype=float).reshape(n, 6) + 1.0,
        index=x.index,
        columns=[f"K{i}" for i in range(6)],
    )
    embed = fit_x_embedding_svd_clr(x, min_prev_x_abs=1, pseudocount_x=0.5, n_components=3, seed=0)

    oof_clr, oof_tss, folds = fixed_param_oof_knn_on_embedding(
        X=x,
        Y_tpm=y,
        embed=embed,
        ko_to_superclass=None,
        outer_splits=3,
        seed=0,
        min_prev_y_abs=1,
        y_detect_threshold=0.0,
        neigh_k=2,
        tau_mult=1.0,
        lam=0.0,
        y_latent_k=0,
        use_metric_learning=False,
        ood_shrink=False,
    )

    assert oof_clr.shape == y.shape
    assert oof_tss.shape == y.shape
    assert not folds.empty
    assert np.allclose(oof_tss.dropna().sum(axis=1), 1.0)


def test_iter_grid_configs_expands_cartesian_product_without_mutating_input():
    cfg = {
        "model": {"neigh_k": 2, "lam": 0.0},
        "embed": {"n_components": 2},
        "grid": {
            "model": {"neigh_k": [2, 3]},
            "embed": {"n_components": [2, 4]},
        },
    }

    configs = _iter_grid_configs(cfg)

    assert len(configs) == 4
    assert cfg["model"]["neigh_k"] == 2
    assert {tuple(sorted(overrides.items())) for _, overrides in configs} == {
        (("embed.n_components", 2), ("model.neigh_k", 2)),
        (("embed.n_components", 4), ("model.neigh_k", 2)),
        (("embed.n_components", 2), ("model.neigh_k", 3)),
        (("embed.n_components", 4), ("model.neigh_k", 3)),
    }
