from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.extrapolate.embedding import fit_x_embedding_svd_clr
from intelligrate.extrapolate.full_fit import fit_final_model, load_model, save_model
from intelligrate.extrapolate.full_predict import predict_final_model
from intelligrate.extrapolate.metrics import dm_spearman_union, evaluate_union_metrics


def _small_x() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [8, 1, 0, 0],
            [7, 2, 0, 1],
            [0, 8, 2, 0],
            [1, 7, 1, 0],
            [0, 1, 8, 1],
            [1, 0, 7, 2],
        ],
        index=[f"s{i}" for i in range(6)],
        columns=["x1", "x2", "x3", "x4"],
        dtype=float,
    )


def _small_y() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [90, 10, 1, 0],
            [85, 14, 1, 0],
            [10, 80, 8, 2],
            [12, 78, 8, 2],
            [2, 8, 85, 5],
            [1, 7, 86, 6],
        ],
        index=[f"s{i}" for i in range(6)],
        columns=["K001", "K002", "K003", "K004"],
        dtype=float,
    )


def test_full_fit_predict_and_joblib_roundtrip(tmp_path):
    x = _small_x()
    y = _small_y()
    embed = fit_x_embedding_svd_clr(x, min_prev_x_abs=1, pseudocount_x=0.5, n_components=2, seed=0)

    model = fit_final_model(
        x,
        y,
        embed,
        min_prev_y_abs=1,
        y_detect_threshold=0.0,
        neigh_k=2,
        tau_mult=1.0,
        y_latent_k=0,
        use_metric_learning=False,
        ood_shrink=False,
        seed=0,
    )

    model_path = save_model(model, tmp_path / "model.joblib")
    loaded = load_model(model_path)
    pred_clr, pred_tss, diag = predict_final_model(x, loaded)

    assert pred_clr.shape == y.shape
    assert pred_tss.shape == y.shape
    assert np.allclose(pred_tss.sum(axis=1), 1.0)
    assert list(diag.columns) == ["ood_nn_min", "ood_median_nn", "ood_max_nn"]
    assert np.isfinite(diag.to_numpy()).all()


def test_union_metrics_handle_nonidentical_ko_sets():
    truth = _small_y()
    pred = truth.drop(columns=["K004"]).copy()
    pred["K999"] = 1.0

    score = dm_spearman_union(
        truth,
        pred,
        pseudocount=0.5 / 1e6,
        detect_threshold=0.0,
        fillna_zero=True,
    )
    metrics = evaluate_union_metrics(
        truth,
        pred,
        pseudocount=0.5 / 1e6,
        detect_threshold=0.0,
        prf_thresh=1e-6,
        prf_weight="binary",
        fillna_zero=True,
    )

    assert np.isfinite(score)
    assert {"dm_spearman", "bray_spearman", "soft_precision", "soft_recall", "soft_f1"} <= set(metrics)
    assert 0.0 <= metrics["soft_f1"] <= 1.0
