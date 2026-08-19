from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.extrapolate.knn_core import (
    apply_ood_shrinkage,
    decode_y_latent,
    encode_y_latent,
    fit_supervised_diag_metric,
    fit_y_latent_svd,
    knn_kernel_predict_tau_abs,
    median_nn_distance,
    prf_thresholded,
)


def test_knn_kernel_predict_returns_weighted_neighbor_average():
    z_train = np.array([[0.0], [1.0], [10.0]])
    z_test = np.array([[0.0]])
    t_train = np.array([[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])

    pred = knn_kernel_predict_tau_abs(z_train, z_test, t_train, k=2, tau_abs=0.5, lam=0.0)

    assert pred.shape == (1, 2)
    assert pred[0, 0] > pred[0, 1]
    assert pred[0, 0] < 1.0


def test_median_nn_distance_is_positive_for_distinct_points():
    z = np.array([[0.0], [1.0], [3.0]])

    assert median_nn_distance(z, k=1) > 0.0


def test_latent_svd_encode_decode_preserves_shape_and_centering():
    y_clr = pd.DataFrame(
        [
            [-1.0, 0.0, 1.0],
            [-0.5, -0.5, 1.0],
            [1.0, -0.5, -0.5],
            [0.0, 1.0, -1.0],
        ],
        columns=["K1", "K2", "K3"],
    )

    svd, mean = fit_y_latent_svd(y_clr, k=2, random_state=0)
    latent = encode_y_latent(y_clr, svd, mean)
    decoded = decode_y_latent(latent, svd, mean)

    assert latent.shape[0] == y_clr.shape[0]
    assert decoded.shape == y_clr.shape
    assert np.allclose(decoded.mean(axis=1), 0.0)


def test_supervised_diag_metric_is_finite_nonnegative():
    x = pd.DataFrame([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    y = pd.DataFrame([[0.0, 0.0], [1.0, 0.0], [0.0, 4.0], [1.0, 4.0]])

    weights = fit_supervised_diag_metric(x, y, max_pairs=6, random_state=0, ridge=1.0)

    assert weights.shape == (2,)
    assert np.isfinite(weights).all()
    assert (weights >= 0.0).all()


def test_ood_shrinkage_moves_predictions_toward_training_mean():
    y_train = pd.DataFrame([[0.0, 0.0], [2.0, 2.0]], columns=["K1", "K2"])
    pred = np.array([[10.0, -10.0]])

    shrunk, lam = apply_ood_shrinkage(
        pred,
        y_train,
        nn_min=np.array([10.0]),
        lam_base=1.0,
        lam_cap=0.5,
        return_lam=True,
    )

    assert np.allclose(lam, [0.5])
    assert np.linalg.norm(shrunk - y_train.mean().to_numpy()) < np.linalg.norm(pred - y_train.mean().to_numpy())


def test_prf_thresholded_supports_weight_modes():
    truth = np.array([[1.0, 0.0, 1.0]])
    pred = np.array([[1.0, 1.0, 0.0]])

    for weight in ["binary", "truth_abundance", "pred_abundance"]:
        precision, recall, f1 = prf_thresholded(truth, pred, thresh=0.5, weight=weight)
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0
