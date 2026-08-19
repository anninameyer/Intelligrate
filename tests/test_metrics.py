from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.extrapolate.metrics import (
    bray_curtis_dm,
    collapse_by_group,
    corr_upper_triangle,
    feature_weights_from_variance,
    jsd_rows,
    pathway_rmse_tss,
    pathway_rmse_tss_per_group,
    samplewise_spearman,
    weighted_clr_mse,
)
from intelligrate.extrapolate.transforms import clr_rows, tss_rows


def test_distance_correlation_and_samplewise_spearman_are_finite():
    table = pd.DataFrame(
        np.arange(60, dtype=float).reshape(5, 12) + 1.0,
        index=[f"s{i}" for i in range(5)],
    )
    distance = bray_curtis_dm(tss_rows(table))

    assert distance.shape == (5, 5)
    assert np.isfinite(corr_upper_triangle(distance, distance, method="spearman"))
    assert samplewise_spearman(table, table).eq(1.0).all()


def test_optional_metric_helpers_return_expected_shapes():
    truth = pd.DataFrame(
        [[10.0, 0.0, 5.0], [2.0, 8.0, 0.0], [1.0, 1.0, 9.0]],
        index=["s1", "s2", "s3"],
        columns=["K1", "K2", "K3"],
    )
    pred = truth.copy()
    groups = {"K1": "A", "K2": "A", "K3": "B"}
    truth_tss = tss_rows(truth)
    pred_tss = tss_rows(pred)
    truth_clr = clr_rows(truth_tss, pseudocount=1e-6)
    pred_clr = clr_rows(pred_tss, pseudocount=1e-6)

    collapsed = collapse_by_group(truth_tss, groups)
    weights = feature_weights_from_variance(truth_clr)

    assert list(collapsed.columns) == ["A", "B"]
    assert weights.shape == (3,)
    assert weighted_clr_mse(truth_clr, pred_clr, weights) == 0.0
    assert jsd_rows(truth_tss.to_numpy(), pred_tss.to_numpy()) == 0.0
    assert pathway_rmse_tss(truth_tss, pred_tss, groups) == 0.0
    assert pathway_rmse_tss_per_group(truth_tss, pred_tss, groups).eq(0.0).all()
