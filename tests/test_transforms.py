from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.extrapolate.transforms import clr_rows, clr_to_comp, keep_by_prevalence, tss_rows


def test_tss_rows_normalizes_nonzero_rows_and_preserves_zero_rows_as_nan():
    table = pd.DataFrame(
        [[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
        index=["s1", "s2"],
        columns=["a", "b", "c"],
    )

    normalized = tss_rows(table)

    assert np.isclose(normalized.loc["s1"].sum(), 1.0)
    assert normalized.loc["s2"].isna().all()


def test_clr_rows_are_centered_per_sample():
    table = pd.DataFrame(
        [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]],
        index=["s1", "s2"],
        columns=["a", "b", "c"],
    )

    clr = clr_rows(table, pseudocount=1e-6)

    assert np.allclose(clr.mean(axis=1), 0.0)


def test_clr_to_comp_returns_closed_compositions():
    clr = pd.DataFrame(
        [[-1.0, 0.0, 1.0], [2.0, -1.0, -1.0]],
        index=["s1", "s2"],
        columns=["a", "b", "c"],
    )

    comp = clr_to_comp(clr)

    assert np.all(comp.to_numpy() >= 0.0)
    assert np.allclose(comp.sum(axis=1), 1.0)


def test_keep_by_prevalence_uses_detection_threshold():
    table = pd.DataFrame(
        [[0.0, 2.0, 5.0], [1.0, 0.0, 6.0], [0.0, 3.0, 0.0]],
        columns=["rare", "medium", "common"],
    )

    kept = keep_by_prevalence(table, min_prev_abs=2, detect_threshold=1.0)

    assert list(kept) == ["medium", "common"]
