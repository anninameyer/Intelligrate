from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.subset.utils import check_alignment, clr_transform, ensure_relative_abundance


def test_check_alignment_counts_overlaps_and_missing_ids():
    feature_table = pd.DataFrame(index=["s1", "s2", "s3"])
    metadata = pd.DataFrame(index=["s2", "s3", "s4"])

    alignment = check_alignment(feature_table, metadata)

    assert alignment == {
        "feature_samples": 3,
        "metadata_samples": 3,
        "common_samples": 2,
        "missing_in_metadata": 1,
        "missing_in_feature_table": 1,
    }


def test_ensure_relative_abundance_normalizes_counts():
    counts = pd.DataFrame([[1.0, 3.0], [2.0, 2.0]], index=["s1", "s2"])

    relative = ensure_relative_abundance(counts)

    assert np.allclose(relative.sum(axis=1), 1.0)


def test_subset_clr_transform_centers_rows():
    table = pd.DataFrame([[0.25, 0.75], [0.5, 0.5]])

    clr = clr_transform(table, pseudocount=1e-6)

    assert np.allclose(clr.mean(axis=1), 0.0)
