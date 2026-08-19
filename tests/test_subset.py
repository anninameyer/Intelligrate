from __future__ import annotations

import numpy as np
import pandas as pd

from intelligrate.subset import compute_distance_matrix, fit_kmedoids, ga_subset


def test_compute_distance_matrix_is_square_symmetric_with_zero_diagonal():
    feature_table = pd.DataFrame(
        [[0.5, 0.5, 0.0], [0.1, 0.8, 0.1], [0.0, 0.3, 0.7]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2", "f3"],
    )

    distance = compute_distance_matrix(feature_table, metric="bray", assume_relative=True)

    assert distance.shape == (3, 3)
    assert distance.index.equals(feature_table.index)
    assert distance.columns.equals(feature_table.index)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)


def test_kmedoids_and_ga_subset_return_requested_samples():
    samples = [f"s{i}" for i in range(6)]
    distance = pd.DataFrame(
        [
            [0.0, 0.1, 0.9, 0.8, 0.7, 0.6],
            [0.1, 0.0, 0.8, 0.7, 0.6, 0.5],
            [0.9, 0.8, 0.0, 0.2, 0.7, 0.8],
            [0.8, 0.7, 0.2, 0.0, 0.6, 0.7],
            [0.7, 0.6, 0.7, 0.6, 0.0, 0.1],
            [0.6, 0.5, 0.8, 0.7, 0.1, 0.0],
        ],
        index=samples,
        columns=samples,
    )
    metadata = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "latitude": [47.0, 47.1, 46.0, 46.1, 45.0, 45.1],
            "longitude": [8.0, 8.1, 7.0, 7.1, 6.0, 6.1],
        },
        index=samples,
    )

    kmedoids = fit_kmedoids(distance, k=3, random_state=0)
    selected, best_scores, fitness = ga_subset(
        kmedoids["cluster_df"],
        metadata,
        total_samples=3,
        balance_vars=["group"],
        coord_vars=("latitude", "longitude"),
        min_category_n=1,
        min_per_category=0,
        population_size=6,
        generations=2,
        random_state=0,
    )

    assert selected.shape[0] == 3
    assert "Cluster" in selected.columns
    assert len(best_scores) == 2
    assert len(fitness) == 2
