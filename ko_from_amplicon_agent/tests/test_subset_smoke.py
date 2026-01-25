import pandas as pd

from intelligrate.subset import compute_distance_matrix, suggest_k, fit_kmedoids, ga_subset


def test_subset_smoke(tmp_path):
    ft = pd.read_csv("data/feature_table_rel.tsv", sep="\t", index_col=0)
    md = pd.read_csv("data/metadata.tsv", sep="\t", index_col=0)

    D = compute_distance_matrix(ft, metric="bray", assume_relative=True)
    assert D.shape[0] == ft.shape[0]

    k_metrics = suggest_k(D, ft, k_range=range(2, 5), gap_B=2, random_state=0)
    assert len(k_metrics["k_values"]) == 3

    km = fit_kmedoids(D, k=3, random_state=0)
    clusters = km["cluster_df"]
    assert "Cluster" in clusters.columns

    selected, best_scores, fitness = ga_subset(
        clusters,
        md,
        total_samples=30,
        balance_vars=["r_samp_country"],
        coord_vars=("latitude", "longitude"),
        min_category_n=2,
        min_per_category=2,
        population_size=10,
        generations=3,
        random_state=0,
    )
    assert len(selected) == 30
    assert len(best_scores) == 3
    assert len(fitness) == 3
