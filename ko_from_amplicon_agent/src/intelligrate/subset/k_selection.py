from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from .kmedoids import _kmedoids_precomputed


def _compute_gap_statistic(feature_data: np.ndarray, k_range, B: int, random_state: int):
    rng = np.random.default_rng(int(random_state))
    gaps = []
    s_k = []
    n_samples, n_features = feature_data.shape

    fmin = feature_data.min(axis=0)
    fmax = feature_data.max(axis=0)

    for k in k_range:
        ref_disps = np.zeros(B)
        for _b in range(B):
            reference = rng.uniform(fmin, fmax, size=(n_samples, n_features))
            km = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state))
            km.fit(reference)
            ref_disps[_b] = np.mean(np.min(cdist(reference, km.cluster_centers_), axis=1))

        km = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state))
        km.fit(feature_data)
        orig_disp = np.mean(np.min(cdist(feature_data, km.cluster_centers_), axis=1))

        gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)
        std_dev = np.std(np.log(ref_disps)) * np.sqrt(1 + 1 / B)

        gaps.append(gap)
        s_k.append(std_dev)

    return gaps, s_k


def suggest_k(
    distance_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    *,
    k_range=range(2, 31),
    gap_B: int = 5,
    random_state: int = 42,
    return_fig: bool = False,
):
    """
    Compute k diagnostics (silhouette, Davies-Bouldin, gap statistic).
    Returns a dict with metrics and top-k rankings.
    """
    D = distance_df.to_numpy(float)
    X = feature_table.to_numpy(float)

    silhouette_scores = []
    db_scores = []

    for k in k_range:
        try:
            labels, _medoids = _kmedoids_precomputed(D, int(k), random_state=int(random_state))
            sil_score = silhouette_score(D, labels, metric="precomputed")
            db_score = davies_bouldin_score(X, labels)
            silhouette_scores.append(sil_score)
            db_scores.append(db_score)
        except Exception:
            silhouette_scores.append(np.nan)
            db_scores.append(np.nan)

    gap_scores, gap_std = _compute_gap_statistic(X, k_range, B=int(gap_B), random_state=int(random_state))

    def _top_k(scores, reverse=False, top_n=5):
        scores = np.array(scores)
        valid = ~np.isnan(scores)
        idx = np.argsort(scores[valid])
        if not reverse:
            idx = idx[::-1]
        ks = np.array(list(k_range))[valid][idx]
        vals = scores[valid][idx]
        return list(zip(ks[:top_n], vals[:top_n]))

    result = {
        "k_values": list(k_range),
        "silhouette": silhouette_scores,
        "davies_bouldin": db_scores,
        "gap": gap_scores,
        "gap_std": gap_std,
        "top_silhouette": _top_k(silhouette_scores, reverse=False),
        "top_davies_bouldin": _top_k(db_scores, reverse=True),
        "top_gap": _top_k(gap_scores, reverse=False),
    }

    if return_fig:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        axs[0].plot(result["k_values"], result["silhouette"], marker="o", color="#345084FF")
        axs[0].set_title("Silhouette Score")
        axs[0].set_xlabel("k")
        axs[0].set_ylabel("Silhouette Score")

        axs[1].plot(result["k_values"], result["davies_bouldin"], marker="o", color="#345084FF")
        axs[1].set_title("Davies-Bouldin Index")
        axs[1].set_xlabel("k")
        axs[1].set_ylabel("DB Index")

        axs[2].errorbar(result["k_values"], result["gap"], yerr=result["gap_std"], marker="o", color="#345084FF")
        axs[2].set_title("Gap Statistic")
        axs[2].set_xlabel("k")
        axs[2].set_ylabel("Gap Value")
        plt.tight_layout()
        result["figure"] = fig

    return result
