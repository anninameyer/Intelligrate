from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn_extra.cluster import KMedoids  # type: ignore
    _HAVE_SKLEARN_EXTRA = True
except Exception:
    KMedoids = None
    _HAVE_SKLEARN_EXTRA = False


def _kmedoids_precomputed(D: np.ndarray, k: int, *, random_state: int = 42, max_iter: int = 300):
    rng = np.random.default_rng(int(random_state))
    n = D.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and number of samples")

    medoids = rng.choice(n, size=int(k), replace=False)
    for _ in range(int(max_iter)):
        dist_to_medoids = D[:, medoids]
        labels = np.argmin(dist_to_medoids, axis=1)

        new_medoids = medoids.copy()
        for ci in range(k):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                pool = [i for i in range(n) if i not in new_medoids]
                if not pool:
                    continue
                new_medoids[ci] = rng.choice(pool, size=1)[0]
                continue
            subD = D[np.ix_(members, members)]
            costs = subD.sum(axis=1)
            new_medoids[ci] = members[int(np.argmin(costs))]

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    dist_to_medoids = D[:, medoids]
    labels = np.argmin(dist_to_medoids, axis=1)
    return labels, medoids


def fit_kmedoids(distance_df: pd.DataFrame, k: int, *, random_state: int = 42) -> dict:
    D = distance_df.to_numpy(float)
    sample_ids = distance_df.index.tolist()

    if _HAVE_SKLEARN_EXTRA:
        model = KMedoids(n_clusters=int(k), metric="precomputed", random_state=int(random_state))
        labels = model.fit_predict(D)
        medoid_indices = model.medoid_indices_
    else:
        labels, medoid_indices = _kmedoids_precomputed(D, int(k), random_state=int(random_state))
        model = None

    medoid_samples = [sample_ids[i] for i in medoid_indices]

    cluster_df = pd.DataFrame({"Sample": sample_ids, "Cluster": labels})
    cluster_df["Is_Medoid"] = cluster_df["Sample"].isin(medoid_samples)
    cluster_df = cluster_df.set_index("Sample")

    counts = cluster_df["Cluster"].value_counts().sort_index()

    return {
        "model": model,
        "cluster_df": cluster_df,
        "medoid_samples": medoid_samples,
        "cluster_counts": counts,
    }
