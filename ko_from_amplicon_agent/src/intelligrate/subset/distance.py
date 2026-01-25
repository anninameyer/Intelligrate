from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .utils import clr_transform, ensure_relative_abundance


def compute_distance_matrix(
    feature_table: pd.DataFrame,
    *,
    metric: str = "bray",
    assume_relative: bool = False,
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute a sample-sample distance matrix.

    Supported metrics:
    - "bray": Bray-Curtis on relative abundance
    - "jaccard": Jaccard on presence/absence (binary)
    - "aitchison": Euclidean distance in CLR space
    """
    metric = metric.lower()
    X = feature_table.copy()

    if metric in {"bray", "aitchison"}:
        X = ensure_relative_abundance(X, assume_relative=assume_relative)

    if metric == "jaccard":
        X = (X > 0).astype(int)
        D = squareform(pdist(X.to_numpy(float), metric="jaccard"))
    elif metric == "bray":
        D = squareform(pdist(X.to_numpy(float), metric="braycurtis"))
    elif metric == "aitchison":
        X_clr = clr_transform(X, pseudocount=float(pseudocount))
        D = squareform(pdist(X_clr.to_numpy(float), metric="euclidean"))
    else:
        raise ValueError("metric must be one of: bray, jaccard, aitchison")

    return pd.DataFrame(D, index=X.index, columns=X.index)
