from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .transforms import keep_by_prevalence, tss_rows, clr_rows


def fit_x_embedding_svd_clr(
    X_full: pd.DataFrame,
    *,
    min_prev_x_abs: int = 2,
    pseudocount_x: float = 0.5,
    n_components: int = 128,
    seed: int = 0,
) -> dict:
    """
    Fit an unsupervised CLR->SVD embedding on X_full.
    Returns a dict to transform any subset with the same columns.
    """
    x_keep = keep_by_prevalence(X_full, min_prev_abs=int(min_prev_x_abs), detect_threshold=0.0)
    X0 = X_full.loc[:, x_keep]

    pc_x = float(pseudocount_x) / max(1.0, float(np.nanmax(X0.sum(axis=1))))
    X_clr = clr_rows(tss_rows(X0), pseudocount=pc_x)

    Xmat = X_clr.to_numpy(dtype=float)
    col_mean = np.nanmean(Xmat, axis=0, keepdims=True)
    Xc = Xmat - col_mean

    k_eff = min(int(n_components), min(Xc.shape) - 1)
    k_eff = max(k_eff, 2)

    svd = TruncatedSVD(n_components=k_eff, random_state=int(seed))
    svd.fit(Xc)

    return {"x_keep": x_keep, "pc_x": pc_x, "col_mean": col_mean, "svd": svd}


def transform_x_embedding_svd_clr(X: pd.DataFrame, embed: dict) -> np.ndarray:
    X0 = X.loc[:, embed["x_keep"]]
    X_clr = clr_rows(tss_rows(X0), pseudocount=float(embed["pc_x"]))
    Xmat = X_clr.to_numpy(dtype=float)
    Xc = Xmat - embed["col_mean"]
    return embed["svd"].transform(Xc)
