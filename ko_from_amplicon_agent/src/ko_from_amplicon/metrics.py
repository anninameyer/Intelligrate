from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import spearmanr


def aitchison_dm(Y_clr: pd.DataFrame) -> np.ndarray:
    # Euclidean in CLR space equals Aitchison distance
    return squareform(pdist(Y_clr.to_numpy(dtype=float), metric="euclidean"))


def bray_curtis_dm(Y_tss: pd.DataFrame) -> np.ndarray:
    return squareform(pdist(Y_tss.to_numpy(dtype=float), metric="braycurtis"))


def corr_upper_triangle(A: np.ndarray, B: np.ndarray, method: str = "spearman") -> float:
    iu = np.triu_indices_from(A, k=1)
    a = A[iu]
    b = B[iu]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    if method == "spearman":
        return float(spearmanr(a[m], b[m]).correlation)
    if method == "pearson":
        return float(np.corrcoef(a[m], b[m])[0, 1])
    raise ValueError(method)


def _cmdscale(D: np.ndarray, n_components: int) -> np.ndarray:
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D**2) @ H
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    w = np.clip(w[:n_components], 0, None)
    return v[:, :n_components] * np.sqrt(w[None, :])


def procrustes_similarity_from_dm(D_true: np.ndarray, D_pred: np.ndarray, n_components: int = 10) -> float:
    good = np.isfinite(D_true).all(axis=1) & np.isfinite(D_pred).all(axis=1)
    if good.sum() < 3:
        return np.nan
    D_true = D_true[good][:, good]
    D_pred = D_pred[good][:, good]
    n = D_true.shape[0]
    k = max(2, min(int(n_components), n - 1))
    X = _cmdscale(D_true, k)
    Y = _cmdscale(D_pred, k)
    _, _, disparity = procrustes(X, Y)
    return float(1.0 - disparity)


def feature_weights_from_variance(Y_clr: pd.DataFrame, eps: float = 1e-12) -> np.ndarray:
    v = np.nanvar(Y_clr.to_numpy(float), axis=0)
    w = np.sqrt(v + eps)
    w = w / np.median(w[w > 0])
    return w


def weighted_clr_mse(Y_true_clr: pd.DataFrame, Y_pred_clr: pd.DataFrame, w: np.ndarray) -> float:
    T = Y_true_clr.to_numpy(float)
    P = Y_pred_clr.reindex_like(Y_true_clr).to_numpy(float)
    w = np.asarray(w, float)[None, :]
    return float(np.nanmean(((T - P) ** 2) * w))


def jsd_rows(T: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> float:
    T = np.clip(T, 0, None)
    P = np.clip(P, 0, None)
    T = T / (T.sum(axis=1, keepdims=True) + eps)
    P = P / (P.sum(axis=1, keepdims=True) + eps)
    M = 0.5 * (T + P)

    def kl(A, B):
        A = np.clip(A, eps, None)
        B = np.clip(B, eps, None)
        return np.sum(A * np.log(A / B), axis=1)

    jsd = 0.5 * kl(T, M) + 0.5 * kl(P, M)
    return float(np.mean(jsd))


def collapse_by_group(Y: pd.DataFrame, ko_to_group: dict) -> pd.DataFrame:
    groups = [ko_to_group.get(c, "UNMAPPED") for c in Y.columns]
    out = Y.copy()
    out.columns = groups
    return out.T.groupby(level=0).sum().T



def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def pathway_rmse_tss(
    Y_true_tss: pd.DataFrame,
    Y_pred_tss: pd.DataFrame,
    ko_to_group: dict,
    log1p: bool = True,
) -> float:
    T_pw = collapse_by_group(Y_true_tss, ko_to_group)
    P_pw = collapse_by_group(Y_pred_tss, ko_to_group).reindex_like(T_pw).fillna(0.0)
    T = T_pw.to_numpy(float)
    P = P_pw.to_numpy(float)
    if log1p:
        T = np.log1p(T)
        P = np.log1p(P)
    return rmse(T, P)
