from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression


def median_nn_distance(Z: np.ndarray, k: int = 10) -> float:
    D = cdist(Z, Z, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    k_eff = min(int(k), Z.shape[0] - 1)
    nn = np.partition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    return float(np.median(nn))


def knn_kernel_predict_tau_abs(
    Z_tr: np.ndarray,
    Z_te: np.ndarray,
    T_tr: np.ndarray,
    *,
    k: int,
    tau_abs: float,
    lam: float,
) -> np.ndarray:
    D = cdist(Z_te, Z_tr, metric="euclidean")
    k_eff = min(int(k), Z_tr.shape[0])
    idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    out = np.zeros((Z_te.shape[0], T_tr.shape[1]), dtype=float)
    tbar = T_tr.mean(axis=0, keepdims=True)

    tau = float(max(tau_abs, 1e-12))
    lam = float(lam)

    for i in range(Z_te.shape[0]):
        jj = idx[i]
        di = D[i, jj]
        w = np.exp(-(di**2) / (2.0 * tau**2))
        sw = w.sum()
        if sw <= 0 or not np.isfinite(sw):
            pred = tbar[0]
        else:
            pred = (w[:, None] * T_tr[jj]).sum(axis=0) / sw
        out[i] = (1.0 - lam) * pred + lam * tbar[0]

    return out


def fit_y_latent_svd(Y_clr: pd.DataFrame, k: int = 20, random_state: int = 0):
    Y = Y_clr.to_numpy(dtype=float)
    col_mean = np.nanmean(Y, axis=0, keepdims=True)
    Yc = Y - col_mean

    k_eff = min(int(k), min(Yc.shape) - 1)
    k_eff = max(k_eff, 2)

    svd = TruncatedSVD(n_components=k_eff, random_state=int(random_state))
    svd.fit(Yc)
    return svd, col_mean


def encode_y_latent(Y_clr: pd.DataFrame, svd, col_mean: np.ndarray) -> np.ndarray:
    Y = Y_clr.to_numpy(dtype=float)
    return svd.transform(Y - col_mean)


def decode_y_latent(Z_latent: np.ndarray, svd, col_mean: np.ndarray) -> np.ndarray:
    Yc_hat = svd.inverse_transform(Z_latent)
    Y_hat = Yc_hat + col_mean
    Y_hat = Y_hat - Y_hat.mean(axis=1, keepdims=True)
    return Y_hat


def fit_supervised_diag_metric(
    X_clr: pd.DataFrame,
    Y_clr: pd.DataFrame,
    *,
    max_pairs: int = 20000,
    random_state: int = 0,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Fit nonnegative diagonal weights w so that squared distances in X match squared distances in Y.
    """
    rng = np.random.default_rng(int(random_state))
    Zx = X_clr.to_numpy(dtype=float)
    Zy = Y_clr.to_numpy(dtype=float)

    n, p = Zx.shape
    all_pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    if all_pairs.shape[0] > int(max_pairs):
        sel = rng.choice(all_pairs.shape[0], size=int(max_pairs), replace=False)
        pairs = all_pairs[sel]
    else:
        pairs = all_pairs

    i = pairs[:, 0]
    j = pairs[:, 1]
    dx = Zx[i] - Zx[j]
    dy = Zy[i] - Zy[j]

    A = dx**2
    b = (dy**2).sum(axis=1)

    if ridge and ridge > 0:
        A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(p)])
        b_aug = np.concatenate([b, np.zeros(p)])
    else:
        A_aug, b_aug = A, b

    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(A_aug, b_aug)
    w = reg.coef_.astype(float)

    if not np.isfinite(w).any() or w.sum() <= 0:
        w = np.ones(p, dtype=float)

    w = w / (np.median(w[w > 0]) if np.any(w > 0) else 1.0)
    return w


def apply_ood_shrinkage(
    Yhat_clr_arr: np.ndarray,
    Ytr_clr: pd.DataFrame,
    nn_min: np.ndarray,
    lam_base: float = 0.1,
    lam_cap: float = 0.8,
    *,
    return_lam: bool = False,
):
    ybar = Ytr_clr.mean(axis=0).to_numpy(dtype=float)[None, :]
    s = float(np.median(nn_min) + 1e-12)
    z = nn_min / s
    lam_vec = np.clip(float(lam_base) * z, 0.0, float(lam_cap))
    Ymix = (1.0 - lam_vec[:, None]) * Yhat_clr_arr + lam_vec[:, None] * ybar
    if return_lam:
        return Ymix, lam_vec
    return Ymix


def prf_thresholded(
    T: np.ndarray,
    P: np.ndarray,
    thresh: float = 1e-6,
    weight: str = "binary",
    eps: float = 1e-12,
):
    T = np.clip(T, 0, None)
    P = np.clip(P, 0, None)

    Tpos = (T >= thresh)
    Ppos = (P >= thresh)

    if weight == "binary":
        wT = np.ones_like(T)
        wP = np.ones_like(P)
    elif weight == "truth_abundance":
        wT = T
        wP = T
    elif weight == "pred_abundance":
        wT = P
        wP = P
    else:
        raise ValueError("weight must be 'binary', 'truth_abundance', or 'pred_abundance'")

    tp = (Tpos & Ppos) * wT
    fp = (~Tpos & Ppos) * wP
    fn = (Tpos & ~Ppos) * wT

    tp_s = tp.sum(axis=1)
    fp_s = fp.sum(axis=1)
    fn_s = fn.sum(axis=1)

    prec = tp_s / (tp_s + fp_s + eps)
    rec = tp_s / (tp_s + fn_s + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1))
