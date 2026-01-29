from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import spearmanr, norm

from .transforms import clr_rows
from .knn_core import prf_thresholded


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


def samplewise_spearman(
    A: pd.DataFrame, B: pd.DataFrame, *, min_non_nan: int = 10
) -> pd.Series:
    """
    Compute per-sample Spearman correlation across features.
    A and B must be aligned on index/columns.
    """
    if not A.index.equals(B.index):
        B = B.reindex(index=A.index)
    if not A.columns.equals(B.columns):
        B = B.reindex(columns=A.columns)

    a = A.to_numpy(float)
    b = B.to_numpy(float)
    out = np.full(a.shape[0], np.nan, dtype=float)
    for i in range(a.shape[0]):
        m = np.isfinite(a[i]) & np.isfinite(b[i])
        if m.sum() < min_non_nan:
            continue
        out[i] = float(spearmanr(a[i, m], b[i, m]).correlation)
    return pd.Series(out, index=A.index, name="sample_spearman")


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


def featurewise_oof_spearman(
    Y_true_clr: pd.DataFrame, Y_pred_clr: pd.DataFrame, *, min_n: int = 10
) -> tuple[pd.Series, pd.Series]:
    """
    Per-feature Spearman between truth and predictions in CLR space.
    Returns (rho, n_eff) as Series aligned to features.
    """
    if not Y_true_clr.index.equals(Y_pred_clr.index):
        Y_pred_clr = Y_pred_clr.reindex(index=Y_true_clr.index)
    if not Y_true_clr.columns.equals(Y_pred_clr.columns):
        Y_pred_clr = Y_pred_clr.reindex(columns=Y_true_clr.columns)

    rho = {}
    n_eff = {}
    for c in Y_true_clr.columns:
        a = Y_true_clr[c].to_numpy(float)
        b = Y_pred_clr[c].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        n = int(m.sum())
        n_eff[c] = n
        if n < int(min_n):
            rho[c] = np.nan
        else:
            rho[c] = float(spearmanr(a[m], b[m]).correlation)
    return pd.Series(rho, name="oof_spearman"), pd.Series(n_eff, name="n_eff")


def prob_r_ge_r0(
    r: np.ndarray, n: np.ndarray, *, r0: float = 0.30, eps: float = 1e-12
) -> np.ndarray:
    """
    Approximate P(r >= r0) using Fisher z. (Approximate for Spearman.)
    r and n can be arrays.
    """
    r = np.asarray(r, float)
    n = np.asarray(n, float)
    out = np.full_like(r, np.nan, dtype=float)
    m = np.isfinite(r) & np.isfinite(n) & (n >= 10)
    if m.sum() == 0:
        return out
    r_clip = np.clip(r[m], -1 + 1e-6, 1 - 1e-6)
    z = np.arctanh(r_clip)
    z0 = np.arctanh(np.clip(float(r0), -1 + 1e-6, 1 - 1e-6))
    se = 1.0 / np.sqrt(np.maximum(n[m] - 3.0, 1.0))
    out[m] = 1.0 - norm.cdf((z0 - z) / (se + eps))
    return out


def knn_indices(Z: np.ndarray, *, k: int = 10, exclude_self: bool = True) -> np.ndarray:
    Z = np.asarray(Z, float)
    D = cdist(Z, Z, metric="euclidean")
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def neighbor_var_per_feature(Yarr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Yarr: (n, p), idx: (n, k) -> returns (p,) mean_i Var_{neighbors}(Y)
    """
    Yn = Yarr[idx]  # (n, k, p)
    mean = Yn.mean(axis=1)
    mean_sq = (Yn * Yn).mean(axis=1)
    var = mean_sq - mean * mean
    return var.mean(axis=0)


def stability_confidence_from_null(
    Y_clr: pd.DataFrame,
    *,
    Z: np.ndarray | None = None,
    nn_idx: np.ndarray | None = None,
    k_nn: int = 10,
    R: int = 40,
    seed: int = 0,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Confidence in [0,1] where 1 means local dispersion is low vs null.
    Uses log-variance of neighbor dispersion vs random-neighbor null.
    """
    if nn_idx is None:
        if Z is None:
            raise ValueError("Provide either Z or nn_idx.")
        nn_idx = knn_indices(Z, k=int(k_nn), exclude_self=True)

    rng = np.random.default_rng(int(seed))
    Yarr = Y_clr.to_numpy(np.float32)
    n, p = Yarr.shape
    k = nn_idx.shape[1]

    v_obs = neighbor_var_per_feature(Yarr, nn_idx)
    lv_obs = np.log(v_obs + eps)

    lv_null = np.empty((int(R), p), dtype=np.float32)
    all_idx = np.arange(n)
    for r in range(int(R)):
        ridx = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            choices = np.delete(all_idx, i)
            ridx[i] = rng.choice(choices, size=k, replace=False)
        v_r = neighbor_var_per_feature(Yarr, ridx)
        lv_null[r] = np.log(v_r + eps)

    mu = lv_null.mean(axis=0)
    sd = lv_null.std(axis=0, ddof=1)
    sd = np.maximum(sd, 1e-6)
    z = (lv_obs - mu) / sd
    conf_stab = norm.cdf(-z)
    return pd.Series(conf_stab.astype(float), index=Y_clr.columns, name=f"conf_stab_k{k}")


def ko_confidence_from_oof(
    Y_true_clr: pd.DataFrame,
    Y_pred_clr: pd.DataFrame,
    *,
    Z: np.ndarray,
    r0: float = 0.30,
    k_nn: int = 10,
    R: int = 40,
    seed: int = 0,
    min_n: int = 10,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Combined KO confidence in [0,1] based on OOF Spearman and local stability.
    Returns columns: oof_spearman, n_eff, conf_corr, conf_stab, confidence.
    """
    rho, n_eff = featurewise_oof_spearman(Y_true_clr, Y_pred_clr, min_n=int(min_n))
    conf_corr = prob_r_ge_r0(rho.to_numpy(float), n_eff.to_numpy(float), r0=float(r0), eps=float(eps))
    conf_corr = pd.Series(conf_corr, index=Y_true_clr.columns, name="conf_corr")
    conf_stab = stability_confidence_from_null(
        Y_true_clr, Z=Z, k_nn=int(k_nn), R=int(R), seed=int(seed), eps=float(eps)
    )
    S = pd.DataFrame(
        {
            "oof_spearman": rho,
            "n_eff": n_eff,
            "conf_corr": conf_corr,
            "conf_stab": conf_stab,
        }
    )
    S["confidence"] = S["conf_corr"] * S["conf_stab"]
    return S


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


def pathway_rmse_tss_per_group(
    Y_true_tss: pd.DataFrame,
    Y_pred_tss: pd.DataFrame,
    ko_to_group: dict,
    log1p: bool = True,
) -> pd.Series:
    T_pw = collapse_by_group(Y_true_tss, ko_to_group)
    P_pw = collapse_by_group(Y_pred_tss, ko_to_group).reindex_like(T_pw).fillna(0.0)
    T = T_pw.to_numpy(float)
    P = P_pw.to_numpy(float)
    if log1p:
        T = np.log1p(T)
        P = np.log1p(P)
    rmse_vals = np.sqrt(np.nanmean((T - P) ** 2, axis=0))
    return pd.Series(rmse_vals, index=T_pw.columns, dtype=float)


def _pairwise_union_mats_tss(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float,
    fillna_zero: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = truth_tpm.index.intersection(pred_tpm.index)
    truth = truth_tpm.loc[common]
    pred = pred_tpm.loc[common]
    union_cols = truth.columns.union(pred.columns)
    truth_u = truth.reindex(columns=union_cols, fill_value=0.0)
    pred_u = pred.reindex(columns=union_cols, fill_value=0.0)
    if fillna_zero:
        truth_u = truth_u.fillna(0.0)
        pred_u = pred_u.fillna(0.0)
    truth_tss_u = truth_u.div(truth_u.sum(axis=1).replace(0, np.nan), axis=0)
    pred_tss_u = pred_u.div(pred_u.sum(axis=1).replace(0, np.nan), axis=0)
    if detect_threshold and detect_threshold > 0:
        thr_rel = pd.Series(0.0, index=truth_tss_u.index, dtype=float)
        truth_sum = truth_u.sum(axis=1)
        nonzero = truth_sum > 0
        thr_rel.loc[nonzero] = detect_threshold / truth_sum.loc[nonzero]
        truth_tss_u = truth_tss_u.mask(truth_tss_u.lt(thr_rel, axis=0), 0.0)
        pred_tss_u = pred_tss_u.mask(pred_tss_u.lt(thr_rel, axis=0), 0.0)
    good = (truth_tss_u.sum(axis=1) > 0) & (pred_tss_u.sum(axis=1) > 0)
    return truth_tss_u.loc[good], pred_tss_u.loc[good]


def _pairwise_intersection_mats_tss(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_samples = truth_tpm.index.intersection(pred_tpm.index)
    truth = truth_tpm.loc[common_samples]
    pred = pred_tpm.loc[common_samples]
    common_cols = truth.columns.intersection(pred.columns)
    truth_i = truth.loc[:, common_cols]
    pred_i = pred.loc[:, common_cols]
    truth_tss = truth_i.div(truth_i.sum(axis=1).replace(0, np.nan), axis=0)
    pred_tss = pred_i.div(pred_i.sum(axis=1).replace(0, np.nan), axis=0)
    if detect_threshold and detect_threshold > 0:
        thr_rel = pd.Series(0.0, index=truth_tss.index, dtype=float)
        truth_sum = truth_i.sum(axis=1)
        nonzero = truth_sum > 0
        thr_rel.loc[nonzero] = detect_threshold / truth_sum.loc[nonzero]
        truth_tss = truth_tss.mask(truth_tss.lt(thr_rel, axis=0), 0.0)
        pred_tss = pred_tss.mask(pred_tss.lt(thr_rel, axis=0), 0.0)
    good = (truth_tss.sum(axis=1) > 0) & (pred_tss.sum(axis=1) > 0)
    return truth_tss.loc[good], pred_tss.loc[good]


def dm_spearman_intersection(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float = 0.0,
) -> float:
    truth_tss_i, pred_tss_i = _pairwise_intersection_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold
    )
    truth_clr = clr_rows(truth_tss_i, pseudocount=pseudocount)
    pred_clr = clr_rows(pred_tss_i, pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    if truth_clr.shape[0] < 3:
        return np.nan
    return float(corr_upper_triangle(aitchison_dm(truth_clr), aitchison_dm(pred_clr), method="spearman"))


def bray_spearman_intersection(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float = 0.0,
) -> float:
    truth_tss_i, pred_tss_i = _pairwise_intersection_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold
    )
    if truth_tss_i.shape[0] < 3:
        return np.nan
    return float(corr_upper_triangle(bray_curtis_dm(truth_tss_i), bray_curtis_dm(pred_tss_i), method="spearman"))


def procrustes_intersection_aitchison(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float = 0.0,
) -> float:
    truth_tss_i, pred_tss_i = _pairwise_intersection_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold
    )
    truth_clr = clr_rows(truth_tss_i, pseudocount=pseudocount)
    pred_clr = clr_rows(pred_tss_i, pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    return float(
        procrustes_similarity_from_dm(aitchison_dm(truth_clr), aitchison_dm(pred_clr), n_components=10)
    )


def procrustes_intersection_bray(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float = 0.0,
) -> float:
    truth_tss_i, pred_tss_i = _pairwise_intersection_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold
    )
    return float(
        procrustes_similarity_from_dm(bray_curtis_dm(truth_tss_i), bray_curtis_dm(pred_tss_i), n_components=10)
    )


def evaluate_intersection_metrics(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float = 0.0,
    prf_thresh: float = 1e-6,
    prf_weight: str = "binary",
    compute_wclr: bool = False,
    compute_jsd: bool = False,
    compute_pathway: bool = False,
    compute_per_pathway: bool = False,
    ko_to_group: dict | None = None,
    log1p_pathway: bool = True,
) -> dict:
    truth_tss_i, pred_tss_i = _pairwise_intersection_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold
    )
    out = {}
    if truth_tss_i.shape[0] < 3:
        return {
            "dm_spearman": np.nan,
            "bray_spearman": np.nan,
            "procrustes_aitchison": np.nan,
            "procrustes_bray": np.nan,
            "soft_precision": np.nan,
            "soft_recall": np.nan,
            "soft_f1": np.nan,
            "wclr_mse": np.nan if compute_wclr else 0.0,
            "jsd": np.nan if compute_jsd else 0.0,
            "pathway_rmse": np.nan if compute_pathway else 0.0,
        }

    truth_clr = clr_rows(truth_tss_i, pseudocount=pseudocount)
    pred_clr = clr_rows(pred_tss_i, pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    truth_tss_i = truth_tss_i.loc[good]
    pred_tss_i = pred_tss_i.loc[good]

    if truth_clr.shape[0] < 3:
        return {
            "dm_spearman": np.nan,
            "bray_spearman": np.nan,
            "procrustes_aitchison": np.nan,
            "procrustes_bray": np.nan,
            "soft_precision": np.nan,
            "soft_recall": np.nan,
            "soft_f1": np.nan,
            "wclr_mse": np.nan if compute_wclr else 0.0,
            "jsd": np.nan if compute_jsd else 0.0,
            "pathway_rmse": np.nan if compute_pathway else 0.0,
        }

    out["dm_spearman"] = float(
        corr_upper_triangle(aitchison_dm(truth_clr), aitchison_dm(pred_clr), method="spearman")
    )
    out["bray_spearman"] = float(
        corr_upper_triangle(bray_curtis_dm(truth_tss_i), bray_curtis_dm(pred_tss_i), method="spearman")
    )
    out["procrustes_aitchison"] = float(
        procrustes_similarity_from_dm(aitchison_dm(truth_clr), aitchison_dm(pred_clr), n_components=10)
    )
    out["procrustes_bray"] = float(
        procrustes_similarity_from_dm(bray_curtis_dm(truth_tss_i), bray_curtis_dm(pred_tss_i), n_components=10)
    )

    sp, sr, sf1 = prf_thresholded(
        truth_tss_i.to_numpy(float),
        pred_tss_i.to_numpy(float),
        thresh=float(prf_thresh),
        weight=str(prf_weight),
    )
    out["soft_precision"] = float(sp)
    out["soft_recall"] = float(sr)
    out["soft_f1"] = float(sf1)

    if compute_wclr:
        w_feat = feature_weights_from_variance(truth_clr)
        out["wclr_mse"] = float(weighted_clr_mse(truth_clr, pred_clr, w_feat))
    if compute_jsd:
        out["jsd"] = float(jsd_rows(truth_tss_i.to_numpy(float), pred_tss_i.to_numpy(float)))
    if compute_pathway:
        if ko_to_group is None:
            raise ValueError("ko_to_group is required for pathway RMSE.")
        out["pathway_rmse"] = float(
            pathway_rmse_tss(
                Y_true_tss=truth_tss_i,
                Y_pred_tss=pred_tss_i,
                ko_to_group=ko_to_group,
                log1p=bool(log1p_pathway),
            )
        )
        if compute_per_pathway:
            out["pathway_rmse_per_group"] = pathway_rmse_tss_per_group(
                Y_true_tss=truth_tss_i,
                Y_pred_tss=pred_tss_i,
                ko_to_group=ko_to_group,
                log1p=bool(log1p_pathway),
            )
    return out

def dm_spearman_union(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float,
    fillna_zero: bool,
) -> float:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )
    truth_clr = clr_rows(truth_tss_u, pseudocount=pseudocount)
    pred_clr = clr_rows(pred_tss_u, pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    if truth_clr.shape[0] < 3:
        return np.nan
    return float(corr_upper_triangle(aitchison_dm(truth_clr), aitchison_dm(pred_clr), method="spearman"))


def bray_spearman_union(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float,
    fillna_zero: bool,
) -> float:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )
    if truth_tss_u.shape[0] < 3:
        return np.nan
    return float(corr_upper_triangle(bray_curtis_dm(truth_tss_u), bray_curtis_dm(pred_tss_u), method="spearman"))


def procrustes_union_aitchison(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float,
    fillna_zero: bool,
) -> float:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )
    truth_clr = clr_rows(truth_tss_u, pseudocount=pseudocount)
    pred_clr = clr_rows(pred_tss_u, pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    return float(
        procrustes_similarity_from_dm(aitchison_dm(truth_clr), aitchison_dm(pred_clr), n_components=10)
    )


def procrustes_union_bray(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float,
    fillna_zero: bool,
) -> float:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )
    return float(
        procrustes_similarity_from_dm(bray_curtis_dm(truth_tss_u), bray_curtis_dm(pred_tss_u), n_components=10)
    )


def evaluate_union_metrics(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float,
    prf_thresh: float = 1e-6,
    prf_weight: str = "binary",
    fillna_zero: bool = True,
    compute_wclr: bool = False,
    compute_jsd: bool = False,
    compute_pathway: bool = False,
    compute_per_pathway: bool = False,
    ko_to_group: dict | None = None,
    log1p_pathway: bool = True,
) -> dict:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )

    out = {}
    if truth_tss_u.shape[0] < 3:
        return {
            "dm_spearman": np.nan,
            "bray_spearman": np.nan,
            "procrustes_aitchison": np.nan,
            "procrustes_bray": np.nan,
            "soft_precision": np.nan,
            "soft_recall": np.nan,
            "soft_f1": np.nan,
            "wclr_mse": np.nan if compute_wclr else 0.0,
            "jsd": np.nan if compute_jsd else 0.0,
            "pathway_rmse": np.nan if compute_pathway else 0.0,
        }

    truth_clr = clr_rows(truth_tss_u, pseudocount=pseudocount)
    pred_clr = clr_rows(pred_tss_u, pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    truth_tss_u = truth_tss_u.loc[good]
    pred_tss_u = pred_tss_u.loc[good]

    if truth_clr.shape[0] < 3:
        return {
            "dm_spearman": np.nan,
            "bray_spearman": np.nan,
            "procrustes_aitchison": np.nan,
            "procrustes_bray": np.nan,
            "soft_precision": np.nan,
            "soft_recall": np.nan,
            "soft_f1": np.nan,
            "wclr_mse": np.nan if compute_wclr else 0.0,
            "jsd": np.nan if compute_jsd else 0.0,
            "pathway_rmse": np.nan if compute_pathway else 0.0,
        }

    out["dm_spearman"] = float(
        corr_upper_triangle(aitchison_dm(truth_clr), aitchison_dm(pred_clr), method="spearman")
    )
    out["bray_spearman"] = float(
        corr_upper_triangle(bray_curtis_dm(truth_tss_u), bray_curtis_dm(pred_tss_u), method="spearman")
    )
    out["procrustes_aitchison"] = float(
        procrustes_similarity_from_dm(aitchison_dm(truth_clr), aitchison_dm(pred_clr), n_components=10)
    )
    out["procrustes_bray"] = float(
        procrustes_similarity_from_dm(bray_curtis_dm(truth_tss_u), bray_curtis_dm(pred_tss_u), n_components=10)
    )

    sp, sr, sf1 = prf_thresholded(
        truth_tss_u.to_numpy(float),
        pred_tss_u.to_numpy(float),
        thresh=float(prf_thresh),
        weight=str(prf_weight),
    )
    out["soft_precision"] = float(sp)
    out["soft_recall"] = float(sr)
    out["soft_f1"] = float(sf1)

    if compute_wclr:
        w_feat = feature_weights_from_variance(truth_clr)
        out["wclr_mse"] = float(weighted_clr_mse(truth_clr, pred_clr, w_feat))
    if compute_jsd:
        out["jsd"] = float(jsd_rows(truth_tss_u.to_numpy(float), pred_tss_u.to_numpy(float)))
    if compute_pathway:
        if ko_to_group is None:
            raise ValueError("ko_to_group is required for pathway RMSE.")
        out["pathway_rmse"] = float(
            pathway_rmse_tss(
                Y_true_tss=truth_tss_u,
                Y_pred_tss=pred_tss_u,
                ko_to_group=ko_to_group,
                log1p=bool(log1p_pathway),
            )
        )
        if compute_per_pathway:
            out["pathway_rmse_per_group"] = pathway_rmse_tss_per_group(
                Y_true_tss=truth_tss_u,
                Y_pred_tss=pred_tss_u,
                ko_to_group=ko_to_group,
                log1p=bool(log1p_pathway),
            )
    return out
