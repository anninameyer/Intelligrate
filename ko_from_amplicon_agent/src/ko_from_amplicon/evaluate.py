from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

from .cv_knn import nested_cv_knn_metric_latent_on_embedding
from .embedding import fit_x_embedding_svd_clr, transform_x_embedding_svd_clr
from .knn_core import (
    apply_ood_shrinkage,
    decode_y_latent,
    encode_y_latent,
    fit_supervised_diag_metric,
    fit_y_latent_svd,
    knn_kernel_predict_tau_abs,
    median_nn_distance,
)
from .metrics import aitchison_dm, bray_curtis_dm, corr_upper_triangle, procrustes_similarity_from_dm
from .transforms import clr_rows, clr_to_comp, keep_by_prevalence, tss_rows


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def _read_ko_to_superclass(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t")
    # expects columns: KO, superclass (or similar)
    # use first two columns if not named
    if df.shape[1] < 2:
        raise ValueError("ko_to_superclass.tsv must have at least two columns.")
    ko_col = df.columns[0]
    sc_col = df.columns[1]
    return dict(zip(df[ko_col].astype(str), df[sc_col].astype(str)))


def _pairwise_union_mats_tss(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    *,
    detect_threshold: float,
    fillna_zero: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Pairwise union only for the two inputs (no global/shared union across methods).
    common = truth_tpm.index.intersection(pred_tpm.index)
    truth = truth_tpm.loc[common]
    pred = pred_tpm.loc[common]
    union_cols = truth.columns.union(pred.columns)
    truth_u = truth.reindex(columns=union_cols, fill_value=0.0)
    pred_u = pred.reindex(columns=union_cols, fill_value=0.0)
    if fillna_zero:
        truth_u = truth_u.fillna(0.0)
        pred_u = pred_u.fillna(0.0)
    truth_tss_u = tss_rows(truth_u)
    pred_tss_u = tss_rows(pred_u)
    if detect_threshold and detect_threshold > 0:
        thr_rel = pd.Series(0.0, index=truth_tss_u.index, dtype=float)
        truth_sum = truth_u.sum(axis=1)
        nonzero = truth_sum > 0
        thr_rel.loc[nonzero] = detect_threshold / truth_sum.loc[nonzero]
        truth_tss_u = truth_tss_u.mask(truth_tss_u.lt(thr_rel, axis=0), 0.0)
        pred_tss_u = pred_tss_u.mask(pred_tss_u.lt(thr_rel, axis=0), 0.0)
    good = (truth_tss_u.sum(axis=1) > 0) & (pred_tss_u.sum(axis=1) > 0)
    return truth_tss_u.loc[good], pred_tss_u.loc[good]


def _dm_spearman_union(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    pseudocount: float,
    *,
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


def _bray_spearman_union(
    truth_tpm: pd.DataFrame, pred_tpm: pd.DataFrame, *, detect_threshold: float, fillna_zero: bool
) -> float:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )
    if truth_tss_u.shape[0] < 3:
        return np.nan
    return float(corr_upper_triangle(bray_curtis_dm(truth_tss_u), bray_curtis_dm(pred_tss_u), method="spearman"))


def _procrustes_union_aitchison(
    truth_tpm: pd.DataFrame,
    pred_tpm: pd.DataFrame,
    pseudocount: float,
    *,
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


def _procrustes_union_bray(
    truth_tpm: pd.DataFrame, pred_tpm: pd.DataFrame, *, detect_threshold: float, fillna_zero: bool
) -> float:
    truth_tss_u, pred_tss_u = _pairwise_union_mats_tss(
        truth_tpm, pred_tpm, detect_threshold=detect_threshold, fillna_zero=fillna_zero
    )
    return float(
        procrustes_similarity_from_dm(bray_curtis_dm(truth_tss_u), bray_curtis_dm(pred_tss_u), n_components=10)
    )


def _mode_or_default(series: pd.Series | None, default):
    if series is None:
        return default
    vals = series.dropna()
    if vals.empty:
        return default
    mode = vals.mode()
    return mode.iloc[0] if len(mode) else default


def _knn_kernel_predict_tau_abs_from_distance(
    D: np.ndarray,
    T_tr: np.ndarray,
    *,
    k: int,
    tau_abs: float,
    lam: float,
) -> np.ndarray:
    k_eff = min(int(k), D.shape[1])
    idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    out = np.zeros((D.shape[0], T_tr.shape[1]), dtype=float)
    tbar = T_tr.mean(axis=0, keepdims=True)

    tau = float(max(tau_abs, 1e-12))
    lam = float(lam)

    for i in range(D.shape[0]):
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


def _full_fit_knn_predict(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    embed: dict,
    min_prev_y_abs: int,
    y_detect_threshold: float,
    pseudocount_y: float,
    neigh_k: int,
    tau_mult: float,
    lam: float,
    y_latent_k: int,
    use_metric_learning: bool,
    metric_max_pairs: int,
    metric_ridge: float,
    tau_scale_k_nn: int,
    ood_shrink: bool,
    ood_lam_base: float,
    ood_lam_cap: float,
    ood_tau_inflate: bool,
    ood_tau_gamma: float,
    seed: int,
) -> pd.DataFrame:
    y_keep = keep_by_prevalence(Y, min_prev_abs=int(min_prev_y_abs), detect_threshold=float(y_detect_threshold))
    Y0 = Y.loc[:, y_keep]
    Y_tss = tss_rows(Y0).fillna(0.0)
    Y_clr = clr_rows(Y_tss, pseudocount=float(pseudocount_y))

    Z = transform_x_embedding_svd_clr(X, embed)
    if use_metric_learning:
        X_df = pd.DataFrame(Z, index=Y_clr.index)
        w = fit_supervised_diag_metric(
            X_clr=X_df,
            Y_clr=Y_clr,
            max_pairs=int(metric_max_pairs),
            random_state=int(seed + 9000),
            ridge=float(metric_ridge),
        )
        Z = Z * np.sqrt(w[None, :])

    scale = median_nn_distance(Z, k=min(int(tau_scale_k_nn), Z.shape[0] - 1))
    tau_abs = float(tau_mult) * float(scale)

    D = cdist(Z, Z, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn_min = D.min(axis=1)

    if ood_tau_inflate:
        z_ood = float(np.median(nn_min)) / (float(scale) + 1e-12)
        tau_abs_eff = tau_abs * (1.0 + float(ood_tau_gamma) * z_ood)
    else:
        tau_abs_eff = tau_abs

    if int(y_latent_k) > 0:
        svd, col_mean = fit_y_latent_svd(Y_clr, k=int(y_latent_k), random_state=int(seed + 9100))
        T_tr = encode_y_latent(Y_clr, svd, col_mean)
        T_hat = _knn_kernel_predict_tau_abs_from_distance(
            D, T_tr, k=int(neigh_k), tau_abs=float(tau_abs_eff), lam=float(lam)
        )
        Yhat_arr = decode_y_latent(T_hat, svd, col_mean)
    else:
        T_tr = Y_clr.to_numpy(dtype=float)
        Yhat_arr = _knn_kernel_predict_tau_abs_from_distance(
            D, T_tr, k=int(neigh_k), tau_abs=float(tau_abs_eff), lam=float(lam)
        )
        Yhat_arr = Yhat_arr - Yhat_arr.mean(axis=1, keepdims=True)

    if ood_shrink:
        Yhat_arr = apply_ood_shrinkage(
            Yhat_clr_arr=Yhat_arr,
            Ytr_clr=Y_clr,
            nn_min=nn_min,
            lam_base=float(ood_lam_base),
            lam_cap=float(ood_lam_cap),
        )
        Yhat_arr = Yhat_arr - Yhat_arr.mean(axis=1, keepdims=True)

    return pd.DataFrame(Yhat_arr, index=Y_clr.index, columns=y_keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    repo_root = Path(".").resolve()
    data_dir = repo_root / "data"
    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full = _read_table(data_dir / cfg["data"]["x_full"])
    X = _read_table(data_dir / cfg["data"]["x"])
    Y = _read_table(data_dir / cfg["data"]["y"])
    ko_to_super = _read_ko_to_superclass(data_dir / cfg["data"]["ko_to_superclass"])
    picrust2_path = cfg["data"].get("picrust2")
    picrust2 = _read_table(data_dir / picrust2_path) if picrust2_path else None

    # Align samples (paired set)
    common = X.index.intersection(Y.index)
    X = X.loc[common].sort_index()
    Y = Y.loc[common].sort_index()
    if picrust2 is not None:
        picrust2 = picrust2.loc[picrust2.index.intersection(common)].sort_index()

    # Embedding fit (simple baseline: fit once on X_full)
    embed = fit_x_embedding_svd_clr(
        X_full=X_full,
        min_prev_x_abs=int(cfg["embed"]["min_prev_x_abs"]),
        pseudocount_x=float(cfg["embed"]["pseudocount_x"]),
        n_components=int(cfg["embed"]["n_components"]),
        seed=int(cfg["cv"]["seed"]),
    )

    t0 = time.time()
    oof_clr, oof_tss, folds = nested_cv_knn_metric_latent_on_embedding(
        X=X,
        Y_tpm=Y,
        embed=embed,
        ko_to_superclass=ko_to_super,
        **cfg["model"],
        **cfg["cv"],
        **cfg["objective"],
        **cfg["prf"],
    )
    dt = time.time() - t0

    # Primary objective: mean outer-fold dm_spearman
    dm_mean = float(folds["dm_spearman"].mean())
    dm_std = float(folds["dm_spearman"].std())

    # Also compute “global OOF DM score” on the OOF table in CLR space (optional sanity check)
    # (Uses prevalence filtering on full Y; this is NOT used for model selection.)
    y_keep = keep_by_prevalence(
        Y, min_prev_abs=int(cfg["score"]["min_prev_y_abs"]), detect_threshold=float(cfg["score"]["y_detect_threshold"])
    )
    Y_clr = clr_rows(tss_rows(Y.loc[:, y_keep]), pseudocount=float(cfg["score"]["pseudocount_y"]))
    P_clr = oof_clr.loc[Y_clr.index, Y_clr.columns]
    good = P_clr.notna().all(axis=1)
    Y_clr = Y_clr.loc[good]
    P_clr = P_clr.loc[good]
    oof_dm = corr_upper_triangle(aitchison_dm(Y_clr), aitchison_dm(P_clr), method="spearman")

    # KO-union DM Spearman (model OOF vs truth; and PICRUSt2 vs truth if provided)
    union_pseudo = float(cfg["score"]["pseudocount_y"])
    union_detect = float(cfg["model"]["y_detect_threshold"])
    model_dm_union_raw = _dm_spearman_union(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    model_dm_union_strict = _dm_spearman_union(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    model_dm_union = _dm_spearman_union(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=False
    )
    model_bray_union_raw = _bray_spearman_union(Y, oof_tss, detect_threshold=0.0, fillna_zero=True)
    model_bray_union = _bray_spearman_union(Y, oof_tss, detect_threshold=union_detect, fillna_zero=True)
    model_bray_union_legacy = _bray_spearman_union(Y, oof_tss, detect_threshold=union_detect, fillna_zero=False)
    model_proc_ait_raw = _procrustes_union_aitchison(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    model_proc_ait = _procrustes_union_aitchison(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    model_proc_ait_legacy = _procrustes_union_aitchison(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=False
    )
    model_proc_bray_raw = _procrustes_union_bray(Y, oof_tss, detect_threshold=0.0, fillna_zero=True)
    model_proc_bray = _procrustes_union_bray(Y, oof_tss, detect_threshold=union_detect, fillna_zero=True)
    model_proc_bray_legacy = _procrustes_union_bray(Y, oof_tss, detect_threshold=union_detect, fillna_zero=False)
    picrust2_dm_union = None
    picrust2_bray_union = None
    picrust2_proc_ait = None
    picrust2_proc_bray = None
    delta_union = None
    if picrust2 is not None:
        picrust2_dm_union_raw = _dm_spearman_union(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
        )
        picrust2_dm_union = _dm_spearman_union(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
        )
        picrust2_dm_union_legacy = _dm_spearman_union(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=False
        )
        picrust2_bray_union_raw = _bray_spearman_union(Y, picrust2, detect_threshold=0.0, fillna_zero=True)
        picrust2_bray_union = _bray_spearman_union(Y, picrust2, detect_threshold=union_detect, fillna_zero=True)
        picrust2_bray_union_legacy = _bray_spearman_union(
            Y, picrust2, detect_threshold=union_detect, fillna_zero=False
        )
        picrust2_proc_ait_raw = _procrustes_union_aitchison(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
        )
        picrust2_proc_ait = _procrustes_union_aitchison(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
        )
        picrust2_proc_ait_legacy = _procrustes_union_aitchison(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=False
        )
        picrust2_proc_bray_raw = _procrustes_union_bray(Y, picrust2, detect_threshold=0.0, fillna_zero=True)
        picrust2_proc_bray = _procrustes_union_bray(Y, picrust2, detect_threshold=union_detect, fillna_zero=True)
        picrust2_proc_bray_legacy = _procrustes_union_bray(
            Y, picrust2, detect_threshold=union_detect, fillna_zero=False
        )
        delta_union = float(model_dm_union - picrust2_dm_union)

    # Full-fit deployment score (fit on all paired samples using mode of CV-selected params)
    def _col_or_none(name: str):
        return folds[name] if name in folds.columns else None

    model_cfg = cfg["model"]
    full_params = {
        "neigh_k": int(
            _mode_or_default(_col_or_none("neigh_k"), int(model_cfg.get("neigh_k_grid", [10])[0]))
        ),
        "tau_mult": float(
            _mode_or_default(_col_or_none("tau_mult"), float(model_cfg.get("tau_mult_grid", [1.0])[0]))
        ),
        "lam": float(_mode_or_default(_col_or_none("lam"), float(model_cfg.get("lam_grid", [0.0])[0]))),
        "y_latent_k": int(
            _mode_or_default(_col_or_none("y_latent_k"), int(model_cfg.get("y_latent_k_grid", [0])[0]))
        ),
        "metric_ridge": float(
            _mode_or_default(
                _col_or_none("metric_ridge"), float(model_cfg.get("metric_ridge_grid", [1.0])[0])
            )
        ),
    }

    full_fit_clr = _full_fit_knn_predict(
        X=X,
        Y=Y,
        embed=embed,
        min_prev_y_abs=int(model_cfg["min_prev_y_abs"]),
        y_detect_threshold=float(model_cfg["y_detect_threshold"]),
        pseudocount_y=float(model_cfg["pseudocount_y"]),
        neigh_k=int(full_params["neigh_k"]),
        tau_mult=float(full_params["tau_mult"]),
        lam=float(full_params["lam"]),
        y_latent_k=int(full_params["y_latent_k"]),
        use_metric_learning=bool(model_cfg["use_metric_learning"]),
        metric_max_pairs=int(model_cfg["metric_max_pairs"]),
        metric_ridge=float(full_params["metric_ridge"]),
        tau_scale_k_nn=int(model_cfg.get("tau_scale_k_nn", 10)),
        ood_shrink=bool(model_cfg.get("ood_shrink", False)),
        ood_lam_base=float(model_cfg.get("ood_lam_base", 0.1)),
        ood_lam_cap=float(model_cfg.get("ood_lam_cap", 0.8)),
        ood_tau_inflate=bool(model_cfg.get("ood_tau_inflate", False)),
        ood_tau_gamma=float(model_cfg.get("ood_tau_gamma", 1.0)),
        seed=int(cfg["cv"]["seed"]),
    )
    full_fit_tss = clr_to_comp(full_fit_clr)
    full_fit_dm_union_raw = _dm_spearman_union(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    full_fit_dm_union = _dm_spearman_union(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    full_fit_dm_union_legacy = _dm_spearman_union(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=False
    )
    full_fit_bray_union_raw = _bray_spearman_union(Y, full_fit_tss, detect_threshold=0.0, fillna_zero=True)
    full_fit_bray_union = _bray_spearman_union(Y, full_fit_tss, detect_threshold=union_detect, fillna_zero=True)
    full_fit_bray_union_legacy = _bray_spearman_union(
        Y, full_fit_tss, detect_threshold=union_detect, fillna_zero=False
    )
    full_fit_proc_ait_raw = _procrustes_union_aitchison(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    full_fit_proc_ait = _procrustes_union_aitchison(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    full_fit_proc_ait_legacy = _procrustes_union_aitchison(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=False
    )
    full_fit_proc_bray_raw = _procrustes_union_bray(Y, full_fit_tss, detect_threshold=0.0, fillna_zero=True)
    full_fit_proc_bray = _procrustes_union_bray(Y, full_fit_tss, detect_threshold=union_detect, fillna_zero=True)
    full_fit_proc_bray_legacy = _procrustes_union_bray(
        Y, full_fit_tss, detect_threshold=union_detect, fillna_zero=False
    )

    run = {
        "objective_dm_spearman_mean": dm_mean,
        "objective_dm_spearman_std": dm_std,
        "oof_dm_spearman": float(oof_dm),
        "model_dm_union_raw": float(model_dm_union_raw),
        "model_dm_union": float(model_dm_union),
        "model_dm_union_strict": float(model_dm_union_strict),
        "model_bray_union_raw": float(model_bray_union_raw),
        "model_bray_union": float(model_bray_union),
        "model_bray_union_legacy": float(model_bray_union_legacy),
        "model_procrustes_aitchison_raw": float(model_proc_ait_raw),
        "model_procrustes_aitchison": float(model_proc_ait),
        "model_procrustes_aitchison_legacy": float(model_proc_ait_legacy),
        "model_procrustes_bray_raw": float(model_proc_bray_raw),
        "model_procrustes_bray": float(model_proc_bray),
        "model_procrustes_bray_legacy": float(model_proc_bray_legacy),
        "picrust2_dm_union_raw": float(picrust2_dm_union_raw) if picrust2_dm_union is not None else None,
        "picrust2_dm_union": float(picrust2_dm_union) if picrust2_dm_union is not None else None,
        "picrust2_dm_union_legacy": float(picrust2_dm_union_legacy) if picrust2_dm_union is not None else None,
        "picrust2_bray_union_raw": float(picrust2_bray_union_raw) if picrust2_bray_union is not None else None,
        "picrust2_bray_union": float(picrust2_bray_union) if picrust2_bray_union is not None else None,
        "picrust2_bray_union_legacy": float(picrust2_bray_union_legacy) if picrust2_bray_union is not None else None,
        "picrust2_procrustes_aitchison_raw": float(picrust2_proc_ait_raw) if picrust2_proc_ait is not None else None,
        "picrust2_procrustes_aitchison": float(picrust2_proc_ait) if picrust2_proc_ait is not None else None,
        "picrust2_procrustes_aitchison_legacy": float(picrust2_proc_ait_legacy) if picrust2_proc_ait is not None else None,
        "picrust2_procrustes_bray_raw": float(picrust2_proc_bray_raw) if picrust2_proc_bray is not None else None,
        "picrust2_procrustes_bray": float(picrust2_proc_bray) if picrust2_proc_bray is not None else None,
        "picrust2_procrustes_bray_legacy": float(picrust2_proc_bray_legacy) if picrust2_proc_bray is not None else None,
        "delta_union": float(delta_union) if delta_union is not None else None,
        "full_fit_dm_union_raw": float(full_fit_dm_union_raw),
        "full_fit_dm_union": float(full_fit_dm_union),
        "full_fit_dm_union_legacy": float(full_fit_dm_union_legacy),
        "full_fit_bray_union_raw": float(full_fit_bray_union_raw),
        "full_fit_bray_union": float(full_fit_bray_union),
        "full_fit_bray_union_legacy": float(full_fit_bray_union_legacy),
        "full_fit_procrustes_aitchison_raw": float(full_fit_proc_ait_raw),
        "full_fit_procrustes_aitchison": float(full_fit_proc_ait),
        "full_fit_procrustes_aitchison_legacy": float(full_fit_proc_ait_legacy),
        "full_fit_procrustes_bray_raw": float(full_fit_proc_bray_raw),
        "full_fit_procrustes_bray": float(full_fit_proc_bray),
        "full_fit_procrustes_bray_legacy": float(full_fit_proc_bray_legacy),
        "full_fit_params": full_params,
        "n_samples": int(len(common)),
        "runtime_sec": float(dt),
        "config": cfg,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    oof_clr.to_csv(out_dir / f"oof_clr_{stamp}.tsv", sep="\t")
    oof_tss.to_csv(out_dir / f"oof_tss_{stamp}.tsv", sep="\t")
    folds.to_csv(out_dir / f"folds_{stamp}.csv", index=False)
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(run, indent=2))

    print(f"OBJECTIVE_DM_SPEARMAN_MEAN={dm_mean:.6f}")
    print(f"OOF_DM_SPEARMAN={float(oof_dm):.6f}")
    print(f"MODEL_DM_UNION_RAW={float(model_dm_union_raw):.6f}")
    print(f"MODEL_DM_UNION={float(model_dm_union):.6f}")
    print(f"MODEL_DM_UNION_STRICT={float(model_dm_union_strict):.6f}")
    print(f"FULL_FIT_DM_UNION={float(full_fit_dm_union):.6f}")
    print(f"MODEL_BRAY_UNION_RAW={float(model_bray_union_raw):.6f}")
    print(f"MODEL_BRAY_UNION={float(model_bray_union):.6f}")
    print(f"MODEL_BRAY_UNION_LEGACY={float(model_bray_union_legacy):.6f}")
    print(f"MODEL_PROC_AITCHISON_RAW={float(model_proc_ait_raw):.6f}")
    print(f"MODEL_PROC_AITCHISON={float(model_proc_ait):.6f}")
    print(f"MODEL_PROC_AITCHISON_LEGACY={float(model_proc_ait_legacy):.6f}")
    print(f"MODEL_PROC_BRAY_RAW={float(model_proc_bray_raw):.6f}")
    print(f"MODEL_PROC_BRAY={float(model_proc_bray):.6f}")
    print(f"MODEL_PROC_BRAY_LEGACY={float(model_proc_bray_legacy):.6f}")
    print(f"FULL_FIT_DM_UNION_RAW={float(full_fit_dm_union_raw):.6f}")
    print(f"FULL_FIT_BRAY_UNION={float(full_fit_bray_union):.6f}")
    print(f"FULL_FIT_BRAY_UNION_RAW={float(full_fit_bray_union_raw):.6f}")
    print(f"FULL_FIT_PROC_AITCHISON={float(full_fit_proc_ait):.6f}")
    print(f"FULL_FIT_PROC_AITCHISON_RAW={float(full_fit_proc_ait_raw):.6f}")
    print(f"FULL_FIT_PROC_BRAY={float(full_fit_proc_bray):.6f}")
    print(f"FULL_FIT_PROC_BRAY_RAW={float(full_fit_proc_bray_raw):.6f}")
    print(f"FULL_FIT_DM_UNION_LEGACY={float(full_fit_dm_union_legacy):.6f}")
    print(f"FULL_FIT_BRAY_UNION_LEGACY={float(full_fit_bray_union_legacy):.6f}")
    print(f"FULL_FIT_PROC_AITCHISON_LEGACY={float(full_fit_proc_ait_legacy):.6f}")
    print(f"FULL_FIT_PROC_BRAY_LEGACY={float(full_fit_proc_bray_legacy):.6f}")
    if picrust2_dm_union is not None:
        print(f"PICRUST2_DM_UNION_RAW={float(picrust2_dm_union_raw):.6f}")
        print(f"PICRUST2_DM_UNION={float(picrust2_dm_union):.6f}")
        print(f"PICRUST2_DM_UNION_LEGACY={float(picrust2_dm_union_legacy):.6f}")
        print(f"PICRUST2_BRAY_UNION_RAW={float(picrust2_bray_union_raw):.6f}")
        print(f"PICRUST2_BRAY_UNION={float(picrust2_bray_union):.6f}")
        print(f"PICRUST2_BRAY_UNION_LEGACY={float(picrust2_bray_union_legacy):.6f}")
        print(f"PICRUST2_PROC_AITCHISON_RAW={float(picrust2_proc_ait_raw):.6f}")
        print(f"PICRUST2_PROC_AITCHISON={float(picrust2_proc_ait):.6f}")
        print(f"PICRUST2_PROC_AITCHISON_LEGACY={float(picrust2_proc_ait_legacy):.6f}")
        print(f"PICRUST2_PROC_BRAY_RAW={float(picrust2_proc_bray_raw):.6f}")
        print(f"PICRUST2_PROC_BRAY={float(picrust2_proc_bray):.6f}")
        print(f"PICRUST2_PROC_BRAY_LEGACY={float(picrust2_proc_bray_legacy):.6f}")
        print(f"DELTA_UNION={float(delta_union):.6f}")
    print(f"RESULTS_PREFIX={stamp}")


if __name__ == "__main__":
    main()
