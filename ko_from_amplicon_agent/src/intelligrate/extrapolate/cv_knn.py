from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold

from .embedding import transform_x_embedding_svd_clr
from .knn_core import (
    apply_ood_shrinkage,
    encode_y_latent,
    decode_y_latent,
    fit_supervised_diag_metric,
    fit_y_latent_svd,
    knn_kernel_predict_tau_abs,
    median_nn_distance,
    prf_thresholded,
)
from .metrics import (
    aitchison_dm,
    corr_upper_triangle,
    feature_weights_from_variance,
    jsd_rows,
    pathway_rmse_tss,
    weighted_clr_mse,
)
from .transforms import clr_rows, clr_to_comp, keep_by_prevalence, tss_rows


def _upper_tri_std(D: np.ndarray) -> float:
    iu = np.triu_indices_from(D, k=1)
    return float(np.std(D[iu]))


def nested_cv_knn_metric_latent_on_embedding(
    X: pd.DataFrame,
    Y_tpm: pd.DataFrame,
    embed: dict,
    *,
    ko_to_superclass: dict,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 0,
    min_prev_y_abs: int = 1,
    y_detect_threshold: float = 1.0,
    pseudocount_y: float = 0.5 / 1e6,
    neigh_k_grid=(4, 6, 8, 12),
    tau_mult_grid=(0.5, 1.0, 2.0, 4.0),
    lam_grid=(0.0, 0.05, 0.1),
    y_latent_k_grid=(0, 10, 20),
    use_metric_learning: bool = True,
    metric_max_pairs: int = 10000,
    metric_ridge_grid=(0.1, 1.0, 10.0),
    # objective weights (you will set w_dm=1.0 in config; others secondary)
    w_dm: float = 1.0,
    w_wclr: float = 0.0,
    w_pw_rmse: float = 0.0,
    w_softf1: float = 0.0,
    w_jsd: float = 0.0,
    tau_scale_k_nn: int = 10,
    ood_shrink: bool = False,
    ood_shrink_inner: bool = True,
    ood_lam_base: float = 0.15,
    ood_lam_cap: float = 0.80,
    ood_tau_inflate: bool = False,
    ood_tau_gamma: float = 1.0,
    informed_splits: bool = False,
    informed_kmeans_on: str = "X",
    informed_kmeans_n_init: int = 20,
    informed_kmeans_k: int | None = None,
    outer_iter_override=None,
    prf_thresh: float = 1e-6,
    prf_weight: str = "binary",
):
    assert X.index.equals(Y_tpm.index), "X and Y must have identical sample index ordering."
    samples = X.index
    n_outer = int(outer_splits)
    need_wclr = float(w_wclr) != 0.0
    need_pw = float(w_pw_rmse) != 0.0
    need_softf1 = float(w_softf1) != 0.0
    need_jsd = float(w_jsd) != 0.0

    if outer_iter_override is not None:
        outer_iter = outer_iter_override
        split_mode = "override"
        k_used = np.nan
    else:
        if not informed_splits:
            outer = KFold(n_splits=n_outer, shuffle=True, random_state=int(seed))
            outer_iter = outer.split(samples)
            split_mode = "kfold"
            k_used = np.nan
        else:
            if informed_kmeans_on != "X":
                raise ValueError("Use informed_kmeans_on='X' to avoid target leakage.")

            Z_all = transform_x_embedding_svd_clr(X, embed)
            k_default = max(2, n_outer // 2)
            k = int(informed_kmeans_k if informed_kmeans_k is not None else k_default)

            while True:
                km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=int(informed_kmeans_n_init))
                labels = km.fit_predict(Z_all)
                counts = np.bincount(labels)
                if counts.min() >= n_outer:
                    break
                if k <= 2:
                    outer = KFold(n_splits=n_outer, shuffle=True, random_state=int(seed))
                    outer_iter = outer.split(samples)
                    split_mode = "fallback_kfold"
                    k_used = 0
                    break
                k -= 1

            if "outer_iter" not in locals() or split_mode not in {"fallback_kfold"}:
                outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=int(seed))
                outer_iter = outer.split(np.zeros(len(samples)), labels)
                split_mode = "cluster_stratified"
                k_used = int(k)

    oof_pred_clr = pd.DataFrame(np.nan, index=samples, columns=Y_tpm.columns)
    oof_pred_tss = pd.DataFrame(np.nan, index=samples, columns=Y_tpm.columns)
    rows: list[dict] = []

    for fold, (tr, te) in enumerate(outer_iter, start=1):
        tr_ids = samples[tr]
        te_ids = samples[te]

        Xtr = X.loc[tr_ids]
        Xte = X.loc[te_ids]
        Ytr0 = Y_tpm.loc[tr_ids]
        Yte0 = Y_tpm.loc[te_ids]

        y_keep = keep_by_prevalence(Ytr0, min_prev_abs=int(min_prev_y_abs), detect_threshold=float(y_detect_threshold))
        Ytr0 = Ytr0.loc[:, y_keep]
        Yte0 = Yte0.loc[:, y_keep]
        if Ytr0.shape[1] < 5:
            continue

        Ytr_tss = tss_rows(Ytr0).fillna(0.0)
        Yte_tss = tss_rows(Yte0).fillna(0.0)
        Ytr_clr = clr_rows(Ytr_tss, pseudocount=float(pseudocount_y))
        Yte_clr = clr_rows(Yte_tss, pseudocount=float(pseudocount_y))

        Ztr_base = transform_x_embedding_svd_clr(Xtr, embed)
        Zte_base = transform_x_embedding_svd_clr(Xte, embed)

        inner = KFold(n_splits=int(inner_splits), shuffle=True, random_state=int(seed + 2000 + fold))
        inner_cache = []
        for itr, iva in inner.split(tr_ids):
            itr_ids = tr_ids[itr]
            iva_ids = tr_ids[iva]
            itr_mask = np.isin(tr_ids, itr_ids)
            iva_mask = np.isin(tr_ids, iva_ids)

            Z_itr0 = Ztr_base[itr_mask]
            Z_iva0 = Ztr_base[iva_mask]

            Y_itr = Ytr_clr.loc[itr_ids, y_keep]
            Y_iva = Ytr_clr.loc[iva_ids, y_keep]

            Y_iva_tss = None
            if need_pw or need_softf1 or need_jsd:
                Y_iva_tss = Ytr_tss.loc[iva_ids, y_keep]

            Dy_iva = aitchison_dm(Y_iva)
            w_feat = feature_weights_from_variance(Y_itr) if need_wclr else None

            inner_cache.append(
                {
                    "itr_ids": itr_ids,
                    "iva_ids": iva_ids,
                    "Z_itr0": Z_itr0,
                    "Z_iva0": Z_iva0,
                    "Y_itr": Y_itr,
                    "Y_iva": Y_iva,
                    "Y_iva_tss": Y_iva_tss,
                    "Dy_iva": Dy_iva,
                    "w_feat": w_feat,
                    "metric_cache": {},
                    "latent_cache": {},
                }
            )
        best = None

        for neigh_k in neigh_k_grid:
            for tau_mult in tau_mult_grid:
                for lam in lam_grid:
                    for yk in y_latent_k_grid:
                        for metric_ridge in metric_ridge_grid:

                            comps, dms = [], []

                            for cached in inner_cache:
                                itr_ids = cached["itr_ids"]
                                iva_ids = cached["iva_ids"]
                                Y_itr = cached["Y_itr"]
                                Y_iva = cached["Y_iva"]
                                Y_iva_tss = cached["Y_iva_tss"]
                                Dy_iva = cached["Dy_iva"]
                                w_feat = cached["w_feat"]

                                mkey = float(metric_ridge)
                                metric_cached = cached["metric_cache"].get(mkey)
                                if metric_cached is None:
                                    Z_itr0 = cached["Z_itr0"]
                                    Z_iva0 = cached["Z_iva0"]
                                    if use_metric_learning:
                                        X_itr_df = pd.DataFrame(Z_itr0, index=itr_ids)
                                        w_in = fit_supervised_diag_metric(
                                            X_clr=X_itr_df,
                                            Y_clr=Y_itr,
                                            max_pairs=int(metric_max_pairs),
                                            random_state=int(seed + 3000 + fold),
                                            ridge=float(metric_ridge),
                                        )
                                        Z_itr = Z_itr0 * np.sqrt(w_in[None, :])
                                        Z_iva = Z_iva0 * np.sqrt(w_in[None, :])
                                    else:
                                        Z_itr, Z_iva = Z_itr0, Z_iva0

                                    scale = median_nn_distance(
                                        Z_itr, k=min(int(tau_scale_k_nn), Z_itr.shape[0] - 1)
                                    )
                                    nn_min_inner = cdist(Z_iva, Z_itr).min(axis=1)
                                    metric_cached = {"Z_itr": Z_itr, "Z_iva": Z_iva, "scale": scale, "nn_min": nn_min_inner}
                                    cached["metric_cache"][mkey] = metric_cached

                                Z_itr = metric_cached["Z_itr"]
                                Z_iva = metric_cached["Z_iva"]
                                scale = metric_cached["scale"]
                                nn_min_inner = metric_cached["nn_min"]

                                tau_abs = float(tau_mult) * float(scale)

                                if ood_tau_inflate:
                                    z_ood = float(np.median(nn_min_inner)) / (float(scale) + 1e-12)
                                    tau_abs_eff = tau_abs * (1.0 + float(ood_tau_gamma) * z_ood)
                                else:
                                    tau_abs_eff = tau_abs

                                if int(yk) > 0:
                                    yk_key = int(yk)
                                    latent_cached = cached["latent_cache"].get(yk_key)
                                    if latent_cached is None:
                                        svd, col_mean = fit_y_latent_svd(
                                            Y_itr, k=int(yk), random_state=int(seed + 4000 + fold)
                                        )
                                        T_itr = encode_y_latent(Y_itr, svd, col_mean)
                                        latent_cached = {"svd": svd, "col_mean": col_mean, "T_itr": T_itr}
                                        cached["latent_cache"][yk_key] = latent_cached
                                    else:
                                        svd = latent_cached["svd"]
                                        col_mean = latent_cached["col_mean"]
                                        T_itr = latent_cached["T_itr"]
                                    T_hat = knn_kernel_predict_tau_abs(
                                        Z_tr=Z_itr,
                                        Z_te=Z_iva,
                                        T_tr=T_itr,
                                        k=int(neigh_k),
                                        tau_abs=float(tau_abs_eff),
                                        lam=float(lam),
                                    )
                                    Yhat_arr = decode_y_latent(T_hat, svd, col_mean)
                                else:
                                    T_itr = Y_itr.to_numpy(dtype=float)
                                    Yhat_arr = knn_kernel_predict_tau_abs(
                                        Z_tr=Z_itr,
                                        Z_te=Z_iva,
                                        T_tr=T_itr,
                                        k=int(neigh_k),
                                        tau_abs=float(tau_abs_eff),
                                        lam=float(lam),
                                    )
                                    Yhat_arr = Yhat_arr - Yhat_arr.mean(axis=1, keepdims=True)

                                if ood_shrink and ood_shrink_inner:
                                    Yhat_arr = apply_ood_shrinkage(
                                        Yhat_clr_arr=Yhat_arr,
                                        Ytr_clr=Y_itr,
                                        nn_min=nn_min_inner,
                                        lam_base=float(ood_lam_base),
                                        lam_cap=float(ood_lam_cap),
                                    )
                                    Yhat_arr = Yhat_arr - Yhat_arr.mean(axis=1, keepdims=True)

                                Yhat_iva_clr = pd.DataFrame(Yhat_arr, index=iva_ids, columns=y_keep)
                                Yhat_iva_tss = clr_to_comp(Yhat_iva_clr)

                                dm_sc = corr_upper_triangle(
                                    Dy_iva,
                                    aitchison_dm(Yhat_iva_clr),
                                    method="spearman",
                                )
                                if not np.isfinite(dm_sc):
                                    continue

                                # Secondary metrics (kept for logging; not primary objective)
                                wclr_mse = (
                                    weighted_clr_mse(Y_iva, Yhat_iva_clr, w_feat) if need_wclr else 0.0
                                )

                                pw_rmse = (
                                    pathway_rmse_tss(
                                        Y_true_tss=Y_iva_tss,
                                        Y_pred_tss=Yhat_iva_tss,
                                        ko_to_group=ko_to_superclass,
                                        log1p=True,
                                    )
                                    if need_pw
                                    else 0.0
                                )

                                if need_softf1:
                                    _, _, sf1 = prf_thresholded(
                                        Y_iva_tss.to_numpy(float),
                                        Yhat_iva_tss.to_numpy(float),
                                        thresh=float(prf_thresh),
                                        weight=str(prf_weight),
                                    )
                                else:
                                    sf1 = 0.0

                                jsd = (
                                    jsd_rows(
                                        Y_iva_tss.to_numpy(float),
                                        Yhat_iva_tss.to_numpy(float),
                                    )
                                    if need_jsd
                                    else 0.0
                                )

                                comp = (
                                    float(w_dm) * float(dm_sc)
                                    - float(w_wclr) * float(wclr_mse)
                                    - float(w_pw_rmse) * float(pw_rmse)
                                    + float(w_softf1) * float(sf1)
                                    - float(w_jsd) * float(jsd)
                                )

                                comps.append(comp)
                                dms.append(dm_sc)

                            if not comps:
                                continue

                            q = 0.25
                            mean_comp = float(np.mean(np.sort(comps)[: max(1, int(q * len(comps)))]))

                            if (best is None) or (mean_comp > best[0]):
                                best = (
                                    mean_comp,
                                    {
                                        "neigh_k": int(neigh_k),
                                        "tau_mult": float(tau_mult),
                                        "lam": float(lam),
                                        "y_latent_k": int(yk),
                                        "metric_ridge": float(metric_ridge),
                                        "inner_dm": float(np.mean(dms)),
                                    },
                                )

        if best is None:
            raise RuntimeError(f"Fold {fold}: no valid hyperparameters found.")
        best_comp, params = best

        if use_metric_learning:
            Xtr_df = pd.DataFrame(Ztr_base, index=tr_ids)
            w = fit_supervised_diag_metric(
                X_clr=Xtr_df,
                Y_clr=Ytr_clr.loc[:, y_keep],
                max_pairs=int(metric_max_pairs),
                random_state=int(seed + 1000 + fold),
                ridge=float(params["metric_ridge"]),
            )
            Ztr = Ztr_base * np.sqrt(w[None, :])
            Zte = Zte_base * np.sqrt(w[None, :])
        else:
            Ztr, Zte = Ztr_base, Zte_base

        d_ood = cdist(Zte, Ztr)
        nn_min = d_ood.min(axis=1)
        ood_med_nn = float(np.median(nn_min))
        ood_max_nn = float(np.max(nn_min))

        scale = median_nn_distance(Ztr, k=min(int(tau_scale_k_nn), Ztr.shape[0] - 1))
        tau_abs = float(params["tau_mult"]) * float(scale)

        if ood_tau_inflate:
            z_ood = float(np.median(nn_min)) / (float(scale) + 1e-12)
            tau_abs_eff = tau_abs * (1.0 + float(ood_tau_gamma) * z_ood)
        else:
            tau_abs_eff = tau_abs

        if int(params["y_latent_k"]) > 0:
            svd, col_mean = fit_y_latent_svd(
                Ytr_clr.loc[:, y_keep],
                k=int(params["y_latent_k"]),
                random_state=int(seed + 5000 + fold),
            )
            Ttr = encode_y_latent(Ytr_clr.loc[:, y_keep], svd, col_mean)
            That = knn_kernel_predict_tau_abs(
                Z_tr=Ztr,
                Z_te=Zte,
                T_tr=Ttr,
                k=int(params["neigh_k"]),
                tau_abs=float(tau_abs_eff),
                lam=float(params["lam"]),
            )
            Yhat_te_arr = decode_y_latent(That, svd, col_mean)
        else:
            Ttr = Ytr_clr.loc[:, y_keep].to_numpy(dtype=float)
            Yhat_te_arr = knn_kernel_predict_tau_abs(
                Z_tr=Ztr,
                Z_te=Zte,
                T_tr=Ttr,
                k=int(params["neigh_k"]),
                tau_abs=float(tau_abs_eff),
                lam=float(params["lam"]),
            )
            Yhat_te_arr = Yhat_te_arr - Yhat_te_arr.mean(axis=1, keepdims=True)

        lam_mean = 0.0
        lam_q90 = 0.0
        if ood_shrink:
            Yhat_te_arr, lam_vec = apply_ood_shrinkage(
                Yhat_clr_arr=Yhat_te_arr,
                Ytr_clr=Ytr_clr.loc[:, y_keep],
                nn_min=nn_min,
                lam_base=float(ood_lam_base),
                lam_cap=float(ood_lam_cap),
                return_lam=True,
            )
            lam_mean = float(np.mean(lam_vec))
            lam_q90 = float(np.quantile(lam_vec, 0.90))

        Yhat_te_arr = Yhat_te_arr - Yhat_te_arr.mean(axis=1, keepdims=True)

        Yhat_te_clr = pd.DataFrame(Yhat_te_arr, index=te_ids, columns=y_keep)
        oof_pred_clr.loc[te_ids, y_keep] = Yhat_te_clr

        Yhat_te_tss = clr_to_comp(Yhat_te_clr)
        oof_pred_tss.loc[te_ids, y_keep] = Yhat_te_tss

        Yte_clr_keep = Yte_clr.loc[:, y_keep]
        dm_sc = corr_upper_triangle(aitchison_dm(Yte_clr_keep), aitchison_dm(Yhat_te_clr), method="spearman")

        w_feat_outer = feature_weights_from_variance(Ytr_clr.loc[:, y_keep])
        wclr_mse_outer = weighted_clr_mse(Yte_clr_keep, Yhat_te_clr, w_feat_outer)

        pw_rmse_outer = pathway_rmse_tss(
            Y_true_tss=Yte_tss.loc[:, y_keep],
            Y_pred_tss=Yhat_te_tss,
            ko_to_group=ko_to_superclass,
            log1p=True,
        )

        sp, sr, sf1 = prf_thresholded(
            Yte_tss.loc[:, y_keep].to_numpy(float),
            Yhat_te_tss.to_numpy(float),
            thresh=float(prf_thresh),
            weight=str(prf_weight),
        )

        jsd = jsd_rows(
            Yte_tss.loc[:, y_keep].to_numpy(float),
            Yhat_te_tss.to_numpy(float),
        )

        Dy = aitchison_dm(Yte_clr_keep)
        Dp = aitchison_dm(Yhat_te_clr)
        std_dm_true = _upper_tri_std(Dy)
        std_dm_pred = _upper_tri_std(Dp)

        rows.append(
            {
                "fold": int(fold),
                "split_mode": split_mode,
                "kmeans_k_used": k_used,
                "best_inner_comp": float(best_comp),
                "best_inner_dm": float(params["inner_dm"]),
                "dm_spearman": float(dm_sc),
                "wclr_mse": float(wclr_mse_outer),
                "pw_rmse_log1p": float(pw_rmse_outer),
                "soft_precision": float(sp),
                "soft_recall": float(sr),
                "soft_f1": float(sf1),
                "jsd": float(jsd),
                "ood_median_nn": float(ood_med_nn),
                "ood_max_nn": float(ood_max_nn),
                "tau_abs": float(tau_abs),
                "tau_abs_eff": float(tau_abs_eff),
                "shrink_lam_mean": float(lam_mean),
                "shrink_lam_q90": float(lam_q90),
                "std_dm_true": float(std_dm_true),
                "std_dm_pred": float(std_dm_pred),
                "y_keep_n": int(len(y_keep)),
                **params,
            }
        )

    return oof_pred_clr, oof_pred_tss, pd.DataFrame(rows)


def fixed_param_oof_knn_on_embedding(
    X: pd.DataFrame,
    Y_tpm: pd.DataFrame,
    embed: dict,
    *,
    ko_to_superclass: dict,
    outer_splits: int = 5,
    seed: int = 0,
    min_prev_y_abs: int = 1,
    y_detect_threshold: float = 1.0,
    pseudocount_y: float = 0.5 / 1e6,
    neigh_k: int = 12,
    tau_mult: float = 2.0,
    lam: float = 0.0,
    y_latent_k: int = 10,
    use_metric_learning: bool = True,
    metric_max_pairs: int = 10000,
    metric_ridge: float = 1.0,
    tau_scale_k_nn: int = 10,
    ood_shrink: bool = False,
    ood_lam_base: float = 0.15,
    ood_lam_cap: float = 0.80,
    ood_tau_inflate: bool = False,
    ood_tau_gamma: float = 1.0,
    informed_splits: bool = False,
    informed_kmeans_on: str = "X",
    informed_kmeans_n_init: int = 20,
    informed_kmeans_k: int | None = None,
    prf_thresh: float = 1e-6,
    prf_weight: str = "binary",
):
    assert X.index.equals(Y_tpm.index), "X and Y must have identical sample index ordering."
    samples = X.index
    n_outer = int(outer_splits)

    if not informed_splits:
        outer = KFold(n_splits=n_outer, shuffle=True, random_state=int(seed))
        outer_iter = outer.split(samples)
        split_mode = "kfold"
        k_used = np.nan
    else:
        if informed_kmeans_on != "X":
            raise ValueError("Use informed_kmeans_on='X' to avoid target leakage.")

        Z_all = transform_x_embedding_svd_clr(X, embed)
        k_default = max(2, n_outer // 2)
        k = int(informed_kmeans_k if informed_kmeans_k is not None else k_default)

        while True:
            km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=int(informed_kmeans_n_init))
            labels = km.fit_predict(Z_all)
            counts = np.bincount(labels)
            if counts.min() >= n_outer:
                break
            if k <= 2:
                outer = KFold(n_splits=n_outer, shuffle=True, random_state=int(seed))
                outer_iter = outer.split(samples)
                split_mode = "fallback_kfold"
                k_used = 0
                break
            k -= 1

        if "outer_iter" not in locals() or split_mode not in {"fallback_kfold"}:
            outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=int(seed))
            outer_iter = outer.split(np.zeros(len(samples)), labels)
            split_mode = "cluster_stratified"
            k_used = int(k)

    oof_pred_clr = pd.DataFrame(np.nan, index=samples, columns=Y_tpm.columns)
    oof_pred_tss = pd.DataFrame(np.nan, index=samples, columns=Y_tpm.columns)
    rows: list[dict] = []

    for fold, (tr, te) in enumerate(outer_iter, start=1):
        tr_ids = samples[tr]
        te_ids = samples[te]

        Xtr = X.loc[tr_ids]
        Xte = X.loc[te_ids]
        Ytr0 = Y_tpm.loc[tr_ids]
        Yte0 = Y_tpm.loc[te_ids]

        y_keep = keep_by_prevalence(Ytr0, min_prev_abs=int(min_prev_y_abs), detect_threshold=float(y_detect_threshold))
        Ytr0 = Ytr0.loc[:, y_keep]
        Yte0 = Yte0.loc[:, y_keep]
        if Ytr0.shape[1] < 5:
            continue

        Ytr_tss = tss_rows(Ytr0).fillna(0.0)
        Yte_tss = tss_rows(Yte0).fillna(0.0)
        Ytr_clr = clr_rows(Ytr_tss, pseudocount=float(pseudocount_y))
        Yte_clr = clr_rows(Yte_tss, pseudocount=float(pseudocount_y))

        Ztr_base = transform_x_embedding_svd_clr(Xtr, embed)
        Zte_base = transform_x_embedding_svd_clr(Xte, embed)

        if use_metric_learning:
            Xtr_df = pd.DataFrame(Ztr_base, index=tr_ids)
            w = fit_supervised_diag_metric(
                X_clr=Xtr_df,
                Y_clr=Ytr_clr.loc[:, y_keep],
                max_pairs=int(metric_max_pairs),
                random_state=int(seed + 1000 + fold),
                ridge=float(metric_ridge),
            )
            Ztr = Ztr_base * np.sqrt(w[None, :])
            Zte = Zte_base * np.sqrt(w[None, :])
        else:
            Ztr, Zte = Ztr_base, Zte_base

        d_ood = cdist(Zte, Ztr)
        nn_min = d_ood.min(axis=1)

        scale = median_nn_distance(Ztr, k=min(int(tau_scale_k_nn), Ztr.shape[0] - 1))
        tau_abs = float(tau_mult) * float(scale)

        if ood_tau_inflate:
            z_ood = float(np.median(nn_min)) / (float(scale) + 1e-12)
            tau_abs_eff = tau_abs * (1.0 + float(ood_tau_gamma) * z_ood)
        else:
            tau_abs_eff = tau_abs

        if int(y_latent_k) > 0:
            svd, col_mean = fit_y_latent_svd(
                Ytr_clr.loc[:, y_keep],
                k=int(y_latent_k),
                random_state=int(seed + 5000 + fold),
            )
            Ttr = encode_y_latent(Ytr_clr.loc[:, y_keep], svd, col_mean)
            That = knn_kernel_predict_tau_abs(
                Z_tr=Ztr,
                Z_te=Zte,
                T_tr=Ttr,
                k=int(neigh_k),
                tau_abs=float(tau_abs_eff),
                lam=float(lam),
            )
            Yhat_te_arr = decode_y_latent(That, svd, col_mean)
        else:
            Ttr = Ytr_clr.loc[:, y_keep].to_numpy(dtype=float)
            Yhat_te_arr = knn_kernel_predict_tau_abs(
                Z_tr=Ztr,
                Z_te=Zte,
                T_tr=Ttr,
                k=int(neigh_k),
                tau_abs=float(tau_abs_eff),
                lam=float(lam),
            )
            Yhat_te_arr = Yhat_te_arr - Yhat_te_arr.mean(axis=1, keepdims=True)

        if ood_shrink:
            Yhat_te_arr, _lam_vec = apply_ood_shrinkage(
                Yhat_clr_arr=Yhat_te_arr,
                Ytr_clr=Ytr_clr.loc[:, y_keep],
                nn_min=nn_min,
                lam_base=float(ood_lam_base),
                lam_cap=float(ood_lam_cap),
                return_lam=True,
            )

        Yhat_te_arr = Yhat_te_arr - Yhat_te_arr.mean(axis=1, keepdims=True)

        Yhat_te_clr = pd.DataFrame(Yhat_te_arr, index=te_ids, columns=y_keep)
        oof_pred_clr.loc[te_ids, y_keep] = Yhat_te_clr

        Yhat_te_tss = clr_to_comp(Yhat_te_clr)
        oof_pred_tss.loc[te_ids, y_keep] = Yhat_te_tss

        Yte_clr_keep = Yte_clr.loc[:, y_keep]
        dm_sc = corr_upper_triangle(aitchison_dm(Yte_clr_keep), aitchison_dm(Yhat_te_clr), method="spearman")

        rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(tr_ids)),
                "n_test": int(len(te_ids)),
                "dm_spearman": float(dm_sc),
                "neigh_k": int(neigh_k),
                "tau_mult": float(tau_mult),
                "lam": float(lam),
                "y_latent_k": int(y_latent_k),
                "metric_ridge": float(metric_ridge),
                "split_mode": split_mode,
                "kmeans_k_used": k_used,
            }
        )

    folds = pd.DataFrame(rows)
    return oof_pred_clr, oof_pred_tss, folds
