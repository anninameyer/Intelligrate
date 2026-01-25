def fit_final_model(
    X_train: pd.DataFrame,
    Y_train_tpm: pd.DataFrame,
    embed: dict,
    *,
    # choose KO subset (recommended: core/kept); or keep all
    y_keep: pd.Index | None = None,
    min_prev_y_abs: int = 1,
    y_detect_threshold: float = 0.0,
    pseudocount_y: float = 0.5/1e6,

    # fixed hyperparameters
    neigh_k: int = 12,
    tau_mult: float = 2.0,
    lam: float = 0.0,
    y_latent_k: int = 10,

    # metric learning
    use_metric_learning: bool = True,
    metric_ridge: float = 2.5,
    metric_max_pairs: int = 5000,

    # tau scaling
    tau_scale_k_nn: int = 10,

    # ood shrink
    ood_shrink: bool = True,
    ood_lam_base: float = 0.15,
    ood_lam_cap: float = 0.80,

    seed: int = 0,
):
    # KO selection
    if y_keep is None:
        y_keep = keep_by_prevalence(Y_train_tpm, min_prev_abs=min_prev_y_abs, detect_threshold=y_detect_threshold)
    Y0 = Y_train_tpm.loc[:, y_keep]

    # CLR Y
    Y_clr = clr_rows(tss_rows(Y0), pseudocount=float(pseudocount_y))

    # X -> embedding
    Z_base = transform_x_embedding_svd_clr(X_train, embed)

    # supervised diag metric in embedding space
    if use_metric_learning:
        Zdf = pd.DataFrame(Z_base, index=X_train.index)
        w = fit_supervised_diag_metric(
            X_clr=Zdf, Y_clr=Y_clr,
            max_pairs=int(metric_max_pairs),
            random_state=int(seed + 101),
            ridge=float(metric_ridge),
        )
        Ztr = Z_base * np.sqrt(w[None, :])
    else:
        w = None
        Ztr = Z_base

    # tau_abs
    scale = median_nn_distance(Ztr, k=min(int(tau_scale_k_nn), Ztr.shape[0]-1))
    tau_abs = float(tau_mult) * float(scale)

    # Y latent
    if int(y_latent_k) > 0:
        svd_y, col_mean_y = fit_y_latent_svd(Y_clr, k=int(y_latent_k), random_state=int(seed + 202))
        Ttr = encode_y_latent(Y_clr, svd_y, col_mean_y)
    else:
        svd_y, col_mean_y = None, None
        Ttr = Y_clr.to_numpy(float)

    return {
        "embed": embed,
        "y_cols": y_keep,
        "Y_train_clr": Y_clr,
        "Z_train": Ztr,
        "T_train": Ttr,
        "use_metric_learning": bool(use_metric_learning),
        "w": w,
        "neigh_k": int(neigh_k),
        "tau_abs": float(tau_abs),
        "lam": float(lam),
        "svd_y": svd_y,
        "col_mean_y": col_mean_y,
        "ood_shrink": bool(ood_shrink),
        "ood_lam_base": float(ood_lam_base),
        "ood_lam_cap": float(ood_lam_cap),
    }
