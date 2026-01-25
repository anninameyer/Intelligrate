def predict_final_model(X_new: pd.DataFrame, model: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    Z_new_base = transform_x_embedding_svd_clr(X_new, model["embed"])

    if model["use_metric_learning"]:
        Z_new = Z_new_base * np.sqrt(model["w"][None, :])
    else:
        Z_new = Z_new_base

    Ztr = model["Z_train"]

    # OOD stats
    D = cdist(Z_new, Ztr, metric="euclidean")
    nn_min = D.min(axis=1)

    # predict in latent or full CLR space
    That = knn_kernel_predict_tau_abs(
        Z_tr=Ztr, Z_te=Z_new, T_tr=model["T_train"],
        k=int(model["neigh_k"]), tau_abs=float(model["tau_abs"]), lam=float(model["lam"])
    )

    if model["svd_y"] is not None:
        Yhat = decode_y_latent(That, model["svd_y"], model["col_mean_y"])
    else:
        Yhat = That
        Yhat = Yhat - Yhat.mean(axis=1, keepdims=True)

    if model["ood_shrink"]:
        Yhat = apply_ood_shrinkage(
            Yhat_clr_arr=Yhat,
            Ytr_clr=model["Y_train_clr"],
            nn_min=nn_min,
            lam_base=float(model["ood_lam_base"]),
            lam_cap=float(model["ood_lam_cap"]),
        )
        Yhat = Yhat - Yhat.mean(axis=1, keepdims=True)

    Yhat_df = pd.DataFrame(Yhat, index=X_new.index, columns=model["y_cols"])
    diag = pd.DataFrame({
        "ood_nn_min": nn_min,
        "ood_median_nn": float(np.median(nn_min)),
        "ood_max_nn": float(np.max(nn_min)),
    }, index=X_new.index)

    return Yhat_df, diag

