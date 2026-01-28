from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from .metrics import (
    aitchison_dm,
    bray_spearman_union,
    corr_upper_triangle,
    dm_spearman_union,
    evaluate_union_metrics,
    procrustes_union_aitchison,
    procrustes_union_bray,
)
from .transforms import clr_rows, clr_to_comp, keep_by_prevalence, tss_rows


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def _read_ko_to_superclass(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 2:
        raise ValueError("ko_to_superclass.tsv must have at least two columns.")
    ko_col = df.columns[0]
    sc_col = df.columns[1]
    return dict(zip(df[ko_col].astype(str), df[sc_col].astype(str)))


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


def _iter_grid_configs(cfg: dict) -> list[tuple[dict, dict]]:
    grid = cfg.get("grid", {}) or {}
    items: list[tuple[str, str, list]] = []
    for section, params in grid.items():
        if section not in cfg or params is None:
            continue
        if not isinstance(params, dict):
            raise ValueError(f"grid.{section} must be a mapping of key -> list")
        for key, values in params.items():
            if isinstance(values, (list, tuple)):
                vals = list(values)
            else:
                vals = [values]
            items.append((section, key, vals))

    if not items:
        return [(cfg, {})]

    combos = []
    for combo in itertools.product(*[vals for _, _, vals in items]):
        cfg_run = copy.deepcopy(cfg)
        cfg_run.pop("grid", None)
        overrides = {}
        for (section, key, _), val in zip(items, combo):
            cfg_run[section][key] = val
            overrides[f"{section}.{key}"] = val
        combos.append((cfg_run, overrides))
    return combos


def _run_once(cfg: dict, *, data_dir: Path, out_dir: Path) -> dict:
    X_full = _read_table(data_dir / cfg["data"]["x_full"])
    X = _read_table(data_dir / cfg["data"]["x"])
    Y = _read_table(data_dir / cfg["data"]["y"])
    ko_to_super = _read_ko_to_superclass(data_dir / cfg["data"]["ko_to_superclass"])
    picrust2_path = cfg["data"].get("picrust2")
    picrust2 = _read_table(data_dir / picrust2_path) if picrust2_path else None

    common = X.index.intersection(Y.index)
    X = X.loc[common].sort_index()
    Y = Y.loc[common].sort_index()
    if picrust2 is not None:
        picrust2 = picrust2.loc[picrust2.index.intersection(common)].sort_index()

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

    dm_mean = float(folds["dm_spearman"].mean())
    dm_std = float(folds["dm_spearman"].std())

    y_keep = keep_by_prevalence(
        Y, min_prev_abs=int(cfg["score"]["min_prev_y_abs"]), detect_threshold=float(cfg["score"]["y_detect_threshold"])
    )
    Y_clr = clr_rows(tss_rows(Y.loc[:, y_keep]), pseudocount=float(cfg["score"]["pseudocount_y"]))
    P_clr = oof_clr.loc[Y_clr.index, Y_clr.columns]
    good = P_clr.notna().all(axis=1)
    Y_clr = Y_clr.loc[good]
    P_clr = P_clr.loc[good]
    oof_dm = corr_upper_triangle(aitchison_dm(Y_clr), aitchison_dm(P_clr), method="spearman")

    union_pseudo = float(cfg["score"]["pseudocount_y"])
    union_detect = float(cfg["model"]["y_detect_threshold"])

    model_dm_union_raw = dm_spearman_union(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    model_dm_union_strict = dm_spearman_union(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    model_bray_union_raw = bray_spearman_union(Y, oof_tss, detect_threshold=0.0, fillna_zero=True)
    model_bray_union_strict = bray_spearman_union(
        Y, oof_tss, detect_threshold=union_detect, fillna_zero=True
    )
    model_proc_ait_raw = procrustes_union_aitchison(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    model_proc_ait_strict = procrustes_union_aitchison(
        Y, oof_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    model_proc_bray_raw = procrustes_union_bray(Y, oof_tss, detect_threshold=0.0, fillna_zero=True)
    model_proc_bray_strict = procrustes_union_bray(Y, oof_tss, detect_threshold=union_detect, fillna_zero=True)

    metrics_cfg = cfg.get("metrics", {})
    compute_wclr = bool(metrics_cfg.get("compute_wclr", False))
    compute_jsd = bool(metrics_cfg.get("compute_jsd", False))
    compute_pathway = bool(metrics_cfg.get("compute_pathway_rmse", False))
    compute_per_pathway = bool(metrics_cfg.get("pathway_rmse_per_group", False))
    log1p_pathway = bool(metrics_cfg.get("pathway_rmse_log1p", True))

    model_prf = evaluate_union_metrics(
        Y,
        oof_tss,
        pseudocount=union_pseudo,
        detect_threshold=union_detect,
        prf_thresh=float(cfg["prf"]["prf_thresh"]),
        prf_weight=str(cfg["prf"]["prf_weight"]),
        fillna_zero=True,
        compute_wclr=compute_wclr,
        compute_jsd=compute_jsd,
        compute_pathway=compute_pathway,
        compute_per_pathway=compute_per_pathway,
        ko_to_group=ko_to_super,
        log1p_pathway=log1p_pathway,
    )

    picrust2_dm_union_raw = None
    picrust2_dm_union_strict = None
    picrust2_bray_union_raw = None
    picrust2_bray_union_strict = None
    picrust2_proc_ait_raw = None
    picrust2_proc_ait_strict = None
    picrust2_proc_bray_raw = None
    picrust2_proc_bray_strict = None
    delta_union = None
    picrust2_prf = None
    if picrust2 is not None:
        picrust2_dm_union_raw = dm_spearman_union(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
        )
        picrust2_dm_union_strict = dm_spearman_union(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
        )
        picrust2_bray_union_raw = bray_spearman_union(Y, picrust2, detect_threshold=0.0, fillna_zero=True)
        picrust2_bray_union_strict = bray_spearman_union(
            Y, picrust2, detect_threshold=union_detect, fillna_zero=True
        )
        picrust2_proc_ait_raw = procrustes_union_aitchison(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
        )
        picrust2_proc_ait_strict = procrustes_union_aitchison(
            Y, picrust2, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
        )
        picrust2_proc_bray_raw = procrustes_union_bray(Y, picrust2, detect_threshold=0.0, fillna_zero=True)
        picrust2_proc_bray_strict = procrustes_union_bray(
            Y, picrust2, detect_threshold=union_detect, fillna_zero=True
        )
        delta_union = float(model_dm_union_strict - picrust2_dm_union_strict)
        picrust2_prf = evaluate_union_metrics(
            Y,
            picrust2,
            pseudocount=union_pseudo,
            detect_threshold=union_detect,
            prf_thresh=float(cfg["prf"]["prf_thresh"]),
            prf_weight=str(cfg["prf"]["prf_weight"]),
            fillna_zero=True,
            compute_wclr=compute_wclr,
            compute_jsd=compute_jsd,
            compute_pathway=compute_pathway,
            compute_per_pathway=compute_per_pathway,
            ko_to_group=ko_to_super,
            log1p_pathway=log1p_pathway,
        )

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
    full_fit_dm_union_raw = dm_spearman_union(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    full_fit_dm_union_strict = dm_spearman_union(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    full_fit_bray_union_raw = bray_spearman_union(Y, full_fit_tss, detect_threshold=0.0, fillna_zero=True)
    full_fit_bray_union_strict = bray_spearman_union(
        Y, full_fit_tss, detect_threshold=union_detect, fillna_zero=True
    )
    full_fit_proc_ait_raw = procrustes_union_aitchison(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=0.0, fillna_zero=True
    )
    full_fit_proc_ait_strict = procrustes_union_aitchison(
        Y, full_fit_tss, pseudocount=union_pseudo, detect_threshold=union_detect, fillna_zero=True
    )
    full_fit_proc_bray_raw = procrustes_union_bray(Y, full_fit_tss, detect_threshold=0.0, fillna_zero=True)
    full_fit_proc_bray_strict = procrustes_union_bray(
        Y, full_fit_tss, detect_threshold=union_detect, fillna_zero=True
    )

    full_fit_prf = evaluate_union_metrics(
        Y,
        full_fit_tss,
        pseudocount=union_pseudo,
        detect_threshold=union_detect,
        prf_thresh=float(cfg["prf"]["prf_thresh"]),
        prf_weight=str(cfg["prf"]["prf_weight"]),
        fillna_zero=True,
        compute_wclr=compute_wclr,
        compute_jsd=compute_jsd,
        compute_pathway=compute_pathway,
        compute_per_pathway=compute_per_pathway,
        ko_to_group=ko_to_super,
        log1p_pathway=log1p_pathway,
    )

    run = {
        "objective_dm_spearman_mean": dm_mean,
        "objective_dm_spearman_std": dm_std,
        "oof_dm_spearman": float(oof_dm),
        "model_dm_union_raw": float(model_dm_union_raw),
        "model_dm_union_strict": float(model_dm_union_strict),
        "model_bray_union_raw": float(model_bray_union_raw),
        "model_bray_union_strict": float(model_bray_union_strict),
        "model_procrustes_aitchison_raw": float(model_proc_ait_raw),
        "model_procrustes_aitchison_strict": float(model_proc_ait_strict),
        "model_procrustes_bray_raw": float(model_proc_bray_raw),
        "model_procrustes_bray_strict": float(model_proc_bray_strict),
        "model_soft_precision": float(model_prf.get("soft_precision")),
        "model_soft_recall": float(model_prf.get("soft_recall")),
        "model_soft_f1": float(model_prf.get("soft_f1")),
        "model_wclr_mse": float(model_prf.get("wclr_mse")) if compute_wclr else None,
        "model_jsd": float(model_prf.get("jsd")) if compute_jsd else None,
        "model_pathway_rmse": float(model_prf.get("pathway_rmse")) if compute_pathway else None,
        "picrust2_dm_union_raw": float(picrust2_dm_union_raw) if picrust2_dm_union_strict is not None else None,
        "picrust2_dm_union_strict": float(picrust2_dm_union_strict) if picrust2_dm_union_strict is not None else None,
        "picrust2_bray_union_raw": float(picrust2_bray_union_raw) if picrust2_bray_union_strict is not None else None,
        "picrust2_bray_union_strict": float(picrust2_bray_union_strict) if picrust2_bray_union_strict is not None else None,
        "picrust2_procrustes_aitchison_raw": float(picrust2_proc_ait_raw) if picrust2_proc_ait_strict is not None else None,
        "picrust2_procrustes_aitchison_strict": float(picrust2_proc_ait_strict) if picrust2_proc_ait_strict is not None else None,
        "picrust2_procrustes_bray_raw": float(picrust2_proc_bray_raw) if picrust2_proc_bray_strict is not None else None,
        "picrust2_procrustes_bray_strict": float(picrust2_proc_bray_strict) if picrust2_proc_bray_strict is not None else None,
        "picrust2_soft_precision": float(picrust2_prf.get("soft_precision")) if picrust2_prf else None,
        "picrust2_soft_recall": float(picrust2_prf.get("soft_recall")) if picrust2_prf else None,
        "picrust2_soft_f1": float(picrust2_prf.get("soft_f1")) if picrust2_prf else None,
        "picrust2_wclr_mse": float(picrust2_prf.get("wclr_mse")) if picrust2_prf and compute_wclr else None,
        "picrust2_jsd": float(picrust2_prf.get("jsd")) if picrust2_prf and compute_jsd else None,
        "picrust2_pathway_rmse": float(picrust2_prf.get("pathway_rmse")) if picrust2_prf and compute_pathway else None,
        "delta_union": float(delta_union) if delta_union is not None else None,
        "full_fit_dm_union_raw": float(full_fit_dm_union_raw),
        "full_fit_dm_union_strict": float(full_fit_dm_union_strict),
        "full_fit_bray_union_raw": float(full_fit_bray_union_raw),
        "full_fit_bray_union_strict": float(full_fit_bray_union_strict),
        "full_fit_procrustes_aitchison_raw": float(full_fit_proc_ait_raw),
        "full_fit_procrustes_aitchison_strict": float(full_fit_proc_ait_strict),
        "full_fit_procrustes_bray_raw": float(full_fit_proc_bray_raw),
        "full_fit_procrustes_bray_strict": float(full_fit_proc_bray_strict),
        "full_fit_soft_precision": float(full_fit_prf.get("soft_precision")),
        "full_fit_soft_recall": float(full_fit_prf.get("soft_recall")),
        "full_fit_soft_f1": float(full_fit_prf.get("soft_f1")),
        "full_fit_wclr_mse": float(full_fit_prf.get("wclr_mse")) if compute_wclr else None,
        "full_fit_jsd": float(full_fit_prf.get("jsd")) if compute_jsd else None,
        "full_fit_pathway_rmse": float(full_fit_prf.get("pathway_rmse")) if compute_pathway else None,
        "full_fit_params": full_params,
        "n_samples": int(len(common)),
        "runtime_sec": float(dt),
        "config": cfg,
    }

    return {
        "oof_clr": oof_clr,
        "oof_tss": oof_tss,
        "folds": folds,
        "run": run,
        "model_prf": model_prf,
        "full_fit_prf": full_fit_prf,
        "picrust2_prf": picrust2_prf,
    }


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    repo_root = Path(".").resolve()
    data_dir = repo_root / "data"
    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_runs = _iter_grid_configs(cfg)
    grid_results = []
    best_idx = None
    best_obj = -np.inf
    best_payload = None

    for idx, (cfg_run, overrides) in enumerate(grid_runs, start=1):
        payload = _run_once(cfg_run, data_dir=data_dir, out_dir=out_dir)
        run = payload["run"]
        obj = float(run["objective_dm_spearman_mean"])
        grid_results.append({"grid_index": idx, **overrides, "objective_dm_spearman_mean": obj})
        if obj > best_obj:
            best_obj = obj
            best_idx = idx
            best_payload = payload

    if best_payload is None:
        raise RuntimeError("No valid training runs produced outputs.")

    oof_clr = best_payload["oof_clr"]
    oof_tss = best_payload["oof_tss"]
    folds = best_payload["folds"]
    run = best_payload["run"]
    model_prf = best_payload["model_prf"]
    full_fit_prf = best_payload["full_fit_prf"]
    picrust2_prf = best_payload["picrust2_prf"]

    if len(grid_results) > 1:
        run["grid_best_index"] = best_idx
        run["grid_results"] = grid_results
        pd.DataFrame(grid_results).to_csv(out_dir / "grid_results.tsv", sep="\t", index=False)

    stamp = time.strftime("%Y%m%d_%H%M%S")

    oof_clr.to_csv(out_dir / "oof_clr.tsv", sep="\t")
    oof_tss.to_csv(out_dir / "oof_tss.tsv", sep="\t")
    folds.to_csv(out_dir / "folds.tsv", sep="\t", index=False)
    (out_dir / "summary.json").write_text(json.dumps(run, indent=2))

    summary_flat = {k: v for k, v in run.items() if k != "config"}
    pd.DataFrame([summary_flat]).to_csv(out_dir / "summary.tsv", sep="\t", index=False)

    oof_clr.to_csv(out_dir / f"oof_clr_{stamp}.tsv", sep="\t")
    oof_tss.to_csv(out_dir / f"oof_tss_{stamp}.tsv", sep="\t")
    folds.to_csv(out_dir / f"folds_{stamp}.tsv", sep="\t", index=False)
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(run, indent=2))

    if isinstance(model_prf.get("pathway_rmse_per_group"), pd.Series):
        model_prf["pathway_rmse_per_group"].to_csv(out_dir / "pathway_rmse_oof.tsv", sep="\t", header=True)
    if isinstance(full_fit_prf.get("pathway_rmse_per_group"), pd.Series):
        full_fit_prf["pathway_rmse_per_group"].to_csv(out_dir / "pathway_rmse_full_fit.tsv", sep="\t", header=True)
    if picrust2_prf and isinstance(picrust2_prf.get("pathway_rmse_per_group"), pd.Series):
        picrust2_prf["pathway_rmse_per_group"].to_csv(out_dir / "pathway_rmse_picrust2.tsv", sep="\t", header=True)

    print(f"OBJECTIVE_DM_SPEARMAN_MEAN={run['objective_dm_spearman_mean']:.6f}")
    print(f"OOF_DM_SPEARMAN={float(run['oof_dm_spearman']):.6f}")
    print(f"MODEL_DM_UNION_RAW={float(run['model_dm_union_raw']):.6f}")
    print(f"MODEL_DM_UNION_STRICT={float(run['model_dm_union_strict']):.6f}")
    print(f"FULL_FIT_DM_UNION_STRICT={float(run['full_fit_dm_union_strict']):.6f}")
    print(f"MODEL_BRAY_UNION_RAW={float(run['model_bray_union_raw']):.6f}")
    print(f"MODEL_BRAY_UNION_STRICT={float(run['model_bray_union_strict']):.6f}")
    print(f"MODEL_PROC_AITCHISON_RAW={float(run['model_procrustes_aitchison_raw']):.6f}")
    print(f"MODEL_PROC_AITCHISON_STRICT={float(run['model_procrustes_aitchison_strict']):.6f}")
    print(f"MODEL_PROC_BRAY_RAW={float(run['model_procrustes_bray_raw']):.6f}")
    print(f"MODEL_PROC_BRAY_STRICT={float(run['model_procrustes_bray_strict']):.6f}")
    print(f"FULL_FIT_DM_UNION_RAW={float(run['full_fit_dm_union_raw']):.6f}")
    print(f"FULL_FIT_BRAY_UNION_STRICT={float(run['full_fit_bray_union_strict']):.6f}")
    print(f"FULL_FIT_BRAY_UNION_RAW={float(run['full_fit_bray_union_raw']):.6f}")
    print(f"FULL_FIT_PROC_AITCHISON_STRICT={float(run['full_fit_procrustes_aitchison_strict']):.6f}")
    print(f"FULL_FIT_PROC_AITCHISON_RAW={float(run['full_fit_procrustes_aitchison_raw']):.6f}")
    print(f"FULL_FIT_PROC_BRAY_STRICT={float(run['full_fit_procrustes_bray_strict']):.6f}")
    print(f"FULL_FIT_PROC_BRAY_RAW={float(run['full_fit_procrustes_bray_raw']):.6f}")
    if run.get("picrust2_dm_union_strict") is not None:
        print(f"PICRUST2_DM_UNION_RAW={float(run['picrust2_dm_union_raw']):.6f}")
        print(f"PICRUST2_DM_UNION_STRICT={float(run['picrust2_dm_union_strict']):.6f}")
        print(f"PICRUST2_BRAY_UNION_RAW={float(run['picrust2_bray_union_raw']):.6f}")
        print(f"PICRUST2_BRAY_UNION_STRICT={float(run['picrust2_bray_union_strict']):.6f}")
        print(f"PICRUST2_PROC_AITCHISON_RAW={float(run['picrust2_procrustes_aitchison_raw']):.6f}")
        print(f"PICRUST2_PROC_AITCHISON_STRICT={float(run['picrust2_procrustes_aitchison_strict']):.6f}")
        print(f"PICRUST2_PROC_BRAY_RAW={float(run['picrust2_procrustes_bray_raw']):.6f}")
        print(f"PICRUST2_PROC_BRAY_STRICT={float(run['picrust2_procrustes_bray_strict']):.6f}")
        print(f"DELTA_UNION={float(run['delta_union']):.6f}")
    print(f"RESULTS_PREFIX={stamp}")


if __name__ == "__main__":
    main()
