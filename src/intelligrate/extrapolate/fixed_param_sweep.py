from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .cv_knn import fixed_param_oof_knn_on_embedding
from .embedding import fit_x_embedding_svd_clr
from .full_predict import evaluate_paired_subset


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def run_fixed_param_sweep_explicit(
    *,
    X_full: pd.DataFrame,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    ko_to_superclass: str | dict | None,
    out_path: Path,
    cv_cfg: dict,
    embed_cfg: dict,
    model_cfg: dict,
    prf_cfg: dict,
    metrics_cfg: dict,
    sweep_cfg: dict,
    embed: dict | None = None,
) -> pd.DataFrame:
    """
    Sweep fixed-parameter combos using explicit inputs (DataFrames + config dicts).
    Note: this does NOT use cfg["grid"]; any parameter not listed in fixed_param_sweep
    falls back to a single default value from model/cv config.
    """
    if embed is None:
        embed = fit_x_embedding_svd_clr(
            X_full,
            min_prev_x_abs=int(embed_cfg.get("min_prev_x_abs", 2)),
            pseudocount_x=float(embed_cfg.get("pseudocount_x", 0.5)),
            n_components=int(embed_cfg.get("n_components", 128)),
            seed=int(cv_cfg.get("seed", 0)),
        )

    grid_key_map = {
        "neigh_k": "neigh_k_grid",
        "tau_mult": "tau_mult_grid",
        "y_latent_k": "y_latent_k_grid",
        "metric_ridge": "metric_ridge_grid",
        "lam": "lam_grid",
    }

    def _as_list(key: str, default):
        if key in sweep_cfg:
            val = sweep_cfg.get(key)
            return val if isinstance(val, list) else [val]
        grid_key = grid_key_map.get(key)
        if grid_key and grid_key in model_cfg:
            val = model_cfg.get(grid_key)
            return val if isinstance(val, list) else [val]
        val = model_cfg.get(key, default)
        return val if isinstance(val, list) else [val]

    sweep_neigh_k = _as_list("neigh_k", model_cfg.get("neigh_k", 24))
    sweep_tau_mult = _as_list("tau_mult", model_cfg.get("tau_mult", 1.0))
    sweep_y_latent_k = _as_list("y_latent_k", model_cfg.get("y_latent_k", 10))
    sweep_metric_ridge = _as_list("metric_ridge", model_cfg.get("metric_ridge", 2.5))
    sweep_lam = _as_list("lam", model_cfg.get("lam", 0.0))
    sweep_min_prev_y_abs = _as_list("min_prev_y_abs", model_cfg.get("min_prev_y_abs", 1))
    sweep_y_detect_threshold = _as_list("y_detect_threshold", model_cfg.get("y_detect_threshold", 1.0))
    sweep_pseudocount_y = _as_list("pseudocount_y", model_cfg.get("pseudocount_y", 0.5 / 1e6))
    sweep_metric_max_pairs = _as_list("metric_max_pairs", model_cfg.get("metric_max_pairs", 5000))
    sweep_tau_scale_k_nn = _as_list("tau_scale_k_nn", model_cfg.get("tau_scale_k_nn", 10))
    sweep_ood_shrink = _as_list("ood_shrink", model_cfg.get("ood_shrink", False))
    sweep_ood_lam_base = _as_list("ood_lam_base", model_cfg.get("ood_lam_base", 0.1))
    sweep_ood_lam_cap = _as_list("ood_lam_cap", model_cfg.get("ood_lam_cap", 0.8))
    sweep_ood_tau_inflate = _as_list("ood_tau_inflate", model_cfg.get("ood_tau_inflate", False))
    sweep_ood_tau_gamma = _as_list("ood_tau_gamma", model_cfg.get("ood_tau_gamma", 1.0))
    sweep_use_metric_learning = _as_list("use_metric_learning", model_cfg.get("use_metric_learning", True))
    sweep_outer_splits = _as_list("outer_splits", cv_cfg.get("outer_splits", 5))
    sweep_seed = _as_list("seed", cv_cfg.get("seed", 0))
    sweep_informed_splits = _as_list("informed_splits", cv_cfg.get("informed_splits", False))

    rows = []
    for neigh_k in sweep_neigh_k:
        for tau_mult in sweep_tau_mult:
            for y_latent_k in sweep_y_latent_k:
                for metric_ridge in sweep_metric_ridge:
                    for lam in sweep_lam:
                        for min_prev_y_abs in sweep_min_prev_y_abs:
                            for y_detect_threshold in sweep_y_detect_threshold:
                                for pseudocount_y in sweep_pseudocount_y:
                                    for metric_max_pairs in sweep_metric_max_pairs:
                                        for tau_scale_k_nn in sweep_tau_scale_k_nn:
                                            for ood_shrink in sweep_ood_shrink:
                                                for ood_lam_base in sweep_ood_lam_base:
                                                    for ood_lam_cap in sweep_ood_lam_cap:
                                                        for ood_tau_inflate in sweep_ood_tau_inflate:
                                                            for ood_tau_gamma in sweep_ood_tau_gamma:
                                                                for use_metric_learning in sweep_use_metric_learning:
                                                                    for outer_splits in sweep_outer_splits:
                                                                        for seed in sweep_seed:
                                                                            for informed_splits in sweep_informed_splits:
                                                                                t0 = time.time()
                                                                                oof_clr, oof_tss, _ = fixed_param_oof_knn_on_embedding(
                                                                                    X=X,
                                                                                    Y_tpm=Y,
                                                                                    embed=embed,
                                                                                    ko_to_superclass=ko_to_superclass,
                                                                                    outer_splits=int(outer_splits),
                                                                                    seed=int(seed),
                                                                                    min_prev_y_abs=int(min_prev_y_abs),
                                                                                    y_detect_threshold=float(y_detect_threshold),
                                                                                    pseudocount_y=float(pseudocount_y),
                                                                                    neigh_k=int(neigh_k),
                                                                                    tau_mult=float(tau_mult),
                                                                                    lam=float(lam),
                                                                                    y_latent_k=int(y_latent_k),
                                                                                    use_metric_learning=bool(use_metric_learning),
                                                                                    metric_max_pairs=int(metric_max_pairs),
                                                                                    metric_ridge=float(metric_ridge),
                                                                                    tau_scale_k_nn=int(tau_scale_k_nn),
                                                                                    ood_shrink=bool(ood_shrink),
                                                                                    ood_lam_base=float(ood_lam_base),
                                                                                    ood_lam_cap=float(ood_lam_cap),
                                                                                    ood_tau_inflate=bool(ood_tau_inflate),
                                                                                    ood_tau_gamma=float(ood_tau_gamma),
                                                                                    informed_splits=bool(informed_splits),
                                                                                    informed_kmeans_on="X",
                                                                                    prf_thresh=float(prf_cfg.get("prf_thresh", 1e-6)),
                                                                                    prf_weight=str(prf_cfg.get("prf_weight", "binary")),
                                                                                )
                                                                                metrics = evaluate_paired_subset(
                                                                                    truth_tpm=Y,
                                                                                    pred_tss=oof_tss,
                                                                                    pseudocount=float(pseudocount_y),
                                                                                    detect_threshold=float(y_detect_threshold),
                                                                                    prf_thresh=float(prf_cfg.get("prf_thresh", 1e-6)),
                                                                                    prf_weight=str(prf_cfg.get("prf_weight", "binary")),
                                                                                    compute_wclr=bool(metrics_cfg.get("compute_wclr", False)),
                                                                                    compute_jsd=bool(metrics_cfg.get("compute_jsd", False)),
                                                                                    compute_pathway=bool(metrics_cfg.get("compute_pathway_rmse", False)),
                                                                                    compute_per_pathway=bool(metrics_cfg.get("pathway_rmse_per_group", False)),
                                                                                    ko_to_group=ko_to_superclass,
                                                                                    log1p_pathway=bool(metrics_cfg.get("pathway_rmse_log1p", True)),
                                                                                )
                                                                                rows.append(
                                                                                    {
                                                                                        "neigh_k": int(neigh_k),
                                                                                        "tau_mult": float(tau_mult),
                                                                                        "y_latent_k": int(y_latent_k),
                                                                                        "metric_ridge": float(metric_ridge),
                                                                                        "lam": float(lam),
                                                                                        "min_prev_y_abs": int(min_prev_y_abs),
                                                                                        "y_detect_threshold": float(y_detect_threshold),
                                                                                        "pseudocount_y": float(pseudocount_y),
                                                                                        "metric_max_pairs": int(metric_max_pairs),
                                                                                        "tau_scale_k_nn": int(tau_scale_k_nn),
                                                                                        "ood_shrink": bool(ood_shrink),
                                                                                        "ood_lam_base": float(ood_lam_base),
                                                                                        "ood_lam_cap": float(ood_lam_cap),
                                                                                        "ood_tau_inflate": bool(ood_tau_inflate),
                                                                                        "ood_tau_gamma": float(ood_tau_gamma),
                                                                                        "use_metric_learning": bool(use_metric_learning),
                                                                                        "outer_splits": int(outer_splits),
                                                                                        "seed": int(seed),
                                                                                        "informed_splits": bool(informed_splits),
                                                                                        "dm_union": metrics.get("dm_union"),
                                                                                        "dm_union_strict": metrics.get("dm_union_strict"),
                                                                                        "dm_union_raw": metrics.get("dm_union_raw"),
                                                                                        "runtime_s": time.time() - t0,
                                                                                    }
                                                                                )

    df = pd.DataFrame(rows).sort_values("dm_union", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return df


def run_fixed_param_sweep(cfg: dict, *, out_path: Path) -> pd.DataFrame:
    """
    Sweep fixed-parameter combos from cfg["fixed_param_sweep"].
    Note: this does NOT use cfg["grid"]; any parameter not listed in fixed_param_sweep
    falls back to a single default value from cfg["model"] / cfg["cv"].
    """
    data_dir = Path(cfg.get("data_dir", "data"))
    X_full = _read_table(data_dir / cfg["data"]["x_full"])
    X = _read_table(data_dir / cfg["data"]["x"])
    Y = _read_table(data_dir / cfg["data"]["y"])
    ko_to_superclass = cfg["data"].get("ko_to_superclass")

    cv_cfg = cfg.get("cv", {})
    embed_cfg = cfg.get("embed", {})
    model_cfg = cfg.get("model", {})
    prf_cfg = cfg.get("prf", {})
    metrics_cfg = cfg.get("metrics", {})
    sweep_cfg = cfg.get("fixed_param_sweep", {})

    return run_fixed_param_sweep_explicit(
        X_full=X_full,
        X=X,
        Y=Y,
        ko_to_superclass=ko_to_superclass,
        out_path=out_path,
        cv_cfg=cv_cfg,
        embed_cfg=embed_cfg,
        model_cfg=model_cfg,
        prf_cfg=prf_cfg,
        metrics_cfg=metrics_cfg,
        sweep_cfg=sweep_cfg,
        embed=None,
    )


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out", type=str, default="results/fixed_param_sweep.tsv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = run_fixed_param_sweep(cfg, out_path=Path(args.out))
    best = df.iloc[0].to_dict() if not df.empty else {}
    if best:
        print(
            "BEST_FIXED_PARAM",
            f"neigh_k={best['neigh_k']}",
            f"tau_mult={best['tau_mult']}",
            f"y_latent_k={best['y_latent_k']}",
            f"metric_ridge={best['metric_ridge']}",
            f"dm_union={best['dm_union']}",
        )


if __name__ == "__main__":
    main()
