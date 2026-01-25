from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .embedding import transform_x_embedding_svd_clr
from .knn_core import encode_y_latent, fit_supervised_diag_metric, fit_y_latent_svd, median_nn_distance
from .transforms import clr_rows, keep_by_prevalence, tss_rows


def fit_final_model(
    X_train: pd.DataFrame,
    Y_train_tpm: pd.DataFrame,
    embed: dict,
    *,
    y_keep: pd.Index | None = None,
    min_prev_y_abs: int = 1,
    y_detect_threshold: float = 0.0,
    pseudocount_y: float = 0.5 / 1e6,
    neigh_k: int = 12,
    tau_mult: float = 2.0,
    lam: float = 0.0,
    y_latent_k: int = 10,
    use_metric_learning: bool = True,
    metric_ridge: float = 2.5,
    metric_max_pairs: int = 5000,
    tau_scale_k_nn: int = 10,
    ood_shrink: bool = True,
    ood_lam_base: float = 0.15,
    ood_lam_cap: float = 0.80,
    seed: int = 0,
) -> dict:
    if y_keep is None:
        y_keep = keep_by_prevalence(Y_train_tpm, min_prev_abs=min_prev_y_abs, detect_threshold=y_detect_threshold)
    Y0 = Y_train_tpm.loc[:, y_keep]

    Y_clr = clr_rows(tss_rows(Y0), pseudocount=float(pseudocount_y))

    Z_base = transform_x_embedding_svd_clr(X_train, embed)

    if use_metric_learning:
        Zdf = pd.DataFrame(Z_base, index=X_train.index)
        w = fit_supervised_diag_metric(
            X_clr=Zdf,
            Y_clr=Y_clr,
            max_pairs=int(metric_max_pairs),
            random_state=int(seed + 101),
            ridge=float(metric_ridge),
        )
        Ztr = Z_base * np.sqrt(w[None, :])
    else:
        w = None
        Ztr = Z_base

    scale = median_nn_distance(Ztr, k=min(int(tau_scale_k_nn), Ztr.shape[0] - 1))
    tau_abs = float(tau_mult) * float(scale)

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


def save_model(model: dict, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return out_path


def load_model(path: str | Path) -> dict:
    return joblib.load(Path(path))


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=str, required=True)
    ap.add_argument("--y", type=str, required=True)
    ap.add_argument("--embed-path", type=str, default=None)
    ap.add_argument("--model-out", type=str, required=True)
    ap.add_argument("--min-prev-y-abs", type=int, default=1)
    ap.add_argument("--y-detect-threshold", type=float, default=0.0)
    ap.add_argument("--pseudocount-y", type=float, default=0.5 / 1e6)
    ap.add_argument("--neigh-k", type=int, default=12)
    ap.add_argument("--tau-mult", type=float, default=2.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--y-latent-k", type=int, default=10)
    ap.add_argument("--use-metric-learning", action="store_true")
    ap.add_argument("--metric-ridge", type=float, default=2.5)
    ap.add_argument("--metric-max-pairs", type=int, default=5000)
    ap.add_argument("--tau-scale-k-nn", type=int, default=10)
    ap.add_argument("--ood-shrink", action="store_true")
    ap.add_argument("--ood-lam-base", type=float, default=0.15)
    ap.add_argument("--ood-lam-cap", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.embed_path is None:
        raise ValueError("--embed-path is required (fit embedding separately and pass it here).")

    X = _read_table(Path(args.x))
    Y = _read_table(Path(args.y))
    embed = joblib.load(Path(args.embed_path))

    model = fit_final_model(
        X_train=X,
        Y_train_tpm=Y,
        embed=embed,
        min_prev_y_abs=int(args.min_prev_y_abs),
        y_detect_threshold=float(args.y_detect_threshold),
        pseudocount_y=float(args.pseudocount_y),
        neigh_k=int(args.neigh_k),
        tau_mult=float(args.tau_mult),
        lam=float(args.lam),
        y_latent_k=int(args.y_latent_k),
        use_metric_learning=bool(args.use_metric_learning),
        metric_ridge=float(args.metric_ridge),
        metric_max_pairs=int(args.metric_max_pairs),
        tau_scale_k_nn=int(args.tau_scale_k_nn),
        ood_shrink=bool(args.ood_shrink),
        ood_lam_base=float(args.ood_lam_base),
        ood_lam_cap=float(args.ood_lam_cap),
        seed=int(args.seed),
    )
    save_model(model, args.model_out)


if __name__ == "__main__":
    main()
