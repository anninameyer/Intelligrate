from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .full_fit import load_model
from .knn_core import apply_ood_shrinkage, decode_y_latent, knn_kernel_predict_tau_abs
from .metrics import (
    bray_spearman_union,
    dm_spearman_union,
    evaluate_union_metrics,
    procrustes_union_aitchison,
    procrustes_union_bray,
)
from .transforms import clr_to_comp
from .embedding import transform_x_embedding_svd_clr


def predict_final_model(
    X_new: pd.DataFrame,
    model: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    Z_new_base = transform_x_embedding_svd_clr(X_new, model["embed"])

    if model["use_metric_learning"]:
        Z_new = Z_new_base * np.sqrt(model["w"][None, :])
    else:
        Z_new = Z_new_base

    Ztr = model["Z_train"]

    D = cdist(Z_new, Ztr, metric="euclidean")
    nn_min = D.min(axis=1)

    That = knn_kernel_predict_tau_abs(
        Z_tr=Ztr,
        Z_te=Z_new,
        T_tr=model["T_train"],
        k=int(model["neigh_k"]),
        tau_abs=float(model["tau_abs"]),
        lam=float(model["lam"]),
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

    Yhat_clr = pd.DataFrame(Yhat, index=X_new.index, columns=model["y_cols"])
    Yhat_tss = clr_to_comp(Yhat_clr)

    diag = pd.DataFrame(
        {
            "ood_nn_min": nn_min,
            "ood_median_nn": float(np.median(nn_min)),
            "ood_max_nn": float(np.max(nn_min)),
        },
        index=X_new.index,
    )

    return Yhat_clr, Yhat_tss, diag


def evaluate_paired_subset(
    truth_tpm: pd.DataFrame,
    pred_tss: pd.DataFrame,
    *,
    pseudocount: float,
    detect_threshold: float,
    prf_thresh: float = 1e-6,
    prf_weight: str = "binary",
    compute_wclr: bool = False,
    compute_jsd: bool = False,
    compute_pathway: bool = False,
    compute_per_pathway: bool = False,
    ko_to_group: dict | None = None,
    log1p_pathway: bool = True,
) -> dict:
    metrics = {
        "dm_union_raw": dm_spearman_union(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=0.0, fillna_zero=True
        ),
        "dm_union_strict": dm_spearman_union(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "dm_union": dm_spearman_union(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=detect_threshold, fillna_zero=False
        ),
        "bray_union_raw": bray_spearman_union(truth_tpm, pred_tss, detect_threshold=0.0, fillna_zero=True),
        "bray_union": bray_spearman_union(
            truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "bray_union_legacy": bray_spearman_union(
            truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=False
        ),
        "procrustes_aitchison_raw": procrustes_union_aitchison(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=0.0, fillna_zero=True
        ),
        "procrustes_aitchison": procrustes_union_aitchison(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "procrustes_aitchison_legacy": procrustes_union_aitchison(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=detect_threshold, fillna_zero=False
        ),
        "procrustes_bray_raw": procrustes_union_bray(
            truth_tpm, pred_tss, detect_threshold=0.0, fillna_zero=True
        ),
        "procrustes_bray": procrustes_union_bray(
            truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "procrustes_bray_legacy": procrustes_union_bray(
            truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=False
        ),
    }

    prf = evaluate_union_metrics(
        truth_tpm,
        pred_tss,
        pseudocount=pseudocount,
        detect_threshold=detect_threshold,
        prf_thresh=prf_thresh,
        prf_weight=prf_weight,
        fillna_zero=True,
        compute_wclr=compute_wclr,
        compute_jsd=compute_jsd,
        compute_pathway=compute_pathway,
        compute_per_pathway=compute_per_pathway,
        ko_to_group=ko_to_group,
        log1p_pathway=log1p_pathway,
    )
    metrics.update(prf)
    return metrics


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--x", type=str, required=True)
    ap.add_argument("--out-prefix", type=str, default="results/full_predict")
    ap.add_argument("--y-truth", type=str, default=None)
    ap.add_argument("--pseudocount", type=float, default=0.5 / 1e6)
    ap.add_argument("--detect-threshold", type=float, default=0.0)
    ap.add_argument("--prf-thresh", type=float, default=1e-6)
    ap.add_argument("--prf-weight", type=str, default="binary")
    args = ap.parse_args()

    model = load_model(args.model)
    X_new = _read_table(Path(args.x))

    Yhat_clr, Yhat_tss, diag = predict_final_model(X_new, model)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Yhat_clr.to_csv(out_prefix.with_suffix(".clr.tsv"), sep="\t")
    Yhat_tss.to_csv(out_prefix.with_suffix(".tss.tsv"), sep="\t")
    diag.to_csv(out_prefix.with_suffix(".diag.tsv"), sep="\t")

    if args.y_truth:
        Y_truth = _read_table(Path(args.y_truth))
        metrics = evaluate_paired_subset(
            Y_truth,
            Yhat_tss,
            pseudocount=float(args.pseudocount),
            detect_threshold=float(args.detect_threshold),
            prf_thresh=float(args.prf_thresh),
            prf_weight=str(args.prf_weight),
        )
        pd.DataFrame([metrics]).to_csv(out_prefix.with_suffix(".metrics.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
