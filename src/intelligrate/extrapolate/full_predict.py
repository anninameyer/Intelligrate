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
    evaluate_intersection_metrics,
    procrustes_union_aitchison,
    procrustes_union_bray,
    samplewise_spearman,
    aitchison_dm,
    corr_upper_triangle,
    _pairwise_union_mats_tss,
    _pairwise_intersection_mats_tss,
)
from .transforms import clr_to_comp, clr_rows
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
    ko_to_group: dict | str | Path | None = None,
    log1p_pathway: bool = True,
    include_intersection: bool = True,
    compute_null: bool = False,
    null_n: int = 10,
    null_seed: int = 0,
    null_mean_noise_scale: float = 0.01,
) -> dict:
    if ko_to_group is not None and not isinstance(ko_to_group, dict):
        ko_path = Path(ko_to_group)
        if ko_path.exists():
            ko_df = pd.read_csv(ko_path, sep="\t", header=None)
            if ko_df.shape[1] < 2:
                raise ValueError("ko_to_group file must have at least 2 columns (KO, group).")
            ko_to_group = dict(zip(ko_df.iloc[:, 0], ko_df.iloc[:, 1]))
        else:
            raise ValueError(f"ko_to_group path not found: {ko_path}")

    metrics = {
        "dm_union_raw": dm_spearman_union(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=0.0, fillna_zero=True
        ),
        "dm_union_strict": dm_spearman_union(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "bray_union_raw": bray_spearman_union(truth_tpm, pred_tss, detect_threshold=0.0, fillna_zero=True),
        "bray_union_strict": bray_spearman_union(
            truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "procrustes_aitchison_raw": procrustes_union_aitchison(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=0.0, fillna_zero=True
        ),
        "procrustes_aitchison_strict": procrustes_union_aitchison(
            truth_tpm, pred_tss, pseudocount=pseudocount, detect_threshold=detect_threshold, fillna_zero=True
        ),
        "procrustes_bray_raw": procrustes_union_bray(
            truth_tpm, pred_tss, detect_threshold=0.0, fillna_zero=True
        ),
        "procrustes_bray_strict": procrustes_union_bray(
            truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=True
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

    prf_raw = evaluate_union_metrics(
        truth_tpm,
        pred_tss,
        pseudocount=pseudocount,
        detect_threshold=0.0,
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
    metrics.update({f"{k}_union_raw": v for k, v in prf_raw.items()})

    if include_intersection:
        inter = evaluate_intersection_metrics(
            truth_tpm,
            pred_tss,
            pseudocount=pseudocount,
            detect_threshold=0.0,
            prf_thresh=prf_thresh,
            prf_weight=prf_weight,
            compute_wclr=compute_wclr,
            compute_jsd=compute_jsd,
            compute_pathway=compute_pathway,
            compute_per_pathway=compute_per_pathway,
            ko_to_group=ko_to_group,
            log1p_pathway=log1p_pathway,
        )
        metrics.update({f"intersection_{k}": v for k, v in inter.items()})

    # Sample-wise Spearman correlations (intersection, union_raw, union_strict)
    def _sample_spearman_from_mats(truth_tss: pd.DataFrame, pred_tss: pd.DataFrame, suffix: str) -> None:
        if truth_tss.shape[0] == 0:
            return
        truth_clr = clr_rows(truth_tss, pseudocount=pseudocount)
        pred_clr = clr_rows(pred_tss, pseudocount=pseudocount)
        good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
        truth_tss = truth_tss.loc[good]
        pred_tss = pred_tss.loc[good]
        truth_clr = truth_clr.loc[good]
        pred_clr = pred_clr.loc[good]
        if truth_tss.shape[0] == 0:
            return
        sample_tss = samplewise_spearman(truth_tss, pred_tss)
        sample_clr = samplewise_spearman(truth_clr, pred_clr)
        metrics[f"sample_spearman_tss_{suffix}"] = float(sample_tss.mean())
        metrics[f"sample_spearman_clr_{suffix}"] = float(sample_clr.mean())

    truth_tss_u_strict, pred_tss_u_strict = _pairwise_union_mats_tss(
        truth_tpm, pred_tss, detect_threshold=detect_threshold, fillna_zero=True
    )
    truth_tss_u_raw, pred_tss_u_raw = _pairwise_union_mats_tss(
        truth_tpm, pred_tss, detect_threshold=0.0, fillna_zero=True
    )
    truth_tss_i, pred_tss_i = _pairwise_intersection_mats_tss(
        truth_tpm, pred_tss, detect_threshold=0.0
    )
    _sample_spearman_from_mats(truth_tss_u_strict, pred_tss_u_strict, "union_strict")
    _sample_spearman_from_mats(truth_tss_u_raw, pred_tss_u_raw, "union_raw")
    _sample_spearman_from_mats(truth_tss_i, pred_tss_i, "intersection")

    if compute_null and truth_tss_u_strict.shape[0] > 0:
        rng = np.random.default_rng(int(null_seed))
        n = truth_tss_u_strict.shape[0]
        pred_tss_u_arr = pred_tss_u_strict.to_numpy(float)

        def _compute_null_metrics(pred_tss_null: pd.DataFrame) -> dict:
            return evaluate_union_metrics(
                truth_tpm,
                pred_tss_null,
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

        null_acc = {"sample": [], "feature": [], "mean": []}
        null_sample_spearman_tss = {"sample": [], "feature": [], "mean": []}
        null_sample_spearman_clr = {"sample": [], "feature": [], "mean": []}
        for _ in range(int(null_n)):
            perm = rng.permutation(n)
            pred_tss_perm = pd.DataFrame(
                pred_tss_u_arr[perm, :], index=truth_tss_u_strict.index, columns=truth_tss_u_strict.columns
            )
            null_acc["sample"].append(_compute_null_metrics(pred_tss_perm))
            null_sample_spearman_tss["sample"].append(
                float(samplewise_spearman(truth_tss_u_strict, pred_tss_perm).mean())
            )
            null_sample_spearman_clr["sample"].append(
                float(samplewise_spearman(clr_rows(truth_tss_u_strict, pseudocount=pseudocount),
                                          clr_rows(pred_tss_perm, pseudocount=pseudocount)).mean())
            )

            perm_feat = rng.permutation(pred_tss_u_arr.shape[1])
            pred_tss_feat = pd.DataFrame(
                pred_tss_u_arr[:, perm_feat], index=truth_tss_u_strict.index, columns=truth_tss_u_strict.columns
            )
            null_acc["feature"].append(_compute_null_metrics(pred_tss_feat))
            null_sample_spearman_tss["feature"].append(
                float(samplewise_spearman(truth_tss_u_strict, pred_tss_feat).mean())
            )
            null_sample_spearman_clr["feature"].append(
                float(samplewise_spearman(clr_rows(truth_tss_u_strict, pseudocount=pseudocount),
                                          clr_rows(pred_tss_feat, pseudocount=pseudocount)).mean())
            )

            mu_tss = truth_tss_u_strict.mean(axis=0).to_numpy(float)
            noise_tss = rng.normal(0.0, float(null_mean_noise_scale), size=pred_tss_u_arr.shape)
            pred_tss_mean = pd.DataFrame(
                np.clip(mu_tss[None, :] + noise_tss, 0.0, None),
                index=truth_tss_u_strict.index,
                columns=truth_tss_u_strict.columns,
            )
            null_acc["mean"].append(_compute_null_metrics(pred_tss_mean))
            null_sample_spearman_tss["mean"].append(
                float(samplewise_spearman(truth_tss_u_strict, pred_tss_mean).mean())
            )
            null_sample_spearman_clr["mean"].append(
                float(samplewise_spearman(clr_rows(truth_tss_u_strict, pseudocount=pseudocount),
                                          clr_rows(pred_tss_mean, pseudocount=pseudocount)).mean())
            )

        metric_keys = [
            "dm_spearman",
            "bray_spearman",
            "procrustes_aitchison",
            "procrustes_bray",
            "soft_precision",
            "soft_recall",
            "soft_f1",
            "wclr_mse",
            "jsd",
            "pathway_rmse",
        ]
        for k in metric_keys:
            for mode in ("sample", "feature", "mean"):
                vals = [d.get(k) for d in null_acc[mode] if d.get(k) is not None]
                metrics[f"null_{mode}_{k}_union_strict_mean"] = float(np.nanmean(vals)) if vals else np.nan
                metrics[f"null_{mode}_{k}_union_strict_std"] = float(np.nanstd(vals)) if vals else np.nan
        for mode in ("sample", "feature", "mean"):
            metrics[f"null_{mode}_sample_spearman_tss_union_strict_mean"] = float(
                np.nanmean(null_sample_spearman_tss[mode])
            )
            metrics[f"null_{mode}_sample_spearman_tss_union_strict_std"] = float(
                np.nanstd(null_sample_spearman_tss[mode])
            )
            metrics[f"null_{mode}_sample_spearman_clr_union_strict_mean"] = float(
                np.nanmean(null_sample_spearman_clr[mode])
            )
            metrics[f"null_{mode}_sample_spearman_clr_union_strict_std"] = float(
                np.nanstd(null_sample_spearman_clr[mode])
            )
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
