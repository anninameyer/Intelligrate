from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from .cv_knn import nested_cv_knn_metric_latent_on_embedding
from .embedding import fit_x_embedding_svd_clr
from .metrics import aitchison_dm, corr_upper_triangle
from .transforms import clr_rows, keep_by_prevalence, tss_rows


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


def _dm_spearman_union(
    truth_tss: pd.DataFrame, pred_tss: pd.DataFrame, pseudocount: float
) -> float:
    common = truth_tss.index.intersection(pred_tss.index)
    truth = truth_tss.loc[common]
    pred = pred_tss.loc[common]
    union_cols = truth.columns.union(pred.columns)
    truth_u = truth.reindex(columns=union_cols, fill_value=0.0)
    pred_u = pred.reindex(columns=union_cols, fill_value=0.0)
    truth_clr = clr_rows(tss_rows(truth_u), pseudocount=pseudocount)
    pred_clr = clr_rows(tss_rows(pred_u), pseudocount=pseudocount)
    good = truth_clr.notna().all(axis=1) & pred_clr.notna().all(axis=1)
    truth_clr = truth_clr.loc[good]
    pred_clr = pred_clr.loc[good]
    return float(corr_upper_triangle(aitchison_dm(truth_clr), aitchison_dm(pred_clr), method="spearman"))


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
    model_dm_union = _dm_spearman_union(Y, oof_tss, pseudocount=union_pseudo)
    picrust2_dm_union = None
    delta_union = None
    if picrust2 is not None:
        picrust2_dm_union = _dm_spearman_union(Y, picrust2, pseudocount=union_pseudo)
        delta_union = float(model_dm_union - picrust2_dm_union)

    run = {
        "objective_dm_spearman_mean": dm_mean,
        "objective_dm_spearman_std": dm_std,
        "oof_dm_spearman": float(oof_dm),
        "model_dm_union": float(model_dm_union),
        "picrust2_dm_union": float(picrust2_dm_union) if picrust2_dm_union is not None else None,
        "delta_union": float(delta_union) if delta_union is not None else None,
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
    print(f"MODEL_DM_UNION={float(model_dm_union):.6f}")
    if picrust2_dm_union is not None:
        print(f"PICRUST2_DM_UNION={float(picrust2_dm_union):.6f}")
        print(f"DELTA_UNION={float(delta_union):.6f}")
    print(f"RESULTS_PREFIX={stamp}")


if __name__ == "__main__":
    main()
