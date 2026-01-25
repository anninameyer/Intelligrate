from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .distance import compute_distance_matrix
from .k_selection import suggest_k
from .kmedoids import fit_kmedoids
from .ga import ga_subset
from .utils import check_alignment


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def _read_list(path: str | None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = cfg.get("mode", "distance")

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "distance":
        ft = _read_table(Path(cfg["feature_table"]))
        D = compute_distance_matrix(
            ft,
            metric=str(cfg.get("metric", "bray")),
            assume_relative=bool(cfg.get("assume_relative", False)),
            pseudocount=float(cfg.get("pseudocount", 1e-6)),
        )
        out_path = Path(cfg.get("distance_out", out_dir / "distance.tsv"))
        D.to_csv(out_path, sep="\t")
        meta = {"feature_samples": int(len(ft.index))}
        (out_dir / "distance_meta.json").write_text(json.dumps(meta, indent=2))
        return

    if mode == "suggest_k":
        ft = _read_table(Path(cfg["feature_table"]))
        D = _read_table(Path(cfg["distance_matrix"]))
        metrics = check_alignment(ft, D)
        result = suggest_k(
            D,
            ft,
            k_range=range(int(cfg.get("k_min", 2)), int(cfg.get("k_max", 31)) + 1),
            gap_B=int(cfg.get("gap_B", 5)),
            random_state=int(cfg.get("seed", 42)),
            return_fig=bool(cfg.get("plot", False)),
        )
        pd.DataFrame(
            {
                "k": result["k_values"],
                "silhouette": result["silhouette"],
                "davies_bouldin": result["davies_bouldin"],
                "gap": result["gap"],
                "gap_std": result["gap_std"],
            }
        ).to_csv(out_dir / "k_diagnostics.tsv", sep="\t", index=False)
        (out_dir / "k_suggest_meta.json").write_text(json.dumps(metrics, indent=2))
        if result.get("figure") is not None:
            result["figure"].savefig(out_dir / "k_diagnostics.png", dpi=300)
        return

    if mode == "kmedoids":
        D = _read_table(Path(cfg["distance_matrix"]))
        res = fit_kmedoids(D, int(cfg["k"]), random_state=int(cfg.get("seed", 42)))
        res["cluster_df"].to_csv(out_dir / "kmedoids_clusters.tsv", sep="\t")
        pd.DataFrame({"cluster": res["cluster_counts"].index, "n": res["cluster_counts"].values}).to_csv(
            out_dir / "kmedoids_cluster_counts.tsv", sep="\t", index=False
        )
        return

    if mode == "ga":
        clusters = _read_table(Path(cfg["cluster_table"]))
        metadata = _read_table(Path(cfg["metadata_table"]))
        metrics = check_alignment(clusters, metadata)

        fixed_include = _read_list(cfg.get("fixed_include"))
        fixed_exclude = _read_list(cfg.get("fixed_exclude"))

        result_df, best_scores, fitness_array = ga_subset(
            clusters,
            metadata,
            total_samples=int(cfg["total_samples"]),
            balance_vars=cfg.get("balance_vars", []),
            coord_vars=tuple(cfg.get("coord_vars", ["latitude", "longitude"])),
            min_category_n=int(cfg.get("min_category_n", 5)),
            min_per_category=int(cfg.get("min_per_category", 5)),
            grid_size=float(cfg.get("grid_size", 1.0)),
            population_size=int(cfg.get("population_size", 50)),
            generations=int(cfg.get("generations", 50)),
            random_state=int(cfg.get("seed", 42)),
            fixed_include=fixed_include,
            fixed_exclude=fixed_exclude,
            metadata_weights=cfg.get("metadata_weights"),
            grid_weight=float(cfg.get("grid_weight", 3.0)),
            distance_weight=float(cfg.get("distance_weight", 2.0)),
            balance_weight=float(cfg.get("balance_weight", 1.0)),
            balance_scale=float(cfg.get("balance_scale", 1000.0)),
            hard_penalty_weight=float(cfg.get("hard_penalty_weight", 100.0)),
        )

        result_df.to_csv(out_dir / "ga_selected_samples.tsv", sep="\t")
        pd.DataFrame({"best_score": best_scores}).to_csv(out_dir / "ga_best_scores.tsv", sep="\t", index=False)
        pd.DataFrame(fitness_array).T.to_csv(out_dir / "ga_fitness_array.tsv", sep="\t", index=False)
        (out_dir / "ga_meta.json").write_text(json.dumps(metrics, indent=2))
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
