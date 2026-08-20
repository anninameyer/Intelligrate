from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd


class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _package_version() -> str:
    try:
        return version("intelligrate")
    except PackageNotFoundError:
        return "installed from source"


def _parser(prog: str, description: str, epilog: str | None = None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=_Formatter,
    )


def _dispatch(module_name: str, argv: list[str], prog: str) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [prog, *argv]
        module = importlib.import_module(module_name)
        module.main()
    finally:
        sys.argv = old_argv


def _read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), sep="\t", index_col=0)


def _read_list(path: str | None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_weights(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    weights: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("metadata weights must be comma-separated name=value pairs")
        key, raw_val = item.split("=", 1)
        weights[key.strip()] = float(raw_val)
    return weights


def _write_template(template_name: str, out_path: str | Path, *, force: bool = False) -> Path:
    out = Path(out_path)
    if out.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {out}. Use --force to replace it.")
    out.parent.mkdir(parents=True, exist_ok=True)
    template = resources.files("intelligrate").joinpath("templates").joinpath(template_name)
    out.write_bytes(template.read_bytes())
    return out


def _handle_extrapolate_write_config(args: argparse.Namespace) -> None:
    out = _write_template("extrapolate_default.yaml", args.out, force=bool(args.force))
    print(f"Wrote extrapolate config template: {out}")
    print("Edit the data paths and parameters, then run:")
    print(f"  intelligrate extrapolate train --config {out}")


def _handle_subset_write_configs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    template_names = [
        "subset_distance.yaml",
        "subset_k.yaml",
        "subset_kmedoids.yaml",
        "subset_ga.yaml",
        "fixed_include.tsv",
    ]
    written = [_write_template(name, out_dir / name, force=bool(args.force)) for name in template_names]
    print("Wrote subset config templates:")
    for path in written:
        print(f"  {path}")
    print("Edit the paths and parameters, then run a config step, for example:")
    print(f"  intelligrate subset run-config --config {out_dir / 'subset_distance.yaml'}")


def _handle_subset_distance(args: argparse.Namespace) -> None:
    from intelligrate.subset import compute_distance_matrix

    ft = _read_table(args.feature_table)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    D = compute_distance_matrix(
        ft,
        metric=args.metric,
        assume_relative=bool(args.assume_relative),
        pseudocount=float(args.pseudocount),
    )
    out_path = Path(args.distance_out) if args.distance_out else out_dir / "distance.tsv"
    D.to_csv(out_path, sep="\t")
    (out_dir / "distance_meta.json").write_text(json.dumps({"feature_samples": int(len(ft.index))}, indent=2))
    print(f"Wrote distance matrix: {out_path}")


def _handle_subset_suggest_k(args: argparse.Namespace) -> None:
    from intelligrate.subset import suggest_k
    from intelligrate.subset.utils import check_alignment

    ft = _read_table(args.feature_table)
    D = _read_table(args.distance_matrix)
    metrics = check_alignment(ft, D)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = suggest_k(
        D,
        ft,
        k_range=range(int(args.k_min), int(args.k_max) + 1),
        gap_B=int(args.gap_B),
        random_state=int(args.seed),
        return_fig=bool(args.plot),
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
    print(f"Wrote k diagnostics: {out_dir / 'k_diagnostics.tsv'}")


def _handle_subset_kmedoids(args: argparse.Namespace) -> None:
    from intelligrate.subset import fit_kmedoids

    D = _read_table(args.distance_matrix)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fit_kmedoids(D, int(args.k), random_state=int(args.seed))
    res["cluster_df"].to_csv(out_dir / "kmedoids_clusters.tsv", sep="\t")
    pd.DataFrame({"cluster": res["cluster_counts"].index, "n": res["cluster_counts"].values}).to_csv(
        out_dir / "kmedoids_cluster_counts.tsv", sep="\t", index=False
    )
    print(f"Wrote k-medoids clusters: {out_dir / 'kmedoids_clusters.tsv'}")


def _handle_subset_ga(args: argparse.Namespace) -> None:
    from intelligrate.subset import ga_subset
    from intelligrate.subset.utils import check_alignment

    clusters = _read_table(args.cluster_table)
    metadata = _read_table(args.metadata_table)
    metrics = check_alignment(clusters, metadata)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df, best_scores, fitness_array = ga_subset(
        clusters,
        metadata,
        total_samples=int(args.total_samples),
        balance_vars=_parse_csv_list(args.balance_vars),
        coord_vars=(args.latitude_col, args.longitude_col),
        min_category_n=int(args.min_category_n),
        min_per_category=int(args.min_per_category),
        grid_size=float(args.grid_size),
        population_size=int(args.population_size),
        generations=int(args.generations),
        random_state=int(args.seed),
        fixed_include=_read_list(args.fixed_include),
        fixed_exclude=_read_list(args.fixed_exclude),
        metadata_weights=_parse_weights(args.metadata_weights),
        grid_weight=float(args.grid_weight),
        distance_weight=float(args.distance_weight),
        balance_weight=float(args.balance_weight),
        balance_scale=float(args.balance_scale),
        hard_penalty_weight=float(args.hard_penalty_weight),
    )

    result_df.to_csv(out_dir / "ga_selected_samples.tsv", sep="\t")
    pd.DataFrame({"best_score": best_scores}).to_csv(out_dir / "ga_best_scores.tsv", sep="\t", index=False)
    pd.DataFrame(fitness_array).T.to_csv(out_dir / "ga_fitness_array.tsv", sep="\t", index=False)
    (out_dir / "ga_meta.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote selected samples: {out_dir / 'ga_selected_samples.tsv'}")


def _add_config_arg(parser: argparse.ArgumentParser, *, required: bool, default: str | None = None) -> None:
    parser.add_argument(
        "--config",
        required=required,
        default=default,
        metavar="PATH",
        help=(
            "YAML configuration file. Paths inside the repository examples usually point to "
            "configs/*.yaml and write outputs under results/."
        ),
    )


def build_subset_parser(prog: str = "intelligrate subset") -> argparse.ArgumentParser:
    parser = _parser(
        prog,
        "Select representative sample subsets using distance matrices, k-medoids, and a genetic algorithm.",
        epilog=(
            "Examples:\n"
            "  intelligrate subset write-configs --out-dir configs\n"
            "  intelligrate subset distance --feature-table data/HF_sourdough/feature_table_rel.tsv --assume-relative\n"
            "  intelligrate subset suggest-k --feature-table data/HF_sourdough/feature_table_rel.tsv --distance-matrix results/subset/distance.tsv\n"
            "  intelligrate subset kmedoids --distance-matrix results/subset/distance.tsv --k 3\n"
            "  intelligrate subset ga --cluster-table results/subset/kmedoids_clusters.tsv --metadata-table data/HF_sourdough/metadata.tsv --total-samples 30"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    subparsers = parser.add_subparsers(dest="action", metavar="COMMAND")

    write_configs = subparsers.add_parser(
        "write-configs",
        help="Write editable subset YAML config templates.",
        description=(
            "Write editable subset config templates to a directory. These templates contain the "
            "same default values as the repository examples and are useful after pip installation."
        ),
        formatter_class=_Formatter,
    )
    write_configs.add_argument("--out-dir", default="configs", metavar="DIR", help="Directory to write templates into.")
    write_configs.add_argument("--force", action="store_true", help="Overwrite existing template files.")
    write_configs.set_defaults(_handler=_handle_subset_write_configs)

    distance = subparsers.add_parser(
        "distance",
        help="Compute a sample-sample distance matrix.",
        description="Compute a distance matrix from a samples x features table.",
        formatter_class=_Formatter,
    )
    _add_subset_distance_args(distance)
    distance.set_defaults(_handler=_handle_subset_distance)

    suggest_k = subparsers.add_parser(
        "suggest-k",
        help="Compute diagnostics to help choose k.",
        description="Compute silhouette, Davies-Bouldin, and gap-statistic diagnostics over a k range.",
        formatter_class=_Formatter,
    )
    _add_subset_suggest_k_args(suggest_k)
    suggest_k.set_defaults(_handler=_handle_subset_suggest_k)

    kmedoids = subparsers.add_parser(
        "kmedoids",
        help="Fit k-medoids clusters from a distance matrix.",
        description="Cluster samples using a precomputed distance matrix and write cluster assignments.",
        formatter_class=_Formatter,
    )
    _add_subset_kmedoids_args(kmedoids)
    kmedoids.set_defaults(_handler=_handle_subset_kmedoids)

    ga = subparsers.add_parser(
        "ga",
        help="Select a balanced subset with a genetic algorithm.",
        description=(
            "Select samples from k-medoids clusters while balancing metadata categories and optional "
            "geographic spread."
        ),
        formatter_class=_Formatter,
    )
    _add_subset_ga_args(ga)
    ga.set_defaults(_handler=_handle_subset_ga)

    run_config = subparsers.add_parser(
        "run-config",
        help="Run one subset step from a YAML config.",
        description=(
            "Run a config-driven subset step. The config field 'mode' selects one of: "
            "distance, suggest_k, kmedoids, ga."
        ),
        formatter_class=_Formatter,
    )
    _add_config_arg(run_config, required=True)
    run_config.set_defaults(_module="intelligrate.subset.cli")
    parser.set_defaults(_help_if_no_action=True)
    return parser


def _add_subset_distance_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--feature-table", required=True, metavar="TSV", help="Samples x features table.")

    params = parser.add_argument_group("Parameters")
    params.add_argument(
        "--metric",
        default="bray",
        choices=["bray", "jaccard", "aitchison"],
        help="Distance metric. Bray and Aitchison are abundance-aware; Jaccard is presence/absence.",
    )
    params.add_argument("--assume-relative", action="store_true", help="Treat rows as already relative abundance.")
    params.add_argument("--pseudocount", type=float, default=1e-6, help="Pseudocount for Aitchison/CLR distances.")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--output-dir", default="results/subset", metavar="DIR", help="Directory for output files.")
    outputs.add_argument("--distance-out", metavar="TSV", help="Optional custom path for the distance matrix.")


def _add_subset_suggest_k_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--feature-table", required=True, metavar="TSV", help="Samples x features table used for diagnostics.")
    inputs.add_argument("--distance-matrix", required=True, metavar="TSV", help="Square sample-sample distance matrix.")

    params = parser.add_argument_group("Parameters")
    params.add_argument("--k-min", type=int, default=2, help="Smallest k to evaluate.")
    params.add_argument("--k-max", type=int, default=31, help="Largest k to evaluate.")
    params.add_argument("--gap-B", type=int, default=5, help="Number of reference draws for the gap statistic.")
    params.add_argument("--seed", type=int, default=42, help="Random seed for reproducible diagnostics.")
    params.add_argument("--plot", action="store_true", help="Also write k_diagnostics.png.")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--output-dir", default="results/subset", metavar="DIR", help="Directory for output files.")


def _add_subset_kmedoids_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--distance-matrix", required=True, metavar="TSV", help="Square sample-sample distance matrix.")

    params = parser.add_argument_group("Parameters")
    params.add_argument("--k", type=int, required=True, help="Number of k-medoids clusters.")
    params.add_argument("--seed", type=int, default=42, help="Random seed for reproducible clustering.")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--output-dir", default="results/subset", metavar="DIR", help="Directory for output files.")


def _add_subset_ga_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--cluster-table", required=True, metavar="TSV", help="k-medoids cluster table with a Cluster column.")
    inputs.add_argument("--metadata-table", required=True, metavar="TSV", help="Sample metadata table.")
    inputs.add_argument("--fixed-include", metavar="TXT", help="Optional newline-delimited sample IDs to force include.")
    inputs.add_argument("--fixed-exclude", metavar="TXT", help="Optional newline-delimited sample IDs to exclude.")

    params = parser.add_argument_group("Selection parameters")
    params.add_argument("--total-samples", type=int, required=True, help="Target number of selected samples.")
    params.add_argument("--balance-vars", default="", help="Comma-separated metadata columns to balance.")
    params.add_argument("--latitude-col", default="latitude", help="Latitude metadata column.")
    params.add_argument("--longitude-col", default="longitude", help="Longitude metadata column.")
    params.add_argument("--population-size", type=int, default=50, help="GA population size.")
    params.add_argument("--generations", type=int, default=50, help="Number of GA generations.")
    params.add_argument("--seed", type=int, default=42, help="Random seed for reproducible GA selection.")

    constraints = parser.add_argument_group("Balance constraints")
    constraints.add_argument("--min-category-n", type=int, default=5, help="Ignore categories with fewer samples than this.")
    constraints.add_argument("--min-per-category", type=int, default=5, help="Minimum selected samples per retained category.")
    constraints.add_argument("--metadata-weights", help="Comma-separated name=value weights for balance variables.")

    objective = parser.add_argument_group("Objective weights")
    objective.add_argument("--grid-size", type=float, default=1.0, help="Latitude/longitude grid size for spatial spread.")
    objective.add_argument("--grid-weight", type=float, default=3.0, help="Weight for geographic grid coverage.")
    objective.add_argument("--distance-weight", type=float, default=2.0, help="Weight for geographic pairwise distance.")
    objective.add_argument("--balance-weight", type=float, default=1.0, help="Weight for metadata balance.")
    objective.add_argument("--balance-scale", type=float, default=1000.0, help="Scale factor for metadata balance score.")
    objective.add_argument("--hard-penalty-weight", type=float, default=100.0, help="Penalty for minimum-category violations.")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--output-dir", default="results/subset", metavar="DIR", help="Directory for output files.")


def build_extrapolate_parser(prog: str = "intelligrate extrapolate") -> argparse.ArgumentParser:
    parser = _parser(
        prog,
        "Train and use models that extrapolate follow-up profiles from starting-layer features.",
        epilog=(
            "Examples:\n"
            "  intelligrate extrapolate write-config --out configs/default.yaml\n"
            "  intelligrate extrapolate train --config configs/default.yaml\n"
            "  intelligrate extrapolate fixed-param-sweep --config configs/default.yaml\n"
            "  intelligrate extrapolate full-fit --x data/HF_sourdough/X_kmers.tsv --y data/HF_sourdough/Y_kos.tsv \\\n"
            "      --embed-path results/embed.joblib --model-out results/model.joblib\n"
            "  intelligrate extrapolate full-predict --model results/model.joblib \\\n"
            "      --x data/HF_sourdough/X_kmers_full.tsv --out-prefix results/pred_full"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    subparsers = parser.add_subparsers(dest="action", metavar="COMMAND")

    write_config = subparsers.add_parser(
        "write-config",
        help="Write an editable extrapolate YAML config template.",
        description=(
            "Write an editable extrapolate config template. The template contains default values "
            "for data paths, cross-validation, embedding, model grids, objective weights, metrics, "
            "and fixed-parameter sweeps."
        ),
        formatter_class=_Formatter,
    )
    write_config.add_argument(
        "--out",
        default="configs/default.yaml",
        metavar="PATH",
        help="Output YAML path for the template config.",
    )
    write_config.add_argument("--force", action="store_true", help="Overwrite an existing config file.")
    write_config.set_defaults(_handler=_handle_extrapolate_write_config)

    train = subparsers.add_parser(
        "train",
        help="Run nested-CV training and write OOF predictions and metrics.",
        description=(
            "Run leakage-aware nested CV from a YAML config. Outputs include OOF CLR/TSS "
            "predictions, fold metrics, and summary files under results/.\n\n"
            "Start from the installed template:\n"
            "  intelligrate extrapolate write-config --out configs/default.yaml\n"
            "Then edit the data paths and parameters in that YAML file and run:\n"
            "  intelligrate extrapolate train --config configs/default.yaml\n\n"
            "Key config sections and fields:\n"
            "  data      x_full, x, y, optional picrust2, optional ko_to_superclass\n"
            "  cv        outer_splits, inner_splits, seed, informed_splits\n"
            "  embed     min_prev_x_abs, pseudocount_x, n_components\n"
            "  model     min_prev_y_abs, y_detect_threshold, pseudocount_y,\n"
            "            neigh_k_grid, tau_mult_grid, lam_grid, y_latent_k_grid,\n"
            "            use_metric_learning, metric_ridge_grid, metric_max_pairs,\n"
            "            tau_scale_k_nn, ood_shrink, ood_lam_base, ood_lam_cap\n"
            "  objective w_dm, w_wclr, w_pw_rmse, w_softf1, w_jsd\n"
            "  metrics   compute_wclr, compute_jsd, compute_pathway_rmse,\n"
            "            pathway_rmse_per_group, pathway_rmse_log1p\n"
            "  prf       prf_thresh, prf_weight\n"
            "  score     min_prev_y_abs, y_detect_threshold, pseudocount_y"
        ),
        formatter_class=_Formatter,
    )
    _add_config_arg(train, required=False, default="configs/default.yaml")
    train.set_defaults(_module="intelligrate.extrapolate.train")

    sweep = subparsers.add_parser(
        "fixed-param-sweep",
        help="Evaluate fixed hyperparameter combinations with OOF predictions.",
        description=(
            "Run a fixed-parameter sweep from the config's fixed_param_sweep block. "
            "Use this to choose one stable hyperparameter set before full fitting.\n\n"
            "Start from the installed template if you do not already have a config:\n"
            "  intelligrate extrapolate write-config --out configs/default.yaml\n\n"
            "Common fixed_param_sweep fields:\n"
            "  neigh_k, tau_mult, y_latent_k, metric_ridge, lam,\n"
            "  min_prev_y_abs, y_detect_threshold, pseudocount_y,\n"
            "  ood_lam_base, ood_lam_cap, use_metric_learning,\n"
            "  metric_max_pairs, tau_scale_k_nn, outer_splits, seed\n\n"
            "If a field is absent from fixed_param_sweep, the value is taken from the main "
            "config. Fields can be single values or lists to sweep."
        ),
        formatter_class=_Formatter,
    )
    _add_config_arg(sweep, required=False, default="configs/default.yaml")
    sweep.add_argument(
        "--out",
        default="results/fixed_param_sweep.tsv",
        metavar="PATH",
        help="Output TSV with one row per fixed-parameter combination.",
    )
    sweep.set_defaults(_module="intelligrate.extrapolate.fixed_param_sweep")

    full_fit = subparsers.add_parser(
        "full-fit",
        help="Fit the final deployable model on all paired samples.",
        description=(
            "Fit a final extrapolation model on all paired X/Y samples using fixed "
            "hyperparameters and a precomputed X embedding."
        ),
        formatter_class=_Formatter,
    )
    _add_full_fit_args(full_fit)
    full_fit.set_defaults(_module="intelligrate.extrapolate.full_fit")

    full_predict = subparsers.add_parser(
        "full-predict",
        help="Predict follow-up profiles for new samples.",
        description=(
            "Predict target profiles for samples with X features. Writes CLR predictions, TSS "
            "predictions, and OOD diagnostics. If --y-truth is provided, also writes metrics."
        ),
        formatter_class=_Formatter,
    )
    _add_full_predict_args(full_predict)
    full_predict.set_defaults(_module="intelligrate.extrapolate.full_predict")

    parser.set_defaults(_help_if_no_action=True)
    return parser


def _add_full_fit_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--x", required=True, metavar="TSV", help="Paired X table: samples x starting-layer features.")
    inputs.add_argument("--y", required=True, metavar="TSV", help="Paired Y table: samples x target features.")
    inputs.add_argument("--embed-path", required=True, metavar="JOBLIB", help="Fitted X embedding joblib file.")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--model-out", required=True, metavar="JOBLIB", help="Path for the fitted model artifact.")

    yprep = parser.add_argument_group("Y preprocessing")
    yprep.add_argument("--min-prev-y-abs", type=int, default=1, help="Keep Y features seen in at least this many samples.")
    yprep.add_argument("--y-detect-threshold", type=float, default=0.0, help="Detection threshold applied to Y before prevalence filtering.")
    yprep.add_argument("--pseudocount-y", type=float, default=0.5 / 1e6, help="Pseudocount used for Y CLR transforms.")

    model = parser.add_argument_group("Model hyperparameters")
    model.add_argument("--neigh-k", type=int, default=12, help="Number of nearest neighbors used for prediction.")
    model.add_argument("--tau-mult", type=float, default=2.0, help="Multiplier for the kNN kernel bandwidth.")
    model.add_argument("--lam", type=float, default=0.0, help="Shrinkage toward the training-set mean in latent/CLR space.")
    model.add_argument("--y-latent-k", type=int, default=10, help="Number of Y latent dimensions; use 0 to predict CLR directly.")
    model.add_argument("--seed", type=int, default=0, help="Random seed for reproducible fitting.")

    metric = parser.add_argument_group("Metric learning")
    metric.add_argument("--use-metric-learning", action="store_true", help="Use supervised diagonal metric learning in embedded X space.")
    metric.add_argument("--metric-ridge", type=float, default=2.5, help="Ridge regularization for metric learning.")
    metric.add_argument("--metric-max-pairs", type=int, default=5000, help="Maximum sample pairs used for metric learning.")
    metric.add_argument("--tau-scale-k-nn", type=int, default=10, help="Neighbor count used to estimate the distance scale for tau.")

    ood = parser.add_argument_group("OOD shrinkage")
    ood.add_argument("--ood-shrink", action="store_true", help="Shrink predictions for samples far from the training set.")
    ood.add_argument("--ood-lam-base", type=float, default=0.15, help="Baseline OOD shrinkage strength.")
    ood.add_argument("--ood-lam-cap", type=float, default=0.80, help="Maximum OOD shrinkage strength.")


def _add_full_predict_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--model", required=True, metavar="JOBLIB", help="Model artifact produced by full-fit.")
    inputs.add_argument("--x", required=True, metavar="TSV", help="X table to predict for: samples x starting-layer features.")
    inputs.add_argument("--y-truth", metavar="TSV", help="Optional Y truth table for evaluating paired samples.")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument(
        "--out-prefix",
        default="results/full_predict",
        metavar="PREFIX",
        help="Output prefix. Writes .clr.tsv, .tss.tsv, .diag.tsv, and optionally .metrics.tsv.",
    )

    metrics = parser.add_argument_group("Evaluation options")
    metrics.add_argument("--pseudocount", type=float, default=0.5 / 1e6, help="Pseudocount used for CLR-based evaluation.")
    metrics.add_argument("--detect-threshold", type=float, default=0.0, help="Detection threshold for strict union metrics.")
    metrics.add_argument("--prf-thresh", type=float, default=1e-6, help="Threshold for precision/recall/F1 evaluation.")
    metrics.add_argument(
        "--prf-weight",
        default="binary",
        choices=["binary", "truth_abundance", "pred_abundance"],
        help="Weighting scheme for thresholded precision/recall/F1.",
    )


def build_root_parser() -> argparse.ArgumentParser:
    parser = _parser(
        "intelligrate",
        "Diversity-aware sample subsetting and extrapolation between paired data layers.",
        epilog=(
            "Examples:\n"
            "  intelligrate subset --help\n"
            "  intelligrate extrapolate write-config --out configs/default.yaml\n"
            "  intelligrate subset distance --feature-table data/HF_sourdough/feature_table_rel.tsv --assume-relative\n"
            "  intelligrate extrapolate --help\n"
            "  intelligrate extrapolate train --config configs/default.yaml"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    subparsers = parser.add_subparsers(dest="workflow", metavar="COMMAND")

    subset_parser = build_subset_parser("intelligrate subset")
    extrapolate_parser = build_extrapolate_parser("intelligrate extrapolate")
    subparsers.add_parser(
        "subset",
        parents=[subset_parser],
        add_help=False,
        help="Representative sample selection workflows.",
    )
    subparsers.add_parser(
        "extrapolate",
        parents=[extrapolate_parser],
        add_help=False,
        help="Train and use extrapolation models.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_root_parser()
    if not argv:
        parser.print_help()
        return
    args, remaining = parser.parse_known_args(argv)
    if getattr(args, "_help_if_no_action", False) and getattr(args, "action", None) is None:
        # Re-parse the selected workflow to show its command list.
        workflow_parser = build_subset_parser() if args.workflow == "subset" else build_extrapolate_parser()
        workflow_parser.print_help()
        return
    handler = getattr(args, "_handler", None)
    if handler is not None:
        handler(args)
        return
    module_name = getattr(args, "_module", None)
    if module_name is None:
        parser.print_help()
        return
    action_argv = argv[2:]
    prog = "intelligrate " + " ".join(argv[:2])
    _dispatch(module_name, action_argv, prog)


if __name__ == "__main__":
    main()
