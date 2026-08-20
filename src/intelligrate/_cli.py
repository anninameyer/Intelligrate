from __future__ import annotations

import argparse
import importlib
import sys
from importlib.metadata import PackageNotFoundError, version


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
            "  intelligrate subset run --config configs/subset_distance.yaml\n"
            "  intelligrate subset run --config configs/subset_kmedoids.yaml\n"
            "  intelligrate subset run --config configs/subset_ga.yaml\n\n"
            "The subset workflow is config-driven. The config field 'mode' selects one of: "
            "distance, suggest_k, kmedoids, ga."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    subparsers = parser.add_subparsers(dest="action", metavar="COMMAND")

    run = subparsers.add_parser(
        "run",
        help="Run a subset workflow from a YAML config.",
        description=(
            "Run one configured subset step. The config file controls which step runs via its "
            "'mode' field.\n\n"
            "Common modes and key config fields:\n"
            "  distance   feature_table, metric, assume_relative, pseudocount, output_dir\n"
            "  suggest_k  feature_table, distance_matrix, k_min, k_max, gap_B, seed, output_dir\n"
            "  kmedoids   distance_matrix, k, seed, output_dir\n"
            "  ga         cluster_table, metadata_table, total_samples, balance_vars, coord_vars,\n"
            "             population_size, generations, fixed_include, output_dir"
        ),
        formatter_class=_Formatter,
    )
    _add_config_arg(run, required=True)
    run.set_defaults(_module="intelligrate.subset.cli")
    parser.set_defaults(_help_if_no_action=True)
    return parser


def build_extrapolate_parser(prog: str = "intelligrate extrapolate") -> argparse.ArgumentParser:
    parser = _parser(
        prog,
        "Train and use models that extrapolate follow-up profiles from starting-layer features.",
        epilog=(
            "Examples:\n"
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

    train = subparsers.add_parser(
        "train",
        help="Run nested-CV training and write OOF predictions and metrics.",
        description=(
            "Run leakage-aware nested CV from a YAML config. Outputs include OOF CLR/TSS "
            "predictions, fold metrics, and summary files under results/."
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
            "Use this to choose one stable hyperparameter set before full fitting."
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
            "  intelligrate subset run --config configs/subset_ga.yaml\n"
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
    module_name = getattr(args, "_module", None)
    if module_name is None:
        parser.print_help()
        return
    action_argv = argv[2:]
    prog = "intelligrate " + " ".join(argv[:2])
    _dispatch(module_name, action_argv, prog)


def extrapolate_train() -> None:
    parser = build_extrapolate_parser("intelligrate-extrapolate")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        action = parser._subparsers._group_actions[0].choices["train"]  # type: ignore[attr-defined]
        action.prog = "intelligrate-extrapolate-train"
        action.parse_args(sys.argv[1:])
        return
    _dispatch("intelligrate.extrapolate.train", sys.argv[1:], "intelligrate-extrapolate-train")


def extrapolate_full_fit() -> None:
    parser = build_extrapolate_parser("intelligrate-extrapolate")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        action = parser._subparsers._group_actions[0].choices["full-fit"]  # type: ignore[attr-defined]
        action.prog = "intelligrate-extrapolate-full-fit"
        action.parse_args(sys.argv[1:])
        return
    _dispatch("intelligrate.extrapolate.full_fit", sys.argv[1:], "intelligrate-extrapolate-full-fit")


def extrapolate_full_predict() -> None:
    parser = build_extrapolate_parser("intelligrate-extrapolate")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        action = parser._subparsers._group_actions[0].choices["full-predict"]  # type: ignore[attr-defined]
        action.prog = "intelligrate-extrapolate-full-predict"
        action.parse_args(sys.argv[1:])
        return
    _dispatch("intelligrate.extrapolate.full_predict", sys.argv[1:], "intelligrate-extrapolate-full-predict")


def extrapolate_fixed_param_sweep() -> None:
    parser = build_extrapolate_parser("intelligrate-extrapolate")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        action = parser._subparsers._group_actions[0].choices["fixed-param-sweep"]  # type: ignore[attr-defined]
        action.prog = "intelligrate-extrapolate-fixed-param-sweep"
        action.parse_args(sys.argv[1:])
        return
    _dispatch(
        "intelligrate.extrapolate.fixed_param_sweep",
        sys.argv[1:],
        "intelligrate-extrapolate-fixed-param-sweep",
    )


def subset() -> None:
    parser = build_subset_parser("intelligrate-subset")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        action = parser._subparsers._group_actions[0].choices["run"]  # type: ignore[attr-defined]
        action.prog = "intelligrate-subset"
        action.parse_args(sys.argv[1:])
        return
    _dispatch("intelligrate.subset.cli", sys.argv[1:], "intelligrate-subset")


if __name__ == "__main__":
    main()
