from __future__ import annotations

import argparse
import importlib
import sys


def _dispatch_or_show_help(parser: argparse.ArgumentParser, module_name: str) -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        parser.parse_args()
        return

    module = importlib.import_module(module_name)
    module.main()


def extrapolate_train() -> None:
    parser = argparse.ArgumentParser(prog="intelligrate-extrapolate-train")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    _dispatch_or_show_help(parser, "intelligrate.extrapolate.train")


def extrapolate_full_fit() -> None:
    parser = argparse.ArgumentParser(prog="intelligrate-extrapolate-full-fit")
    parser.add_argument("--x", type=str, required=True)
    parser.add_argument("--y", type=str, required=True)
    parser.add_argument("--embed-path", type=str, default=None)
    parser.add_argument("--model-out", type=str, required=True)
    parser.add_argument("--min-prev-y-abs", type=int, default=1)
    parser.add_argument("--y-detect-threshold", type=float, default=0.0)
    parser.add_argument("--pseudocount-y", type=float, default=0.5 / 1e6)
    parser.add_argument("--neigh-k", type=int, default=12)
    parser.add_argument("--tau-mult", type=float, default=2.0)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--y-latent-k", type=int, default=10)
    parser.add_argument("--use-metric-learning", action="store_true")
    parser.add_argument("--metric-ridge", type=float, default=2.5)
    parser.add_argument("--metric-max-pairs", type=int, default=5000)
    parser.add_argument("--tau-scale-k-nn", type=int, default=10)
    parser.add_argument("--ood-shrink", action="store_true")
    parser.add_argument("--ood-lam-base", type=float, default=0.15)
    parser.add_argument("--ood-lam-cap", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=0)
    _dispatch_or_show_help(parser, "intelligrate.extrapolate.full_fit")


def extrapolate_full_predict() -> None:
    parser = argparse.ArgumentParser(prog="intelligrate-extrapolate-full-predict")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--x", type=str, required=True)
    parser.add_argument("--out-prefix", type=str, default="results/full_predict")
    parser.add_argument("--y-truth", type=str, default=None)
    parser.add_argument("--pseudocount", type=float, default=0.5 / 1e6)
    parser.add_argument("--detect-threshold", type=float, default=0.0)
    parser.add_argument("--prf-thresh", type=float, default=1e-6)
    parser.add_argument("--prf-weight", type=str, default="binary")
    _dispatch_or_show_help(parser, "intelligrate.extrapolate.full_predict")


def extrapolate_fixed_param_sweep() -> None:
    parser = argparse.ArgumentParser(prog="intelligrate-extrapolate-fixed-param-sweep")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, default="results/fixed_param_sweep.tsv")
    _dispatch_or_show_help(parser, "intelligrate.extrapolate.fixed_param_sweep")


def subset() -> None:
    parser = argparse.ArgumentParser(prog="intelligrate-subset")
    parser.add_argument("--config", type=str, required=True)
    _dispatch_or_show_help(parser, "intelligrate.subset.cli")
