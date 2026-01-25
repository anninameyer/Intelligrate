from __future__ import annotations

import numpy as np
import pandas as pd


def check_alignment(feature_table: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    ft_ids = pd.Index(feature_table.index)
    md_ids = pd.Index(metadata.index)
    common = ft_ids.intersection(md_ids)
    return {
        "feature_samples": int(len(ft_ids)),
        "metadata_samples": int(len(md_ids)),
        "common_samples": int(len(common)),
        "missing_in_metadata": int(len(ft_ids.difference(md_ids))),
        "missing_in_feature_table": int(len(md_ids.difference(ft_ids))),
    }


def ensure_relative_abundance(
    feature_table: pd.DataFrame,
    *,
    assume_relative: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    if assume_relative:
        return feature_table
    row_sums = feature_table.sum(axis=1)
    if np.allclose(row_sums.to_numpy(float), 1.0, atol=1e-6):
        return feature_table
    denom = row_sums.replace(0, np.nan)
    return feature_table.div(denom, axis=0)


def clr_transform(table: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    X = table.to_numpy(float) + float(pseudocount)
    X = np.log(X)
    X = X - X.mean(axis=1, keepdims=True)
    return pd.DataFrame(X, index=table.index, columns=table.columns)
