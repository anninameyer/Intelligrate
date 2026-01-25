from __future__ import annotations

import numpy as np
import pandas as pd


def tss_rows(df: pd.DataFrame) -> pd.DataFrame:
    denom = df.sum(axis=1).replace(0, np.nan)
    return df.div(denom, axis=0)


def clr_rows(df: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    X = df.to_numpy(dtype=float)
    X = X + float(pseudocount)
    X = np.log(X)
    X = X - X.mean(axis=1, keepdims=True)
    return pd.DataFrame(X, index=df.index, columns=df.columns)


def keep_by_prevalence(df: pd.DataFrame, min_prev_abs: int, detect_threshold: float = 0.0) -> pd.Index:
    prev = (df.to_numpy(dtype=float) > float(detect_threshold)).sum(axis=0)
    keep = prev >= int(min_prev_abs)
    return df.columns[keep]


def clr_to_comp(Y_clr: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    CLR -> composition via stable exp + closure.
    """
    A = Y_clr.to_numpy(float)
    A = A - np.nanmax(A, axis=1, keepdims=True)  # stability shift
    X = np.exp(A)
    X = np.clip(X, 0, None)
    X = X / (X.sum(axis=1, keepdims=True) + eps)
    return pd.DataFrame(X, index=Y_clr.index, columns=Y_clr.columns)
