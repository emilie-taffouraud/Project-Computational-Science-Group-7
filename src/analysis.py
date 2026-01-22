"""
a basic statistical analysis for the simulation results

Input: a df like results/runs.csv (one row per run), with at least:
- h, b, delta_ndi, ndi_0, ndi_T, island_sep_T

Outputs:
- summary table by (h, b): mean/std of delta_ndi + mean island separation
- correlations: (h vs delta_ndi) per b, and (b vs delta_ndi) per h
- simple significance tests: t-test comparing b=0 vs b=1 within each h (if available)
"""

from __future__ import annotations
from typing import Tuple
import pandas as pd
from scipy import stats


def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    this agregates per (h, b) condition

    retruns a df with:
      h, b, runs, mean_delta_ndi, std_delta_ndi, mean_ndi_T, mean_island_sep_T
    """
    needed = {"h", "b", "delta_ndi", "ndi_T", "island_sep_T"}
    _require_cols(df, needed)

    summary = (
        df.groupby(["h", "b"], as_index=False)
          .agg(
              runs=("delta_ndi", "size"),
              mean_delta_ndi=("delta_ndi", "mean"),
              std_delta_ndi=("delta_ndi", "std"),
              mean_ndi_T=("ndi_T", "mean"),
              mean_island_sep_T=("island_sep_T", "mean"),
          )
          .sort_values(["h", "b"])
          .reset_index(drop=True)
    )
    return summary


def correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    computes simple Pearson correlations:

    1) corr(h, delta_ndi) for each fixed b
    2) corr(b, delta_ndi) for each fixed h

    Returns:
      corr_by_b_df, corr_by_h_df
    """
    needed = {"h", "b", "delta_ndi"}
    _require_cols(df, needed)

    rows_b = []
    for b_val, sub in df.groupby("b"):
        if sub["h"].nunique() < 2:
            continue
        r, p = stats.pearsonr(sub["h"], sub["delta_ndi"])
        rows_b.append({"b": b_val, "pearson_r": r, "p_value": p, "n": len(sub)})

    rows_h = []
    for h_val, sub in df.groupby("h"):
        if sub["b"].nunique() < 2:
            continue
        r, p = stats.pearsonr(sub["b"], sub["delta_ndi"])
        rows_h.append({"h": h_val, "pearson_r": r, "p_value": p, "n": len(sub)})

    corr_by_b = pd.DataFrame(rows_b).sort_values("b").reset_index(drop=True)
    corr_by_h = pd.DataFrame(rows_h).sort_values("h").reset_index(drop=True)
    return corr_by_b, corr_by_h


def ttest_b0_vs_b1(df: pd.DataFrame, b0: float = 0.0, b1: float = 1.0) -> pd.DataFrame:
    """
    simple significance test:
      For each h, compare delta_ndi distributions at b=b0 vs b=b1 ( t-test)

    Returns a df with:
      h, n_b0, n_b1, mean_b0, mean_b1, t_stat, p_value
    """
    needed = {"h", "b", "delta_ndi"}
    _require_cols(df, needed)

    rows = []
    for h_val, sub_h in df.groupby("h"):
        g0 = sub_h[sub_h["b"] == b0]["delta_ndi"].dropna()
        g1 = sub_h[sub_h["b"] == b1]["delta_ndi"].dropna()

        if len(g0) < 2 or len(g1) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)

        rows.append(
            {
                "h": h_val,
                "b0": b0,
                "b1": b1,
                "n_b0": len(g0),
                "n_b1": len(g1),
                "mean_b0": float(g0.mean()),
                "mean_b1": float(g1.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }
        )

    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)


def _require_cols(df: pd.DataFrame, cols: set) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")



