"""
1) NDI over time (for a few selected runs)
2) Heatmap of mean ΔNDI over (h, b)
3) Final opinion distributions (histogram) for a couple showcase conditions
"""

from __future__ import annotations
from typing import List
import os
import ast
import pandas as pd
import matplotlib.pyplot as plt


def plot_ndi_timeseries_from_runs(
    runs_df: pd.DataFrame,
    outpath: str,
    title: str = "NDI over time",
) -> None:
    """
    expects runs_df to have a column 'ndi_series' containing a list of floats
    """
    if "ndi_series" not in runs_df.columns:
        raise ValueError("runs_df must contain 'ndi_series' column to plot time series")

    plt.figure()
    for idx, row in runs_df.iterrows():
        series = row["ndi_series"]
        if isinstance(series, str):
            series = ast.literal_eval(series)

        label = _label_from_row(row)
        plt.plot(range(len(series)), series, label=label)

    plt.xlabel("t")
    plt.ylabel("NDI")
    plt.title(title)
    plt.legend()
    _ensure_dir(outpath)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()


def plot_heatmap_delta_ndi(
    summary_df: pd.DataFrame,
    outpath: str,
    value_col: str = "mean_delta_ndi",
    title: str = "Mean ΔNDI over (h, b)",
) -> None:
    """
    heatmap from a summary table with columns: h, b, value_col
    designed to visualize where polarization (ΔNDI > 0) happens
    """
    needed = {"h", "b", value_col}
    missing = needed - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df missing columns: {sorted(missing)}")

    # pivot to matrix: rows=h, cols=b
    pivot = summary_df.pivot(index="h", columns="b", values=value_col)
    h_vals = pivot.index.to_numpy()
    b_vals = pivot.columns.to_numpy()
    Z = pivot.to_numpy()

    plt.figure()
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[b_vals.min(), b_vals.max(), h_vals.min(), h_vals.max()],
    )
    plt.colorbar(im, label=value_col)
    plt.xlabel("b (bias)")
    plt.ylabel("h (homophily = ps/pd)")
    plt.title(title)

    _ensure_dir(outpath)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()


def plot_final_opinion_hist(
    x: List[float],
    outpath: str,
    title: str = "Final opinion distribution",
    bins: int = 20,
) -> None:
    """
    Simple histogram of final opinions x_T for one run
    """
    plt.figure()
    plt.hist(x, bins=bins)
    plt.xlabel("opinion x")
    plt.ylabel("count")
    plt.title(title)

    _ensure_dir(outpath)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()


def _label_from_row(row: pd.Series) -> str:
    """
    for legend labels
    """
    model = row.get("model", "model")
    h = row.get("h", None)
    b = row.get("b", None)
    if h is None or b is None:
        return str(model)
    return f"{model}, h={h:g}, b={b:g}"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
