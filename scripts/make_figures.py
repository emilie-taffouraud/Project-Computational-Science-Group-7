"""
load results csv and generate figures

Inputs (created by run_experiments.py):
- results/runs.csv
- results/summary.csv

Outputs:
- figures/ndi_timeseries.png
- figures/heatmap_delta_ndi.png
- figures/final_opinion_dist.png

Notes:
- The time-series plot requires 'ndi_series' in runs_df
  By default, our main grid does not store it (to keep csv small),
  so here we generate a few demo runs with record_ndi=True
"""

from __future__ import annotations

import os
import pandas as pd

from src.simulate import run_simulation
from src.plots import (
    plot_ndi_timeseries_from_runs,
    plot_heatmap_delta_ndi,
    plot_final_opinion_hist,
)


def main() -> None:
    os.makedirs("figures", exist_ok=True)

    # heatmap from summary.csv
    summary_path = os.path.join("results", "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("results/summary.csv not found. Run scripts/run_experiments.py first.")

    summary_df = pd.read_csv(summary_path)

    plot_heatmap_delta_ndi(
        summary_df=summary_df,
        outpath=os.path.join("figures", "heatmap_delta_ndi.png"),
        value_col="mean_delta_ndi",
        title="Mean ΔNDI over (h, b)",
    )

    # ndi time-series (generate a few showcase runs)
    # pick a small set of conditions to visually compare:
    # degroot baseline (b=0) vs biased (b=1)
    # low homophily (h=1) vs high homophily (h=8)
    n = 100
    T = 50
    pd_value = 0.10

    demo_conditions = [
        # low h (ps=pd -> h=1)
        {"model": "degroot", "ps": 0.10, "b": 0.0, "seed": 1},
        {"model": "biased",  "ps": 0.10, "b": 1.0, "seed": 2},
        # high h (ps=0.80, pd=0.10 -> h=8)
        {"model": "degroot", "ps": 0.80, "b": 0.0, "seed": 3},
        {"model": "biased",  "ps": 0.80, "b": 1.0, "seed": 4},
    ]

    demo_rows = []
    demo_final_x = None # we’ll use one run for a histogram

    for cond in demo_conditions:
        res = run_simulation(
            model=cond["model"],
            n=n,
            ps=cond["ps"],
            pd_=pd_value,
            T=T,
            seed=cond["seed"],
            b=cond["b"],
            w_self=0.0,
            init_mode="theorem",
            x0=0.7,
            noise=0.0,
            tol=None,
            record_ndi=True,
        )
        # for the histogram we need x_T, but run_simulation currently doesn’t store it
        # Easiest: rerun once here and store the final opinions locally
        demo_rows.append(res)

    demo_df = pd.DataFrame(demo_rows)

    plot_ndi_timeseries_from_runs(
        runs_df=demo_df,
        outpath=os.path.join("figures", "ndi_timeseries.png"),
        title="NDI over time (DeGroot vs Biased, low vs high homophily)",
    )

    # final opinion histogram (one demo run)
    # we rerun that returns x_T by duplicating the loop locally
    from src.network import build_two_island
    from src.models import step_biased

    adj, edges = build_two_island(n=n, ps=0.80, pd=pd_value, seed=123)
    x = [0.7] * n + [0.3] * n
    for _ in range(T):
        x = step_biased(x, adj, b=1.0, w_self=0.0)

    plot_final_opinion_hist(
        x=x,
        outpath=os.path.join("figures", "final_opinion_dist.png"),
        title="Final opinion distribution (h=8, b=1.0)",
        bins=20,
    )

    print("Saved figures:")
    print("figures/heatmap_delta_ndi.png")
    print("figures/ndi_timeseries.png")
    print("figures/final_opinion_dist.png")


if __name__ == "__main__":
    main()
