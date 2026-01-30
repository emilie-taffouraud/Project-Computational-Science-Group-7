"""
runs the simulation grid and save outputs to CSV

Outputs:
- results/runs.csv      (one row per run)
- results/summary.csv   (aggregated by (h,b))
"""

from __future__ import annotations
import os
from src.simulate import run_grid
from src.analysis import summarize_by_condition, correlations, ttest_b0_vs_b1


def main() -> None:
    # experiment settings
    n = 100 # island size (total nodes = 2n)
    T = 50 # number of time steps
    repeats = 30 # runs per (h,b) condition
    seed0 = 0

    # we vary homophily by varying ps while keeping pd fixed
    # choose values so n*ps and n*pd are integers
    pd_value = 0.10
    ps_values = [0.10, 0.20, 0.40, 0.80] # gives h = 1,2,4,8

    # bias values (b=0 corresponds to degroot baseline)
    b_values = [0.0, 0.5, 0.8, 1.0, 1.2]

    # Opinion initialization
    init_mode = "theorem" # "theorem"/ "noisy"/ "random"
    x0 = 0.7
    noise = 0.05 # used only if init_mode is "noisy"

    # run simulations
    runs_df = run_grid(
        n=n,
        ps_values=ps_values,
        pd_value=pd_value,
        b_values=b_values,
        T=T,
        repeats=repeats,
        seed0=seed0,
        w_self=0.0,
        init_mode=init_mode,
        x0=x0,
        noise=noise,
    )

    # analyse
    os.makedirs("results", exist_ok=True)

    runs_path = os.path.join("results", "runs.csv")
    runs_df.to_csv(runs_path, index=False)

    summary_df = summarize_by_condition(runs_df)
    summary_path = os.path.join("results", "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # extra stats outputs 
    corr_by_b, corr_by_h = correlations(runs_df)
    corr_by_b.to_csv(os.path.join("results", "corr_by_b.csv"), index=False)
    corr_by_h.to_csv(os.path.join("results", "corr_by_h.csv"), index=False)

    ttests = ttest_b0_vs_b1(runs_df, b0=0.0, b1=1.0)
    ttests.to_csv(os.path.join("results", "ttest_b0_vs_b1.csv"), index=False)

    print(f"Saved: {runs_path}")
    print(f"Saved: {summary_path}")
    print("Saved: results/corr_by_b.csv, results/corr_by_h.csv, results/ttest_b0_vs_b1.csv")


if __name__ == "__main__":
    main()
