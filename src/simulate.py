"""
simulation runner + simple parameter sweeps

Uses:
- network.build_two_island() to create the undirected exact-degree two-island graph
- models.step_degroot() / models.step_biased() for updates
- metrics.ndi() (+ optional island separation)

outputs are designed to be saved into one big csv (runs.csv)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import random
import pandas as pd
from src.network import build_two_island
from src.models import step_degroot, step_biased
from src.metrics import ndi, island_separation


def init_opinions(
    n: int,
    mode: str = "theorem",
    x0: float = 0.7,
    noise: float = 0.0,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Initialize opinions for 2n nodes

    mode:
      - "theorem": V1 all x0, V2 all (1-x0) (matches the simple theorem-style setup)
      - "noisy": same as theorem but add uniform noise in [-noise, +noise] and clip to [0,1]
      - "random": all opinions ~ Uniform(0,1)

    Returns list length 2n, values in [0,1]
    """
    rng = random.Random(seed)
    N = 2 * n

    if mode == "theorem":
        x = [x0] * n + [1.0 - x0] * n
        return [_clip01(v) for v in x]

    if mode == "noisy":
        x = [x0] * n + [1.0 - x0] * n
        out = []
        for v in x:
            v2 = v + rng.uniform(-noise, noise)
            out.append(_clip01(v2))
        return out

    if mode == "random":
        return [rng.random() for _ in range(N)]

    raise ValueError("mode must be one of: 'theorem', 'noisy', 'random'")


def run_simulation(
    model: str,
    n: int,
    ps: float,
    pd_: float,
    T: int,
    seed: int,
    b: float = 0.0,
    w_self: float = 0.0,
    init_mode: str = "theorem",
    x0: float = 0.7,
    noise: float = 0.0,
    tol: Optional[float] = None,
    record_ndi: bool = False,
) -> Dict[str, Any]:
    """
    Run 1 simulation

    model:
      - "degroot"
      - "biased"

    tol:
      If set, we early-stop when max_i |x_i(t+1)-x_i(t)| < tol

    record_ndi:
      If True, record ndi(t) and gdi(t) for t=0..T 
    """
    adj, edges = build_two_island(n=n, ps=ps, pd=pd_, seed=seed)

    x = init_opinions(n=n, mode=init_mode, x0=x0, noise=noise, seed=seed)
    ndi_series = [ndi(x, edges)] if record_ndi else None
    gdi_series = [gdi(x)] if record_ndi else None
    x0_copy = list(x)

    steps = 0
    for t in range(T):
        if model == "degroot":
            x_next = step_degroot(x, adj, w_self=w_self)
        elif model == "biased":
            x_next = step_biased(x, adj, b=b, w_self=w_self)
        else:
            raise ValueError("model must be 'degroot' or 'biased'")

        steps += 1

        if tol is not None:
            max_change = max(abs(a - b2) for a, b2 in zip(x_next, x))
            x = x_next
            if record_ndi:
                ndi_series.append(ndi(x, edges))
                gdi_series.append(gdi(x))
            if max_change < tol:
                break
        else:
            x = x_next
            if record_ndi:
                ndi_series.append(ndi(x, edges))
                gdi_series.append(gdi(x))

    ndi0 = ndi(x0_copy, edges)
    ndiT = ndi(x, edges)
    gdi0 = gdi(x0_copy)
    gdiT = gdi(x)

    out: Dict[str, Any] = {
        "model": model,
        "n": n,
        "ps": ps,
        "pd": pd_,
        "h": (ps / pd_) if pd_ != 0 else float("inf"),
        "b": b,
        "w_self": w_self,
        "T": T,
        "steps_run": steps,
        "seed": seed,
        "init_mode": init_mode,
        "x0": x0,
        "noise": noise,
        "ndi_0": ndi0,
        "ndi_T": ndiT,
        "delta_ndi": ndiT - ndi0,
        "gdi_0": gdi0,
        "gdi_T": gdiT,
        "delta_gdi": gdiT - gdi0,
        "island_sep_T": island_separation(x, n),
    }

    if record_ndi:
        out["ndi_series"] = ndi_series  # list of floats
        out["gdi_series"] = gdi_series

    # out["x_T"] = x

    return out


def run_grid(
    n: int,
    ps_values: List[float],
    pd_value: float,
    b_values: List[float],
    T: int,
    repeats: int,
    seed0: int = 0,
    w_self: float = 0.0,
    init_mode: str = "theorem",
    x0: float = 0.7,
    noise: float = 0.0,
) -> pd.DataFrame:
    """
    Run a simple grid:
      ps in ps_values (controls homophily with fixed pd_value)
      b in b_values
      repeats per condition (different seeds)

    returns a df with 1 row per run
    """
    rows: List[Dict[str, Any]] = []
    seed = seed0

    for ps in ps_values:
        for b in b_values:
            for _ in range(repeats):
                rows.append(
                    run_simulation(
                        model="biased" if b != 0 else "degroot",
                        n=n,
                        ps=ps,
                        pd_=pd_value,
                        T=T,
                        seed=seed,
                        b=b,
                        w_self=w_self,
                        init_mode=init_mode,
                        x0=x0,
                        noise=noise,
                        tol=None,
                        record_ndi=False,
                    )
                )
                seed += 1

    return pd.DataFrame(rows)


def _clip01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

