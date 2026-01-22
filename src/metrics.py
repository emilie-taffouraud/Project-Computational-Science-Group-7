"""
Metrics for polarization/disagreement

We implement the paper’s main divergence measure:

Network Disagreement Index (NDI) (def 1):
  η(G, x) = sum_{(i,j) in E} w_ij * (x_i - x_j)^2

In our simplified setup:
- undirected graph
- all edge weights w_ij = 1
- edges are provided as (u, v) with u < v (from network.py)

So:
  NDI = sum_{(u,v) in edges} (x_u - x_v)^2
"""

from __future__ import annotations
from typing import Sequence, List, Tuple


def ndi(x: Sequence[float], edges: List[Tuple[int, int]]) -> float:
    """
    compute Network Disagreement Index for an undirected graph

    Args:
      x: opinions (usually in [0,1]), length N
      edges: list of undirected edges (u, v) with u < v

    Returns:
      float: NDI value
    """
    total = 0.0
    for u, v in edges:
        diff = float(x[u]) - float(x[v])
        total += diff * diff
    return total


def delta_ndi(x0: Sequence[float], xT: Sequence[float], edges: List[Tuple[int, int]]) -> float:
    """
    Convenience: ΔNDI = NDI(x_T) - NDI(x_0)
    """
    return ndi(xT, edges) - ndi(x0, edges)


def island_means(x: Sequence[float], n: int) -> Tuple[float, float]:
    """
    mean opinion in each island for the  2 island setup

    Args:
      x: opinions length 2n
      n: size of each island

    Returns:
      (mean_V1, mean_V2)
    """
    if len(x) != 2 * n:
        raise ValueError("Expected len(x) == 2*n for 2 island model")
    m1 = sum(float(v) for v in x[:n]) / n
    m2 = sum(float(v) for v in x[n:]) / n
    return m1, m2


def island_separation(x: Sequence[float], n: int) -> float:
    """
    absolute difference between island means: |mean(V1) - mean(V2)|
    nice secondary polarization indicator for plots/poster
    """
    m1, m2 = island_means(x, n)
    return abs(m1 - m2)



