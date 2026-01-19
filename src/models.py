"""
Opinion update rules from Dandekar et al. (2013):
- undirected graph
- all edge weights w_ij = 1
- (by default) self-weight w_ii = 0

So for node i:
- degree d_i = number of neighbors
- s_i(t) = sum_{j in N(i)} x_j(t)

Degroot (def 2):
  x_i(t+1) = (w_ii * x_i(t) + s_i(t)) / (w_ii + d_i)

Biased opinion formation (def 3):
  x_i(t+1) = (w_ii*x_i(t) + (x_i(t))^b * s_i(t)) /
             (w_ii + (x_i(t))^b * s_i(t) + (1-x_i(t))^b * (d_i - s_i(t)))
"""

from __future__ import annotations
from typing import List, Sequence


def step_degroot(x: Sequence[float], adj: List[List[int]], w_self: float = 0.0) -> List[float]:
    """
    1 synchronous Degroot update
    x: opinions in [0,1]
    adj: adjacency list (undirected)
    w_self: self-weight w_ii (default 0.0 to match theorem 3 setting)
    """
    x_next = [0.0] * len(x)

    for i, nbrs in enumerate(adj):
        d = len(nbrs)
        s = 0.0
        for j in nbrs:
            s += float(x[j])

        denom = w_self + d
        if denom == 0:
            x_next[i] = float(x[i])
        else:
            x_next[i] = (w_self * float(x[i]) + s) / denom

        x_next[i] = _clip01(x_next[i])

    return x_next


def step_biased(
    x: Sequence[float],
    adj: List[List[int]],
    b: float,
    w_self: float = 0.0,
) -> List[float]:
    """
    1 synchronous biased-assimilation update (def 3) with a single global bias b >= 0
    Uses w_ij = 1 and (by default) w_ii = 0

    handle b==0 explicitly so the update matches degrot cleanly (avoids 0**0 corner case)
    """
    if b < 0:
        raise ValueError("b must be >= 0.")

    x_next = [0.0] * len(x)

    for i, nbrs in enumerate(adj):
        xi = float(x[i])
        xi = _clip01(xi)

        d = len(nbrs)
        s = 0.0
        for j in nbrs:
            s += float(x[j])

        # These two terms implement (xi)^b and (1-xi)^b
        # If b==0, both should be 1 so the formula reduces to degroot
        if b == 0.0:
            a = 1.0
            c = 1.0
        else:
            a = xi ** b
            c = (1.0 - xi) ** b

        num = w_self * xi + a * s
        denom = w_self + a * s + c * (d - s)

        if denom == 0:
            x_next[i] = xi
        else:
            x_next[i] = num / denom

        x_next[i] = _clip01(x_next[i])

    return x_next


def _clip01(v: float) -> float:
    """clamp a float to [0, 1]"""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
