"""
Undirected two-island (homophily) network with exact degrees

Two groups:
- V1: 0..n-1
- V2: n..2n-1

Each node has:
- deg_same = n*ps neighbors in its own island
- deg_diff = n*pd neighbors in the other island

Construction:
within each island: deterministic d-regular circulant graph (always works if feasible)
across islands: deterministic regular bipartite graph (always works if d <= n)

this avoids random stub-pairing failures for high degrees that we encoutered
"""

from __future__ import annotations
from typing import List, Tuple


def build_two_island(
    n: int,
    ps: float,
    pd: float,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    deg_same = _int_degree(n, ps, "ps")
    deg_diff = _int_degree(n, pd, "pd")
    _validate(n, deg_same, deg_diff)

    N = 2 * n
    adj: List[List[int]] = [[] for _ in range(N)]
    edges_set = set()

    V1 = list(range(0, n))
    V2 = list(range(n, 2 * n))

    # within-island edges (d regular)
    edges_set |= _circulant_regular_edges(V1, deg_same)
    edges_set |= _circulant_regular_edges(V2, deg_same)

    # cross-island edges (regular bipartite)
    edges_set |= _bipartite_regular_edges(V1, V2, n, deg_diff)

    # build adjacency
    edges = sorted(edges_set)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if not _check_exact_degrees(adj, n, deg_same, deg_diff):
        raise RuntimeError("Internal error: constructed graph does not match exact degrees.")

    for nbrs in adj:
        nbrs.sort()
    return adj, edges


# helper functons 

def _int_degree(n: int, p: float, name: str) -> int:
    val = n * p
    deg = int(round(val))
    if abs(val - deg) > 1e-9:
        raise ValueError(f"n*{name} must be an integer. Got n*{name}={val}.")
    return deg


def _validate(n: int, deg_same: int, deg_diff: int) -> None:
    if n < 2:
        raise ValueError("n must be >= 2.")
    if deg_same < 0 or deg_diff < 0:
        raise ValueError("degrees must be non-negative.")
    if deg_same >= n:
        raise ValueError("deg_same must be < n (no self-loops within an island).")
    if deg_diff > n:
        raise ValueError("deg_diff must be <= n (max cross-island neighbors is n).")

    # For an undirected d-regular graph on n nodes:
    # - n*d must be even
    if (n * deg_same) % 2 != 0:
        raise ValueError("n*deg_same must be even for a regular undirected graph.")

    # If deg_same is odd, our circulant construction needs n even (so n/2 is integer).
    if (deg_same % 2 == 1) and (n % 2 == 1):
        raise ValueError("If deg_same is odd, n must be even for this construction.")


def _circulant_regular_edges(nodes: List[int], d: int) -> set[Tuple[int, int]]:
    """
    Deterministic undirected d-regular graph on given nodes via circulant offsets.

    If d is even:
      connect offsets 1..d/2
    If d is odd (requires n even):
      connect offsets 1..(d-1)/2 plus offset n/2 (perfect matching)

    Returns edges as (u,v) with u<v.
    """
    m = len(nodes)
    if d == 0:
        return set()

    edges = set()
    half = d // 2

    # offsets 1..half
    for offset in range(1, half + 1):
        for i in range(m):
            u = nodes[i]
            v = nodes[(i + offset) % m]
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))

    # if d is odd, add offset m/2 (perfect matching), requires m even
    if d % 2 == 1:
        if m % 2 != 0:
            raise ValueError("Odd degree requires even island size for this construction.")
        offset = m // 2
        for i in range(m):
            u = nodes[i]
            v = nodes[(i + offset) % m]
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))

    return edges


def _bipartite_regular_edges(
    left: List[int],
    right: List[int],
    n: int,
    d: int,
) -> set[Tuple[int, int]]:
    """
    Deterministic d-regular bipartite graph between left and right (both size n).

    left[i] connects to right[(i + k) % n] for k=0..d-1
    Ensures every left and every right has exactly degree d (when d<=n).
    """
    if d == 0:
        return set()

    edges = set()
    for i in range(n):
        u = left[i]
        for k in range(d):
            v = right[(i + k) % n]
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return edges


def _check_exact_degrees(adj: List[List[int]], n: int, deg_same: int, deg_diff: int) -> bool:
    N = 2 * n
    if len(adj) != N:
        return False

    for u in range(N):
        within = 0
        cross = 0
        for v in adj[u]:
            if v == u:
                return False
            same_island = (u < n and v < n) or (u >= n and v >= n)
            if same_island:
                within += 1
            else:
                cross += 1

        if within != deg_same or cross != deg_diff:
            return False

    return True
