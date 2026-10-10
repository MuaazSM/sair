"""Bootstrap confidence intervals for ratio metrics (PRD §13.3).

* Synthetic set: **cluster bootstrap by seed**. Each resample draws K seeds with
  replacement from the K seeds present and takes *all* items of every drawn seed; the
  statistic is ``sum(num) / sum(den)`` over the resample. 10,000 resamples by default,
  fixed RNG seed, 95 % percentile interval.
* Real set: plain bootstrap over items (each item is its own cluster).
* Paired differences (e.g. D+ minus C+): both systems' per-item ratios are aligned by item id
  and evaluated on the *same* resampled clusters, then differenced.

Always reported: point, lo, hi, N items, N clusters, resamples, and ``degenerate`` (True
when fewer than 2 clusters carry a non-zero denominator, in which case lo = hi = point).
Resamples whose denominator is 0 are dropped and counted in ``dropped``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eval.metrics import Ratio

__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_RNG_SEED",
    "CI",
    "cluster_bootstrap",
    "paired_bootstrap",
    "plain_bootstrap",
]

DEFAULT_RESAMPLES = 10_000
DEFAULT_RNG_SEED = 20261010


@dataclass(frozen=True)
class CI:
    point: float | None
    lo: float | None
    hi: float | None
    n_items: int
    n_clusters: int
    resamples: int
    method: str  # "cluster" | "plain" | "paired-cluster" | "paired-plain"
    degenerate: bool
    dropped: int = 0
    den: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cluster_sums(
    values: Sequence[Ratio], clusters: Sequence[str]
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64]]:
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have the same length")
    order: dict[str, int] = {}
    for c in clusters:
        order.setdefault(c, len(order))
    num = np.zeros(len(order))
    den = np.zeros(len(order))
    for v, c in zip(values, clusters, strict=True):
        k = order[c]
        num[k] += v.num
        den[k] += v.den
    return list(order), num, den


def _ratio(num: NDArray[np.float64], den: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        out: NDArray[np.float64] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def _percentiles(stats: NDArray[np.float64], alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cluster_bootstrap(
    values: Sequence[Ratio],
    clusters: Sequence[str],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    alpha: float = 0.05,
    method: str = "cluster",
) -> CI:
    _, num, den = _cluster_sums(values, clusters)
    k = len(num)
    total_den = float(den.sum())
    pt = None if total_den == 0 else float(num.sum() / total_den)
    informative = int((den > 0).sum())
    if pt is None or informative < 2:
        return CI(pt, pt, pt, len(values), k, 0, method, True, 0, total_den)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, k, size=(n_resamples, k))
    stats = _ratio(num[idx].sum(axis=1), den[idx].sum(axis=1))
    ok = stats[~np.isnan(stats)]
    lo, hi = _percentiles(ok, alpha)
    return CI(
        pt,
        lo,
        hi,
        len(values),
        k,
        n_resamples,
        method,
        False,
        int(n_resamples - ok.size),
        total_den,
    )


def plain_bootstrap(
    values: Sequence[Ratio],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    alpha: float = 0.05,
) -> CI:
    """Item-level bootstrap (real set): every item is its own cluster."""
    return cluster_bootstrap(
        values,
        [str(i) for i in range(len(values))],
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        alpha=alpha,
        method="plain",
    )


def paired_bootstrap(
    a: Sequence[Ratio],
    b: Sequence[Ratio],
    clusters: Sequence[str],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    alpha: float = 0.05,
    plain: bool = False,
) -> CI:
    """CI of ratio(a) - ratio(b), items aligned (same index = same item), same resamples."""
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("a, b and clusters must be aligned")
    cl = [str(i) for i in range(len(a))] if plain else list(clusters)
    _, na, da = _cluster_sums(a, cl)
    _, nb, db = _cluster_sums(b, cl)
    k = len(na)
    method = "paired-plain" if plain else "paired-cluster"
    if da.sum() == 0 or db.sum() == 0:
        return CI(None, None, None, len(a), k, 0, method, True)
    pt = float(na.sum() / da.sum() - nb.sum() / db.sum())
    informative = int(((da > 0) | (db > 0)).sum())
    if informative < 2:
        return CI(pt, pt, pt, len(a), k, 0, method, True)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, k, size=(n_resamples, k))
    diff = _ratio(na[idx].sum(axis=1), da[idx].sum(axis=1)) - _ratio(
        nb[idx].sum(axis=1), db[idx].sum(axis=1)
    )
    ok = diff[~np.isnan(diff)]
    lo, hi = _percentiles(ok, alpha)
    return CI(pt, lo, hi, len(a), k, n_resamples, method, False, int(n_resamples - ok.size))
