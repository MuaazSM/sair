"""Cluster / plain / paired bootstrap behaviour (PRD §13.3)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.bootstrap import cluster_bootstrap, paired_bootstrap, plain_bootstrap
from eval.metrics import Ratio


def _bernoulli(p: float, n_clusters: int, per: int, seed: int) -> tuple[list[Ratio], list[str]]:
    rng = np.random.default_rng(seed)
    vals, cl = [], []
    for c in range(n_clusters):
        for _ in range(per):
            vals.append(Ratio(float(rng.random() < p), 1.0))
            cl.append(f"s{c}")
    return vals, cl


def test_ci_contains_true_value() -> None:
    vals, cl = _bernoulli(0.3, 20, 15, seed=1)
    ci = cluster_bootstrap(vals, cl, n_resamples=2000)
    assert ci.lo is not None and ci.hi is not None
    assert ci.lo <= 0.3 <= ci.hi
    assert ci.n_items == 300 and ci.n_clusters == 20 and not ci.degenerate


def test_coverage_is_roughly_nominal() -> None:
    hits = 0
    for rep in range(40):
        vals, cl = _bernoulli(0.4, 15, 10, seed=100 + rep)
        ci = cluster_bootstrap(vals, cl, n_resamples=500, rng_seed=rep)
        assert ci.lo is not None and ci.hi is not None
        hits += ci.lo <= 0.4 <= ci.hi
    assert hits >= 32  # ~95 % nominal; loose bound for 40 replications


def test_reproducible_with_fixed_seed() -> None:
    vals, cl = _bernoulli(0.5, 10, 5, seed=3)
    a = cluster_bootstrap(vals, cl, n_resamples=1000)
    b = cluster_bootstrap(vals, cl, n_resamples=1000)
    assert a == b


def test_default_resamples_is_10k() -> None:
    vals, cl = _bernoulli(0.5, 5, 5, seed=4)
    assert cluster_bootstrap(vals, cl).resamples == 10_000


def test_single_cluster_is_degenerate() -> None:
    vals = [Ratio(1, 1), Ratio(0, 1), Ratio(1, 1)]
    ci = cluster_bootstrap(vals, ["s1"] * 3)
    assert ci.degenerate and ci.point == ci.lo == ci.hi == pytest.approx(2 / 3)
    assert ci.n_clusters == 1 and ci.resamples == 0


def test_zero_denominator_gives_none() -> None:
    ci = cluster_bootstrap([Ratio(0, 0), Ratio(0, 0)], ["a", "b"])
    assert ci.point is None and ci.degenerate


def test_cluster_ci_wider_than_plain_when_clusters_correlate() -> None:
    # each cluster is all-1 or all-0: strong within-cluster correlation
    vals = [Ratio(float(c % 2), 1) for c in range(10) for _ in range(20)]
    cl = [f"s{c}" for c in range(10) for _ in range(20)]
    clu = cluster_bootstrap(vals, cl, n_resamples=2000)
    pla = plain_bootstrap(vals, n_resamples=2000)
    assert clu.lo is not None and clu.hi is not None and pla.lo is not None and pla.hi is not None
    assert (clu.hi - clu.lo) > 2 * (pla.hi - pla.lo)
    assert pla.method == "plain" and pla.n_clusters == 200


def test_paired_difference_contains_true_diff_and_uses_same_clusters() -> None:
    rng = np.random.default_rng(7)
    a, b, cl = [], [], []
    for c in range(20):
        for _ in range(10):
            a.append(Ratio(float(rng.random() < 0.2), 1))
            b.append(Ratio(float(rng.random() < 0.5), 1))
            cl.append(f"s{c}")
    ci = paired_bootstrap(a, b, cl, n_resamples=2000)
    assert ci.lo is not None and ci.hi is not None
    assert ci.lo <= -0.3 <= ci.hi and ci.hi < 0
    same = paired_bootstrap(a, a, cl, n_resamples=500)
    assert same.point == 0 and same.lo == 0 and same.hi == 0


def test_paired_requires_alignment() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([Ratio(1, 1)], [], ["s"])
