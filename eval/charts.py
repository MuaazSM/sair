"""Result charts from the committed eval outputs (PRD §13.3-§13.5).

Usage::

    uv run python eval/charts.py

Reads only committed result files (synthetic test set and synthetic chains):

- ``eval/results/synthetic/metrics.csv`` and ``paired.csv``
- ``eval/results/chains/contamination.json`` and ``paired_fact_minus_prose.json``

Writes ``eval/results/charts/{results_table,contamination,paired_diffs}.{png,svg}``.
No model calls, no network, no real-set data. Dev numbers are never plotted.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
OUT = RES / "charts"

# Brand (PRD §16b). The two series hues are the brand green and amber stepped until they
# clear the dataviz validator on the paper surface (lightness band, chroma floor >= 0.1,
# adjacent CVD dE >= 8, normal-vision dE >= 15, contrast >= 3:1):
#   node validate_palette.js "#1B6B3C,#C47A1E" --mode light --surface "#FAFAF6" -> all PASS
# The stock brand pair (#3F6B4A, #A5641A) fails the chroma floor and sits in the CVD warn band.
PAPER = "#FAFAF6"
INK = "#1D2A22"
MUTED = "#5C6B60"
DEEP_GREEN = "#2C4A34"
GRID = "#E6E8E1"
FACT = "#1B6B3C"  # brand green, stepped
PROSE = "#C47A1E"  # brand amber, stepped

SYSTEMS = ["A", "B", "C", "D", "B+", "C+", "D+"]
WRITERS = ["A", "B", "C", "D"]
SYSTEM_NAMES = {
    "A": "template only",
    "B": "Gemma 4 E2B zero-shot",
    "C": "Qwen3.5-4B few-shot",
    "D": "Sair-Writer (Q8_0)",
    "B+": "B + check, regen, template",
    "C+": "C + check, regen, template",
    "D+": "D + check, regen, template",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "svg.hashsalt": "sair-charts",
        "svg.fonttype": "path",
    }
)

# Reproducible files, no authoring metadata beyond the library name.
PNG_META = {"Software": None}
SVG_META = {"Date": None, "Creator": None}


def _f(x: str) -> float:
    return float(x)


def load_metrics() -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with (RES / "synthetic" / "metrics.csv").open() as fh:
        for r in csv.DictReader(fh):
            rows[(r["system"], r["metric"])] = {
                "point": _f(r["point"]),
                "lo": _f(r["lo"]),
                "hi": _f(r["hi"]),
                "n_items": _f(r["n_items"]),
                "n_clusters": _f(r["n_clusters"]),
            }
    return rows


def load_paired() -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with (RES / "synthetic" / "paired.csv").open() as fh:
        for r in csv.DictReader(fh):
            rows[(r["pair"], r["metric"])] = {
                "diff": _f(r["diff"]),
                "lo": _f(r["lo"]),
                "hi": _f(r["hi"]),
                "n_items": _f(r["n_items"]),
                "n_clusters": _f(r["n_clusters"]),
            }
    return rows


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def save(fig: Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, metadata=PNG_META)
    fig.savefig(OUT / f"{name}.svg", metadata=SVG_META)
    plt.close(fig)


def fmt(v: float | None, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}"


# --------------------------------------------------------------------------- table


def results_table(metrics: dict[tuple[str, str], dict[str, float]]) -> None:
    cols = [
        ("unsupported_rate", "Unsupported-\ndetail rate [95% CI]", True),
        ("details_per_entry", "Details /\nentry", False),
        ("hedge_preservation", "Hedge\npreservation", False),
        ("fact_precision", "Fact\nprecision", False),
        ("fact_recall", "Fact recall\n[95% CI]", True),
        ("template_fallback_rate", "Template-\nfallback rate", False),
        ("mission_accuracy", "Mission\naccuracy", False),
    ]
    n_items = int(metrics[("D+", "unsupported_rate")]["n_items"])
    n_clusters = int(metrics[("D+", "unsupported_rate")]["n_clusters"])

    fig = plt.figure(figsize=(12.6, 5.0))
    ax = fig.add_axes((0.02, 0.17, 0.96, 0.66))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(SYSTEMS) + 1.2, 0)

    x_sys, x_name = 0.005, 0.045
    col_x = [0.30, 0.43, 0.52, 0.61, 0.73, 0.86, 0.96]
    for (_key, head, _ci), x in zip(cols, col_x, strict=True):
        ax.text(x, 0.55, head, ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(x_sys, 0.55, "System", ha="left", va="center", fontsize=9.5, color=MUTED)
    ax.axhline(1.15, color=MUTED, lw=0.8)

    for i, s in enumerate(SYSTEMS):
        y = i + 1.75
        if s == "B+":
            ax.axhline(y - 0.5, color=GRID, lw=1.6)
        weight = "bold" if s in ("D", "D+") else "normal"
        ax.text(x_sys, y, s, ha="left", va="center", fontsize=11, weight="bold", color=INK)
        ax.text(x_name, y, SYSTEM_NAMES[s], ha="left", va="center", fontsize=9, color=MUTED)
        for (key, _head, ci), x in zip(cols, col_x, strict=True):
            m = metrics.get((s, key))
            if m is None:
                txt = "n/a"
                color = MUTED
            elif ci:
                txt = f"{fmt(m['point'])}\n[{fmt(m['lo'])}, {fmt(m['hi'])}]"
                color = INK
            else:
                txt = fmt(m["point"])
                color = INK
            ax.text(
                x,
                y,
                txt,
                ha="center",
                va="center",
                fontsize=9.5 if ci else 10,
                color=color,
                weight=weight,
                linespacing=1.15,
            )

    fig.text(
        0.02,
        0.93,
        "Writer results on the synthetic test set (point and 95% cluster-bootstrap CI)",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.02,
        0.885,
        "Final entries after each system's pipeline. Unsupported details and fallbacks: lower is better; "
        "the others: higher is better.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.02,
        0.035,
        f"N items = {n_items}, N seed clusters = {n_clusters} (held-out seeds). CIs: cluster bootstrap by seed, "
        "10,000 resamples. Synthetic set only; real set PENDING-OWNER.\n"
        "Checker-scored (no human annotation yet). n/a = no denominator (A writes no facts). "
        "B and B+ mission accuracy equals A's: B never emits mission_span, so check 7 coerces every result.",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.4,
    )
    save(fig, "results_table")


def _signed(text: str) -> str:
    """Typographic minus for negative numbers in labels."""
    return text.replace("-", "\u2212")


# ------------------------------------------------------------------- contamination


def contamination(cont: dict[str, Any], paired: dict[str, Any]) -> None:
    rows = [
        r
        for r in cont["rows"]
        if r["provenance"] == "off" and r["headline_measure"] == "raw_planner_output"
    ]
    by = {(r["system"], r["memory"]): r for r in rows}
    pdiff = {r["system"]: r for r in paired["rows"] if r["provenance"] == "off"}
    n_chains = sorted({int(r["n_chains"]) for r in rows})
    n_days = sorted({int(r["n_days"]) for r in rows})
    assert len(n_chains) == 1 and len(n_days) == 1
    nc, nd = n_chains[0], n_days[0]

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.4), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.72, bottom=0.25, wspace=0.10)
    days = list(range(1, nd + 1))
    for ax, w in zip(axes, WRITERS, strict=True):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        end: dict[str, float] = {}
        for mem, color, ls, mk, mfc, z in (
            ("prose", PROSE, (0, (4, 2.5)), "s", PAPER, 2),
            ("fact", FACT, "-", "o", FACT, 3),
        ):
            r = by[(w, mem)]
            ys = [100 * float(r["contaminated_by_day"][str(d)]) for d in days]
            ax.plot(
                days,
                ys,
                color=color,
                lw=2,
                ls=ls,
                marker=mk,
                ms=7 if mem == "fact" else 8.5,
                mfc=mfc,
                mec=color,
                mew=1.6,
                zorder=z,
                clip_on=False,
            )
            ci = r["pct_contaminated_by_last_day"]
            lo, hi = 100 * ci["lo"], 100 * ci["hi"]
            dx = 0.18 if mem == "fact" else -0.18
            ax.plot(
                [nd + dx, nd + dx], [lo, hi], color=color, lw=1.2, zorder=1, solid_capstyle="butt"
            )
            end[mem] = ys[-1]
        # direct labels at the right edge; nudge apart when the lines coincide
        f_y, p_y = end["fact"], end["prose"]
        f_off, p_off = (2.0, -2.2)
        ax.text(
            nd + 0.35,
            f_y + f_off,
            f"fact {f_y:.0f}%",
            color=INK,
            fontsize=9,
            va="center",
            ha="left",
        )
        ax.text(
            nd + 0.35,
            p_y + p_off,
            f"prose {p_y:.0f}%",
            color=INK,
            fontsize=9,
            va="center",
            ha="left",
        )
        pd = pdiff[w]
        ax.set_title(
            f"Writer {w}\nfact − prose, day {nd}: {100 * pd['point']:+.0f} pp "
            f"[{100 * pd['lo']:.0f}, {100 * pd['hi']:.0f}]",
            fontsize=10,
            color=INK,
            loc="left",
            linespacing=1.4,
        )
        ax.set_xlim(0.6, nd + 2.2)
        ax.set_xticks(days)
        ax.set_ylim(-1, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_yticklabels([f"{t}%" for t in (0, 5, 10, 15, 20, 25)])
        ax.set_xlabel("Day")
    axes[0].set_ylabel("Chains contaminated (cumulative)")

    fig.text(
        0.02,
        0.93,
        "No measurable contamination with either memory design",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.02,
        0.875,
        "Share of chains whose planner has proposed a mission with a concrete detail absent from every "
        "transcript so far. Solid green = fact memory, dashed amber = prose memory.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.02,
        0.035,
        f"N = {nc} synthetic chains × {nd} days per writer and memory arm; raw planner output, provenance filter off; "
        "planner gemma4:e2b. Vertical bars at day 7: 95% plain bootstrap\nCI over chains (10,000 resamples). "
        "Null result: prose memory never contaminated (0/20 for every writer). The one fact-arm flag for A and for D "
        "is for a number word only ('one', 'many').",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.4,
    )
    save(fig, "contamination")


# --------------------------------------------------------------------- forest plot


def paired_diffs(paired: dict[tuple[str, str], dict[str, float]]) -> None:
    pairs = ["D+ - C+", "D+ - B+", "D - C"]
    panels = [
        ("unsupported_rate", "Unsupported-detail rate", "lower is better", -1),
        ("fact_recall", "Fact recall", "higher is better", 1),
        ("template_fallback_rate", "Template-fallback rate", "lower is better", -1),
    ]
    first = paired[(pairs[0], panels[0][0])]
    n_items, n_clusters = int(first["n_items"]), int(first["n_clusters"])

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.72, bottom=0.25, wspace=0.12)
    ys = list(range(len(pairs)))
    for ax, (metric, title, better, _sign) in zip(axes, panels, strict=True):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
        for y, p in zip(ys, pairs, strict=True):
            r = paired[(p, metric)]
            ax.plot(
                [r["lo"], r["hi"]], [y, y], color=DEEP_GREEN, lw=2, zorder=2, solid_capstyle="round"
            )
            ax.plot([r["diff"]], [y], "o", color=DEEP_GREEN, ms=8, mec=PAPER, mew=2, zorder=3)
            ax.text(
                0.99,
                y - 0.30,
                _signed(f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        ax.set_xlim(-0.75, 0.75)
        ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
        ax.set_xticklabels(["−0.6", "−0.3", "0", "+0.3", "+0.6"])
        ax.set_title(f"{title}\n({better})", fontsize=10, color=INK, loc="left", linespacing=1.4)
        ax.set_xlabel("Paired difference (proportion)")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(
        ["D+ − C+", "D+ − B+", "D − C\n(first attempt)"], fontsize=10, color=INK
    )
    axes[0].set_ylim(len(pairs) - 0.4, -0.6)

    fig.text(
        0.02,
        0.93,
        "Paired differences between writers on the same items",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.02,
        0.875,
        "Point estimate and 95% cluster-bootstrap CI. A CI that excludes 0 is a difference "
        "this test set can resolve.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.02,
        0.035,
        f"Synthetic test set, N items = {n_items}, N seed clusters = {n_clusters}; paired cluster bootstrap by seed, "
        "10,000 resamples. Every '+' system ends at 0 unsupported\nby construction "
        "(grounding check, one regeneration, then the code-built template), so D+ − C+ and D+ − B+ on "
        "unsupported are 0 [0, 0]. Real set PENDING-OWNER.",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.4,
    )
    save(fig, "paired_diffs")


def main() -> None:
    metrics = load_metrics()
    paired = load_paired()
    cont = load_json(RES / "chains" / "contamination.json")
    pcont = load_json(RES / "chains" / "paired_fact_minus_prose.json")
    results_table(metrics)
    contamination(cont, pcont)
    paired_diffs(paired)
    for name in ("results_table", "contamination", "paired_diffs"):
        for ext in ("png", "svg"):
            print(f"wrote eval/results/charts/{name}.{ext}")


if __name__ == "__main__":
    main()
