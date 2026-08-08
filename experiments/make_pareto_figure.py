"""Render the README cost/quality figure from locked evaluation numbers.

    python experiments/make_pareto_figure.py

Every number below is transcribed from committed evaluation docs; nothing is
recomputed here. Sources:

- Quality (LoCoMo, 5 split seeds 7/11/23/31/47, raw dense top-500 pool):
  docs/EVIDENCE_RERANKER.md, "The locked v363 headline table".
- Latency (ms/query, RTX 4080 SUPER, memory embeddings precomputed):
  docs/EVIDENCE_RERANKER.md, "Cost", the v362 run.
- Raw dense latency was not measured in the v362 run. It is cosine scoring over
  precomputed embeddings and measures 0.01-0.13 ms/query in this repo's
  LongMemEval harness (see README). It is drawn at 0.1 ms and marked as an
  estimate so it is never read as a same-run measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets"

# name, latency ms/query, MRR, R@10, colour slot, label offset (points), ha, va, estimated latency
POINTS = [
    ("raw dense (no reranker)", 0.1, 0.3254, 0.5345, "aqua", (14, -2), "left", "center", True),
    ("ConvMemory v1", 16.8, 0.5824, 0.7798, "blue", (-14, -6), "right", "top", False),
    ("ConvMemory v1 + v2", 28.6, 0.6560, 0.7798, "blue", (-14, 6), "right", "bottom", False),
    ("mxbai-rerank-large-v1 (full top-500 pool)", 1960.2, 0.6688, 0.8080, "orange", (-10, -26), "right", "top", False),
]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#e4e3df",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#333331",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
    },
}

TITLE = "Most of the reranking gain, without a large cross-encoder over the whole pool"
SUBTITLE = "LoCoMo, 5 split seeds, top-500 candidate pool.  Latency: RTX 4080 SUPER, memory embeddings precomputed."


def render(mode: str) -> Path:
    theme = THEMES[mode]
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    ax.plot(
        [p[1] for p in POINTS],
        [p[2] for p in POINTS],
        color=theme["text_secondary"],
        linewidth=1.2,
        alpha=0.45,
        zorder=1,
    )

    for name, latency, mrr, recall, slot, offset, ha, va, estimated in POINTS:
        ax.scatter(
            [latency],
            [mrr],
            s=170,
            color=theme[slot],
            edgecolors=theme["surface"],
            linewidths=2.0,
            zorder=3,
        )
        latency_text = f"~{latency:.1f} ms (est.)" if estimated else f"{latency:,.1f} ms"
        ax.annotate(
            f"{name}\n{latency_text}   MRR {mrr:.3f}   R@10 {recall:.3f}",
            xy=(latency, mrr),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=theme["text_primary"],
            linespacing=1.5,
            ha=ha,
            va=va,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.04, 9000)
    ax.set_ylim(0.28, 0.74)
    ax.set_xlabel("reranking latency per query (ms, log scale)", fontsize=10, color=theme["text_secondary"])
    ax.set_ylabel("retrieval quality (MRR)", fontsize=10, color=theme["text_secondary"])

    ax.set_title(TITLE, fontsize=13, color=theme["text_primary"], loc="left", pad=26, weight="bold")
    ax.text(
        0.0,
        1.035,
        SUBTITLE,
        transform=ax.transAxes,
        fontsize=8.5,
        color=theme["text_secondary"],
    )

    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["text_secondary"], labelsize=9)

    ax.text(
        0.46,
        0.40,
        "ConvMemory v1 + v2 reaches 98% of the mxbai MRR at 1/68 of its latency.",
        transform=ax.transAxes,
        fontsize=10,
        color=theme["text_primary"],
        va="top",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pareto-{mode}.png"
    fig.savefig(out_path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print("wrote", render(mode))
