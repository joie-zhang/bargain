#!/usr/bin/env python3
"""Plot the audited harmful-coalition proposal and acceptance flows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
ASSET_DIR = ROOT / "docs/analysis/assets/minimum_winning_coalition_20260817"
OUTPUT_STEM = ASSET_DIR / "coalition_proposal_acceptance_sankey"
EDGE_CSV = ASSET_DIR / "coalition_proposal_acceptance_sankey_edges.csv"

COLORS = {
    "root": "#334155",
    "game1": "#2563EB",
    "game2": "#0F766E",
    "game3": "#7C3AED",
    "heterogeneous": "#2563EB",
    "monoculture": "#EA580C",
    "adversary": "#7C3AED",
    "accepted": "#15803D",
    "rejected": "#94A3B8",
    "literal_zero": "#B91C1C",
    "harmful_nonzero": "#D97706",
    "full_support": "#15803D",
    "replacement": "#475569",
}


@dataclass(frozen=True)
class Node:
    key: str
    label: str
    x: float
    y: float
    height: float
    color: str
    label_side: str = "right"


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    value: int
    color: str
    label: str = ""


def rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b = mpl.colors.to_rgb(hex_color)
    return r, g, b, alpha


def draw_sankey(
    ax: plt.Axes,
    nodes: list[Node],
    links: list[Link],
    scale: float,
    node_width: float = 0.018,
) -> None:
    """Draw a fixed-layout Sankey with cubic ribbons in axis coordinates."""
    by_key = {node.key: node for node in nodes}
    outgoing_offsets = {node.key: 0.0 for node in nodes}
    incoming_offsets = {node.key: 0.0 for node in nodes}

    # Draw links before nodes so the rectangles remain sharp.
    for link in links:
        source = by_key[link.source]
        target = by_key[link.target]
        thickness = link.value * scale
        sy0 = source.y + outgoing_offsets[source.key]
        sy1 = sy0 + thickness
        ty0 = target.y + incoming_offsets[target.key]
        ty1 = ty0 + thickness
        outgoing_offsets[source.key] += thickness
        incoming_offsets[target.key] += thickness

        sx = source.x + node_width
        tx = target.x
        curve = 0.46 * (tx - sx)
        vertices = [
            (sx, sy0),
            (sx + curve, sy0),
            (tx - curve, ty0),
            (tx, ty0),
            (tx, ty1),
            (tx - curve, ty1),
            (sx + curve, sy1),
            (sx, sy1),
            (sx, sy0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(
            PathPatch(
                MplPath(vertices, codes),
                facecolor=rgba(link.color, 0.55),
                edgecolor="none",
                zorder=1,
            )
        )

    for node in nodes:
        ax.add_patch(
            Rectangle(
                (node.x, node.y),
                node_width,
                node.height,
                facecolor=node.color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        )
        if node.label_side == "left":
            label_x = node.x - 0.008
            ha = "right"
        else:
            label_x = node.x + node_width + 0.008
            ha = "left"
        ax.text(
            label_x,
            node.y + node.height / 2,
            node.label,
            ha=ha,
            va="center",
            fontsize=11.5,
            color="#111827",
            linespacing=1.15,
            zorder=4,
        )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel_a() -> tuple[list[Node], list[Link], float]:
    scale = 0.0092
    nodes = [
        Node("all", "All harmful coalition\nproposals: 53", 0.01, 0.25, 53 * scale, COLORS["root"]),
        Node("g3", "Game 3\n2/752 (0.3%)", 0.19, 0.035, 2 * scale, COLORS["game3"]),
        Node("g2", "Game 2\n6/752 (0.8%)", 0.19, 0.105, 6 * scale, COLORS["game2"]),
        Node("g1", "Game 1\n45/940 (4.8%)", 0.19, 0.34, 45 * scale, COLORS["game1"]),
        Node("g3_no", "Not accepted: 1", 0.42, 0.010, 1 * scale, COLORS["rejected"]),
        Node("g3_yes", "Accepted: 1/2 (50.0%)", 0.42, 0.055, 1 * scale, COLORS["accepted"]),
        Node("g2_no", "Not accepted: 4", 0.42, 0.100, 4 * scale, COLORS["rejected"]),
        Node("g2_yes", "Accepted: 2/6 (33.3%)", 0.42, 0.165, 2 * scale, COLORS["accepted"]),
        Node(
            "adv",
            "Homogeneous adversary\n5/400 proposed (1.25%)",
            0.42,
            0.285,
            5 * scale,
            COLORS["adversary"],
        ),
        Node(
            "mono",
            "Random monoculture\n12/100 proposed (12.0%)",
            0.42,
            0.43,
            12 * scale,
            COLORS["monoculture"],
        ),
        Node(
            "hetero",
            "Heterogeneous\n28/400 proposed (7.0%)",
            0.42,
            0.67,
            28 * scale,
            COLORS["heterogeneous"],
        ),
        Node("adv_no", "Not accepted: 1", 0.74, 0.265, 1 * scale, COLORS["rejected"]),
        Node("adv_yes", "Accepted: 4/5 (80.0%)", 0.74, 0.31, 4 * scale, COLORS["accepted"]),
        Node("mono_no", "Not accepted: 2", 0.74, 0.415, 2 * scale, COLORS["rejected"]),
        Node("mono_yes", "Accepted: 10/12 (83.3%)", 0.74, 0.465, 10 * scale, COLORS["accepted"]),
        Node("hetero_no", "Not accepted: 12", 0.74, 0.64, 12 * scale, COLORS["rejected"]),
        Node("hetero_yes", "Accepted: 16/28 (57.1%)", 0.74, 0.79, 16 * scale, COLORS["accepted"]),
    ]
    links = [
        Link("all", "g3", 2, COLORS["game3"]),
        Link("all", "g2", 6, COLORS["game2"]),
        Link("all", "g1", 45, COLORS["game1"]),
        Link("g3", "g3_no", 1, COLORS["rejected"]),
        Link("g3", "g3_yes", 1, COLORS["game3"]),
        Link("g2", "g2_no", 4, COLORS["rejected"]),
        Link("g2", "g2_yes", 2, COLORS["game2"]),
        Link("g1", "adv", 5, COLORS["adversary"]),
        Link("g1", "mono", 12, COLORS["monoculture"]),
        Link("g1", "hetero", 28, COLORS["heterogeneous"]),
        Link("adv", "adv_no", 1, COLORS["rejected"]),
        Link("adv", "adv_yes", 4, COLORS["accepted"]),
        Link("mono", "mono_no", 2, COLORS["rejected"]),
        Link("mono", "mono_yes", 10, COLORS["accepted"]),
        Link("hetero", "hetero_no", 12, COLORS["rejected"]),
        Link("hetero", "hetero_yes", 16, COLORS["accepted"]),
    ]
    return nodes, links, scale


def panel_b() -> tuple[list[Node], list[Link], float]:
    scale = 0.0130
    nodes = [
        Node("g1_accepted", "Accepted Game 1\nproposals: 30", 0.01, 0.31, 30 * scale, COLORS["accepted"]),
        Node("a_family", "Homogeneous\nadversary: 4", 0.22, 0.08, 4 * scale, COLORS["adversary"]),
        Node("m_family", "Random\nmonoculture: 10", 0.22, 0.36, 10 * scale, COLORS["monoculture"]),
        Node("h_family", "Heterogeneous: 16", 0.22, 0.69, 16 * scale, COLORS["heterogeneous"]),
        Node("a_zero", "Literal zero: 2", 0.49, 0.045, 2 * scale, COLORS["literal_zero"]),
        Node("a_nonzero", "Harmful nonzero: 2", 0.49, 0.125, 2 * scale, COLORS["harmful_nonzero"]),
        Node("m_nonzero", "Harmful nonzero: 2", 0.49, 0.315, 2 * scale, COLORS["harmful_nonzero"]),
        Node("m_zero", "Literal zero: 8", 0.49, 0.405, 8 * scale, COLORS["literal_zero"]),
        Node("h_nonzero", "Harmful nonzero: 12", 0.49, 0.625, 12 * scale, COLORS["harmful_nonzero"]),
        Node("h_zero", "Literal zero: 4", 0.49, 0.835, 4 * scale, COLORS["literal_zero"]),
        Node("a_replace", "Outside accept(s)\nrequired: 3", 0.79, 0.045, 3 * scale, COLORS["replacement"]),
        Node("a_full", "All coalition members\naccepted: 1", 0.79, 0.13, 1 * scale, COLORS["full_support"]),
        Node("m_full", "All coalition members\naccepted: 10", 0.79, 0.38, 10 * scale, COLORS["full_support"]),
        Node("h_replace", "Outside accept(s)\nrequired: 12", 0.79, 0.61, 12 * scale, COLORS["replacement"]),
        Node("h_full", "All coalition members\naccepted: 4", 0.79, 0.835, 4 * scale, COLORS["full_support"]),
    ]
    links = [
        Link("g1_accepted", "a_family", 4, COLORS["adversary"]),
        Link("g1_accepted", "m_family", 10, COLORS["monoculture"]),
        Link("g1_accepted", "h_family", 16, COLORS["heterogeneous"]),
        Link("a_family", "a_zero", 2, COLORS["literal_zero"]),
        Link("a_family", "a_nonzero", 2, COLORS["harmful_nonzero"]),
        Link("m_family", "m_nonzero", 2, COLORS["harmful_nonzero"]),
        Link("m_family", "m_zero", 8, COLORS["literal_zero"]),
        Link("h_family", "h_nonzero", 12, COLORS["harmful_nonzero"]),
        Link("h_family", "h_zero", 4, COLORS["literal_zero"]),
        Link("a_zero", "a_replace", 2, COLORS["replacement"]),
        Link("a_nonzero", "a_replace", 1, COLORS["replacement"]),
        Link("a_nonzero", "a_full", 1, COLORS["full_support"]),
        Link("m_nonzero", "m_full", 2, COLORS["full_support"]),
        Link("m_zero", "m_full", 8, COLORS["full_support"]),
        Link("h_nonzero", "h_replace", 9, COLORS["replacement"]),
        Link("h_nonzero", "h_full", 3, COLORS["full_support"]),
        Link("h_zero", "h_replace", 3, COLORS["replacement"]),
        Link("h_zero", "h_full", 1, COLORS["full_support"]),
    ]
    return nodes, links, scale


def validate() -> None:
    game_counts = {"Game 1": (45, 30), "Game 2": (6, 2), "Game 3": (2, 1)}
    assert sum(proposed for proposed, _ in game_counts.values()) == 53
    assert sum(accepted for _, accepted in game_counts.values()) == 33

    families = {
        "heterogeneous": (400, 28, 16, 14, 4, 4),
        "monoculture": (100, 12, 10, 10, 8, 10),
        "homogeneous_adversary": (400, 5, 4, 2, 2, 1),
    }
    # Tuple fields: eligible, proposed, accepted, zero proposed, zero accepted,
    # accepted with full planned-coalition support.
    assert sum(row[1] for row in families.values()) == 45
    assert sum(row[2] for row in families.values()) == 30
    assert sum(row[3] for row in families.values()) == 26
    assert sum(row[4] for row in families.values()) == 14
    assert sum(row[5] for row in families.values()) == 15


def write_edge_csv(panel_links: dict[str, list[Link]]) -> None:
    with EDGE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "source", "target", "value"])
        writer.writeheader()
        for panel, links in panel_links.items():
            for link in links:
                writer.writerow(
                    {"panel": panel, "source": link.source, "target": link.target, "value": link.value}
                )


def main() -> None:
    validate()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 12.2))
    fig.patch.set_facecolor("white")

    nodes_a, links_a, scale_a = panel_a()
    draw_sankey(axes[0], nodes_a, links_a, scale_a)
    axes[0].text(
        0.0,
        1.025,
        "(a) Harmful coalition proposals and accepted outcomes",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    axes[0].add_patch(
        FancyBboxPatch(
            (0.39, 0.88),
            0.32,
            0.10,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            transform=axes[0].transAxes,
            facecolor="#F8FAFC",
            edgecolor="#94A3B8",
            linewidth=1.0,
            linestyle=(0, (3, 2)),
        )
    )
    axes[0].text(
        0.55,
        0.93,
        "Game 1 literal zero: 26/45 proposed; 14/30 accepted\nAll-Nano homogeneous control: 0/40 proposed",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        color="#334155",
    )

    nodes_b, links_b, scale_b = panel_b()
    draw_sankey(axes[1], nodes_b, links_b, scale_b)
    axes[1].text(
        0.0,
        1.025,
        "(b) Severity and voting support among the 30 accepted Game 1 proposals",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    axes[1].text(
        0.99,
        0.985,
        "14 literal zero  |  16 harmful nonzero  |  15 full coalition support  |  15 required outside accept(s)",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color="#334155",
    )

    footer = (
        "Accepted means the proposal became the selected agreement. Literal-zero classification applies only to Game 1 because Games 2 and 3 use different protocols.\n"
        "Game 1 proposer pattern: 44/45 had Elo ≥1400; all 12 monoculture proposals came from Gemini 3.1 Pro; all 5 homogeneous-adversary proposals came from the adversary model."
    )
    fig.text(0.02, 0.012, footer, ha="left", va="bottom", fontsize=11.3, color="#334155", linespacing=1.35)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.095, hspace=0.24)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_STEM.with_suffix(f".{suffix}"),
            dpi=350 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
    write_edge_csv({"a": links_a, "b": links_b})
    plt.close(fig)


if __name__ == "__main__":
    main()
