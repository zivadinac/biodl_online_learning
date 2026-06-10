"""Draw the canonical-microcircuit connectivity in the style of the paper's Fig. 1a.

PYR is drawn as a pyramidal cell (grey soma triangle + apical dendrite shaft and
tuft); the interneurons are coloured circles (VIP green, SST cyan, PV pink).
Excitatory connections end in an arrowhead, inhibitory ones in a filled ball, and
the external drives (Teacher/prediction, Context/attention, Input) are dark red,
exactly as in the paper. Every drawn connection is checked against the real edge
data in ``biodl.microcircuit`` so the schematic cannot drift from the network.

Run:  python experiments/connectivity_graph.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.microcircuit import DRIVE_EDGES, MOTIF_EDGES, RECURRENT_EDGES  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# paper Fig. 1a palette
GREY = "#8C8C8C"      # PYR soma / dendrites
VIP_C = "#3FAE3A"     # green
SST_C = "#3FB7E8"     # cyan
PV_C = "#EC5BA8"      # pink
DRIVE_C = "#9B1B1B"   # dark red (teacher / context / input)
EXC_GREY = "#A8A8A8"  # PYR excitatory projections

# interneuron centres and radius
VIP_P, SST_P, PV_P = (2.6, 2.5), (2.6, 0.8), (2.6, -1.1)
R = 0.42

# PYR pyramidal-cell geometry
APEX = (0.0, 0.35)
BASE_L, BASE_R = (-0.75, -1.35), (0.75, -1.35)
TUFT_Y = 2.75
SHAFT_TOP = (0.0, TUFT_Y)


def _build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for src, dst, _w, receptor, sign in MOTIF_EDGES + RECURRENT_EDGES + DRIVE_EDGES:
        g.add_edge(src, dst, receptor=receptor, sign=sign)
    return g


def _check(g: nx.DiGraph, src: str, dst: str) -> None:
    if not g.has_edge(src, dst):
        raise AssertionError(f"drawn edge {src}->{dst} not in the network's edge data")


def exc(ax, p0, p1, color=EXC_GREY, lw=2.4, rad=0.0):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=20,
                                shrinkA=6, shrinkB=6, connectionstyle=f"arc3,rad={rad}"))


def inh(ax, p0, p1, color, lw=2.4, rad=0.0, ball=0.11):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-", color=color, lw=lw,
                                shrinkA=6, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
    ax.add_patch(Circle(p1, ball, color=color, zorder=6))


def draw_pyramidal(ax) -> None:
    # soma (filled triangle, apex up)
    ax.add_patch(Polygon([APEX, BASE_L, BASE_R], closed=True, facecolor=GREY,
                         edgecolor="#6E6E6E", lw=1.5, zorder=4))
    # apical dendrite shaft
    ax.plot([APEX[0], SHAFT_TOP[0]], [APEX[1], SHAFT_TOP[1]], color=GREY, lw=2.5, zorder=2)
    # apical tuft (fork)
    for dx in (-0.4, -0.15, 0.15, 0.4):
        ax.plot([SHAFT_TOP[0], dx], [SHAFT_TOP[1], TUFT_Y + 0.5], color=GREY, lw=1.6, zorder=2)
    # basal dendrites
    for (bx, by), dx in ((BASE_L, -0.35), (BASE_R, 0.35)):
        ax.plot([bx, bx + dx], [by, by - 0.35], color=GREY, lw=1.6, zorder=2)
    ax.text(0.0, -0.55, "PYR", fontsize=12, color="white", fontweight="bold",
            ha="center", va="center", zorder=5)


def main() -> None:
    g = _build_graph()
    fig, ax = plt.subplots(figsize=(8.5, 9.5))

    draw_pyramidal(ax)
    for centre, color, label in ((VIP_P, VIP_C, "VIP"), (SST_P, SST_C, "SST"), (PV_P, PV_C, "PV")):
        ax.add_patch(Circle(centre, R, facecolor=color, edgecolor="white", lw=2, zorder=4))
        ax.text(*centre, label, fontsize=12, color="white", fontweight="bold",
                ha="center", va="center", zorder=5)

    # --- external drives (dark red, arrowheads) ---
    ax.text(-1.1, 3.55, "Teacher/prediction", fontsize=12, ha="center", color="black")
    _check(g, "teacher", "pyr")
    ax.annotate("", xy=(0.0, TUFT_Y + 0.45), xytext=(-1.1, 3.35),
                arrowprops=dict(arrowstyle="-|>", color=DRIVE_C, lw=2.6, mutation_scale=22,
                                connectionstyle="arc3,rad=-0.3", shrinkB=2))
    ax.text(2.6, 3.75, "Context/attention", fontsize=12, ha="center", color="black")
    _check(g, "cue", "vip")
    ax.annotate("", xy=(VIP_P[0], VIP_P[1] + R), xytext=(2.7, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=DRIVE_C, lw=2.6, mutation_scale=22,
                                connectionstyle="arc3,rad=-0.3", shrinkB=2))

    # input rail along the bottom -> PYR basal (plastic) and -> PV
    ax.text(-1.85, -2.65, "Input", fontsize=12, ha="left", color="black")
    _check(g, "input", "pyr")
    ax.annotate("", xy=(BASE_L[0] - 0.05, BASE_L[1] - 0.25), xytext=(-1.3, -2.55),
                arrowprops=dict(arrowstyle="-|>", color=DRIVE_C, lw=2.6, mutation_scale=20,
                                connectionstyle="arc3,rad=0.4", shrinkB=2))
    _check(g, "input", "pv")
    ax.annotate("", xy=(PV_P[0], PV_P[1] - R), xytext=(-1.0, -2.75),
                arrowprops=dict(arrowstyle="-|>", color=DRIVE_C, lw=2.6, mutation_scale=20,
                                connectionstyle="arc3,rad=-0.35", shrinkB=2))
    # "Plastic weights" annotation on the basal dendrite
    ax.annotate("Plastic\nweights", xy=(BASE_L[0] - 0.18, BASE_L[1] - 0.18),
                xytext=(-2.05, -1.35), fontsize=10.5, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))

    # --- inhibitory motif (filled balls) ---
    _check(g, "vip", "sst")
    inh(ax, (VIP_P[0], VIP_P[1] - R), (SST_P[0], SST_P[1] + R), VIP_C)        # VIP -| SST
    _check(g, "sst", "pyr")
    inh(ax, (SST_P[0] - R, SST_P[1]), (0.08, 0.95), SST_C)                    # SST -| PYR apical (gate)
    _check(g, "pv", "pyr")
    inh(ax, (PV_P[0] - R, PV_P[1]), (0.55, -1.05), PV_C, rad=0.15)            # PV -| PYR
    _check(g, "pv", "pv")
    inh(ax, (PV_P[0] + 0.15, PV_P[1] + R - 0.05), (PV_P[0] + R - 0.02, PV_P[1] + 0.18),
        PV_C, lw=2.0, rad=-2.8, ball=0.09)                                    # PV -| PV (self)

    # --- excitatory PYR projections (grey arrowheads) ---
    _check(g, "pyr", "sst")
    exc(ax, (0.6, -0.6), (SST_P[0] - R - 0.05, SST_P[1] - 0.25), rad=-0.35)   # PYR -> SST
    _check(g, "pyr", "pv")
    exc(ax, (0.7, -1.15), (PV_P[0] - R - 0.05, PV_P[1] - 0.05), rad=-0.25)    # PYR -> PV

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=DRIVE_C, lw=2.6, marker=">", markersize=7, label="excitatory drive"),
        Line2D([0], [0], color=EXC_GREY, lw=2.4, marker=">", markersize=7, label="excitatory (PYR)"),
        Line2D([0], [0], color="#555", lw=2.4, marker="o", markersize=8, label="inhibitory"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)

    ax.set_xlim(-2.6, 3.6)
    ax.set_ylim(-3.0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Canonical cortical microcircuit (after Fig. 1a)", fontsize=13)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "connectivity.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
