"""Reproduce Figure 2 of the paper (behavioral model) in NEST.

Builds the four panels on top of biodl.network.ClassifierNetwork (2c, 2d) and
biodl.microcircuit (2b dynamics):

  2a  two-column connectivity schematic
  2b  VIP/SST/PV/PYR population rates + calcium over train/inference (the gate)
  2c  classification accuracy vs epoch for device mismatch 0/10/20/30%
  2d  input->PYR weight matrices for columns A/B (init / trained / post-inference)
  rates / 2e  trained classifier PYR readout rates during A/B inference inputs

Honest status (see FIG2.md in this repo): 2c reproduces the qualitative
robustness trend; 2b reproduces the disinhibition gate; 2d shows learned
class-specific blocks that remain stable under read-only inference. Every
non-paper parameter is a GUESS collected in biodl.network.Fig2Config.

Run:  python experiments/fig2.py            # all panels (2c is slow)
      python experiments/fig2.py 2c         # a single panel
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.network import ClassifierNetwork, Fig2Config  # noqa: E402
from biodl.sim import reset  # noqa: E402

nest.set_verbosity("M_ERROR")

FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)

# the two classes: binary patterns over N inputs with an overlap region
N = 16
PAT_A = list(range(0, 10))  # inputs 0..9
PAT_B = list(range(6, 16))  # inputs 6..15  (overlap 6..9 = shared/uninformative)
MISMATCH = [0.0, 0.10, 0.20, 0.30]
PANEL_2D_DEVICE_MISMATCH = 0.20
PANEL_2D_N = 32
PANEL_2D_N_PYR = 16
EPOCHS = 10  # learning needs ~10 epochs x 800ms to converge (0% -> ~0.90)
SEEDS = 2  # GUESS (paper averages several; downscaled for runtime)
T_PRESENT = 800.0
T_TEST = 350.0
N_TEST = 3  # test presentations per class per checkpoint
CHECKPOINTS = (0, 5, 10)  # epochs at which to measure accuracy (keeps 2c cheap)
RATE_SEQUENCE = (
    "A",
    "B",
    "off",
    "A",
    "B",
    "B",
    "off",
    "A",
    "B",
    "off",
    "A",
    "A",
    "B",
    "off",
    "B",
    "A",
)
RATE_BIN_MS = 100.0


def _new(mismatch, seed):
    reset(seed=seed)
    cfg = Fig2Config(n_input=N, n_pyr=4, t_present=T_PRESENT, seed=seed)
    return ClassifierNetwork(cfg, mismatch=mismatch).build()


def _panel_2d_patterns():
    pat_a = list(range(0, 20))
    pat_b = list(range(12, PANEL_2D_N))
    overlap = sorted(set(pat_a) & set(pat_b))
    unique_a = sorted(set(pat_a) - set(pat_b))
    unique_b = sorted(set(pat_b) - set(pat_a))
    return pat_a, pat_b, overlap, unique_a, unique_b


def _trained_panel_2d_classifier():
    pat_a, pat_b, *_ = _panel_2d_patterns()
    reset(seed=1)
    cfg = Fig2Config(
        n_input=PANEL_2D_N, n_pyr=PANEL_2D_N_PYR, t_present=T_PRESENT, seed=1
    )
    net = (
        ClassifierNetwork(cfg, mismatch=PANEL_2D_DEVICE_MISMATCH)
        .build()
        .randomize_input_weights(seed=1)
    )
    for _ in range(EPOCHS):
        net.present(pat_a, "A", T_PRESENT)
        net.present(pat_b, "B", T_PRESENT)
    return net


def _accuracy(net):
    correct = 0
    for _ in range(N_TEST):
        correct += net.predict(PAT_A, T_TEST) == "A"
        correct += net.predict(PAT_B, T_TEST) == "B"
    return correct / (2 * N_TEST)


# ----------------------------------------------------------------------- 2c
def panel_2c():
    print("[2c] accuracy vs epoch x device mismatch (this is the slow one)")
    acc = np.full((len(MISMATCH), SEEDS, len(CHECKPOINTS)), np.nan)
    for mi, m in enumerate(MISMATCH):
        for s in range(SEEDS):
            net = _new(m, seed=s + 1)
            ep = 0
            for ci, cp in enumerate(CHECKPOINTS):
                while ep < cp:  # train up to this checkpoint
                    net.present(PAT_A, "A", T_PRESENT)
                    net.present(PAT_B, "B", T_PRESENT)
                    ep += 1
                acc[mi, s, ci] = _accuracy(net)
            print(
                f"    mismatch={m:.0%} seed={s + 1}: final acc={acc[mi, s, -1]:.2f}",
                flush=True,
            )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    epochs = np.array(CHECKPOINTS)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(MISMATCH)))
    for mi, m in enumerate(MISMATCH):
        mean = np.nanmean(acc[mi], axis=0)
        sd = np.nanstd(acc[mi], axis=0)
        ax.plot(
            epochs, mean, "-o", color=colors[mi], lw=2, ms=4, label=f"{m:.0%} mismatch"
        )
        ax.fill_between(epochs, mean - sd, mean + sd, color=colors[mi], alpha=0.15)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("classification accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(
        f"Fig. 2c — accuracy vs epoch under device mismatch ({SEEDS} seeds, mean±sd)"
    )
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig2c")


# ----------------------------------------------------------------------- 2d
def panel_2d():
    print(
        f"[2d] weight matrices ({PANEL_2D_DEVICE_MISMATCH:.0%} device mismatch, random init)"
    )
    pat_a, pat_b, overlap, unique_a, unique_b = _panel_2d_patterns()

    # Rebuild once for the pre-training initial matrix, because the helper returns
    # the trained network used for Final/Test.
    reset(seed=1)
    cfg = Fig2Config(
        n_input=PANEL_2D_N, n_pyr=PANEL_2D_N_PYR, t_present=T_PRESENT, seed=1
    )
    init_net = (
        ClassifierNetwork(cfg, mismatch=PANEL_2D_DEVICE_MISMATCH)
        .build()
        .randomize_input_weights(seed=1)
    )
    init = np.hstack([init_net.weight_matrix("A"), init_net.weight_matrix("B")])

    net = _trained_panel_2d_classifier()

    def combined():
        return np.hstack([net.weight_matrix("A"), net.weight_matrix("B")])

    snaps = {"Initial": init, "Final": combined().copy()}
    for _ in range(8):
        net.predict(pat_a, T_TEST)
        net.predict(pat_b, T_TEST)
    snaps["Test"] = combined().copy()

    ideal = np.zeros((PANEL_2D_N, 2 * PANEL_2D_N_PYR))
    ideal[np.ix_(unique_a, range(PANEL_2D_N_PYR))] = 15.0
    ideal[np.ix_(overlap, range(PANEL_2D_N_PYR))] = 7.5
    ideal[np.ix_(overlap, range(PANEL_2D_N_PYR, 2 * PANEL_2D_N_PYR))] = 7.5
    ideal[np.ix_(unique_b, range(PANEL_2D_N_PYR, 2 * PANEL_2D_N_PYR))] = 15.0
    snaps["Ideal"] = ideal

    input_mat = np.column_stack(
        [
            np.isin(np.arange(PANEL_2D_N), pat_a),
            np.isin(np.arange(PANEL_2D_N), pat_b),
        ]
    ).astype(float)

    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(1, 6, width_ratios=[0.28, 1, 1, 1, 1, 0.08], wspace=0.28)
    ax_in = fig.add_subplot(gs[0, 0])
    ax_in.imshow(input_mat, aspect="auto", cmap="gray", vmin=0, vmax=1)
    ax_in.set_title("Input", fontsize=10, pad=4)
    ax_in.set_xticks([0, 1], ["A", "B"], fontsize=8)
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_linewidth(1.2)

    axes = [fig.add_subplot(gs[0, i]) for i in range(1, 5)]
    im = None
    for ax, (title, mat) in zip(axes, snaps.items()):
        im = ax.imshow(
            mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=15, interpolation="nearest"
        )
        ax.axvline(PANEL_2D_N_PYR - 0.5, color="black", lw=1.2)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xticks(
            [PANEL_2D_N_PYR / 2 - 0.5, 1.5 * PANEL_2D_N_PYR - 0.5],
            ["A", "B"],
            fontsize=8,
        )
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    cax = fig.add_subplot(gs[0, 5])
    fig.colorbar(im, cax=cax, label="w [0–15]")
    overlap_label = f"{overlap[0]}–{overlap[-1]}" if overlap else "none"
    fig.suptitle(
        f"Fig. 2d — learned input→PYR weights ({PANEL_2D_DEVICE_MISMATCH:.0%} device mismatch; "
        f"class overlap rows {overlap_label})",
        fontsize=11,
    )
    _save(fig, "fig2d", tight=False)


# ------------------------------------------------------------------- rates
def panel_rates():
    print("[rates] trained classifier PYR rates during read-only A/B inputs")
    pat_a, pat_b, overlap, *_ = _panel_2d_patterns()
    patterns = {"A": pat_a, "B": pat_b, "off": []}
    net = _trained_panel_2d_classifier()

    t_present = T_PRESENT
    n_bins = int(round(t_present / RATE_BIN_MS))
    trace_t = []
    trace = {"A": [], "B": []}
    windows = []
    t_offset = 0.0

    net.set_teacher(None)
    net.set_attention(False)
    net._set_input_plasticity(False)
    try:
        for label in RATE_SEQUENCE:
            for col in net.COLUMNS:
                net.sr[col].set({"n_events": 0})
            t0 = float(nest.GetKernelStatus("biological_time"))
            net.set_pattern(patterns[label])
            nest.Simulate(t_present)

            edges_abs = t0 + np.arange(n_bins + 1) * RATE_BIN_MS
            centers_rel = t_offset + (np.arange(n_bins) + 0.5) * RATE_BIN_MS
            trace_t.extend((centers_rel / 1e3).tolist())
            for col in net.COLUMNS:
                times = np.asarray(net.sr[col].get("events")["times"], dtype=float)
                counts, _ = np.histogram(times, bins=edges_abs)
                rates = counts * 1e3 / (RATE_BIN_MS * len(net.pyr[col]))
                trace[col].extend(rates.tolist())
            windows.append((label, t_offset / 1e3, (t_offset + t_present) / 1e3))
            t_offset += t_present
    finally:
        net._set_input_plasticity(True)

    t = np.asarray(trace_t)
    rate_a = np.asarray(trace["A"])
    rate_b = np.asarray(trace["B"])
    diff = rate_a - rate_b

    fig = plt.figure(figsize=(15.5, 5.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.45, 2.3, 1.1], hspace=0.12)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_rate = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_diff = fig.add_subplot(gs[2, 0], sharex=ax_top)

    class_color = {"A": "#6C3483", "B": "#2874A6", "off": "#7F8C8D"}
    display_label = {"A": "A", "B": "B", "off": "none"}
    for label, start, stop in windows:
        for ax in (ax_rate, ax_diff):
            ax.axvspan(start, stop, color=class_color[label], alpha=0.08, lw=0)
            ax.axvline(start, color="0.75", lw=0.8)
        rect = patches.Rectangle(
            (start, 0.12),
            stop - start,
            0.76,
            facecolor={"A": "0.72", "B": "0.48", "off": "0.90"}[label],
            edgecolor="0.35",
            lw=0.8,
        )
        ax_top.add_patch(rect)
        ax_top.text(
            0.5 * (start + stop),
            0.5,
            display_label[label],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )
    ax_rate.axvline(windows[-1][2], color="0.75", lw=0.8)
    ax_diff.axvline(windows[-1][2], color="0.75", lw=0.8)

    ax_rate.plot(t, rate_a, color="#8E44AD", lw=2.0, label="PYR column A")
    ax_rate.plot(t, rate_b, color="#1F77B4", lw=2.0, label="PYR column B")
    ax_rate.set_ylabel("PYR rate (Hz)")
    ax_rate.legend(loc="upper right", fontsize=8)

    ax_diff.axhline(0, color="0.25", lw=1.0)
    ax_diff.fill_between(
        t, 0, diff, where=diff >= 0, color="#8E44AD", alpha=0.18, interpolate=True
    )
    ax_diff.fill_between(
        t, 0, diff, where=diff < 0, color="#1F77B4", alpha=0.18, interpolate=True
    )
    ax_diff.plot(t, diff, color="black", lw=1.2)
    ax_diff.set_ylabel("A - B\nrate (Hz)")
    ax_diff.set_xlabel("time during read-only inference (s)")

    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_ylabel("input", rotation=0, labelpad=24, va="center")
    ax_top.tick_params(axis="x", labelbottom=False)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    ax_rate.set_xlim(0, t_offset / 1e3)
    overlap_label = f"{overlap[0]}–{overlap[-1]}" if overlap else "none"
    fig.suptitle(
        f"Fig. 2 readout rates — higher PYR population rate encodes class "
        f"({PANEL_2D_DEVICE_MISMATCH:.0%} device mismatch; overlap rows {overlap_label})",
        fontsize=11,
    )
    _save(fig, "fig2_rates")


# ----------------------------------------------------------------------- 2b
def panel_2b():
    """Disinhibition dynamics from the single-column motif (the train/inference
    gate): attention ON -> VIP up, SST down, PYR up into the learning regime."""
    print("[2b] population rates + calcium over the attention gate")
    from biodl.microcircuit import Microcircuit

    reset(seed=1)
    gate = {"teacher_pyr": (7, 6000), "sst_pyr": (7, 8000), "vip_sst": (7, 6000)}
    # the "training" window = attention + teacher together (both gated to 1-2 s),
    # so baseline (0-1 s, 2-3 s) has no drive -> PYR ~silent -> theta ~0.
    net = (
        Microcircuit(1, 1, 1, 1, weights=gate)
        .build()
        .drive_profiles(
            np.array([0.0, 1000.0, 2000.0]),
            input=0.0,
            teacher=[0.0, 150.0, 0.0],
            cue=[0.0, 400.0, 0.0],
        )
        .tonic(sst=200.0)
        .attach_recorders()
    )
    net.pop["pyr"].set({"Iapical_low": 0.0})
    nest.Simulate(3000.0)

    def binned(key, bin_ms=100.0):
        t = net.spikes(key)["times"]
        edges = np.arange(0, 3000 + bin_ms, bin_ms)
        c, _ = np.histogram(t, bins=edges)
        return 0.5 * (edges[:-1] + edges[1:]) / 1e3, c * 1e3 / bin_ms

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(1.0, 2.0, color="#F9E79F", alpha=0.5)
    for key, c, lbl in (
        ("pyr", "#C0392B", "PYR"),
        ("vip", "#3FAE3A", "VIP"),
        ("sst", "#3FB7E8", "SST"),
        ("pv", "#2874A6", "PV"),
    ):
        x, y = binned(key)
        ax1.plot(x, y, color=c, lw=1.8, label=lbl)
    ax1.set_ylabel("rate (Hz)")
    ax1.legend(fontsize=8, ncol=4, loc="upper right")
    ax1.set_title(
        "Fig. 2b — attention gate (shaded = attention ON): VIP↑ SST↓ PYR↑ into windows"
    )
    # calcium proxy θ: exponential low-pass (tau_ca) of the PYR spike train (GUESS)
    tau_ca, dt_b = 300.0, 20.0
    bins = np.arange(0, 3000 + dt_b, dt_b)
    cnt, _ = np.histogram(net.spikes("pyr")["times"], bins=bins)
    theta = np.zeros(len(cnt))
    a = np.exp(-dt_b / tau_ca)
    for i in range(1, len(cnt)):
        theta[i] = a * theta[i - 1] + cnt[i]
    ax2.plot(0.5 * (bins[:-1] + bins[1:]) / 1e3, theta, color="black", lw=1.2)
    ax2.set_ylabel("PYR calcium θ\n(low-pass of spikes)")
    ax2.set_xlabel("time (s)")
    _save(fig, "fig2b")


# ----------------------------------------------------------------------- 2a
def panel_2a():
    """Two-column connectivity schematic (programmatic)."""
    print("[2a] two-column connectivity schematic")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    EXC, INH, DRV = "#C0392B", "#2065A8", "#9B1B1B"
    colcol = {"pyr": "#C0392B", "pv": "#2874A6", "sst": "#3FB7E8", "vip": "#3FAE3A"}

    def column(x0, tag):
        pos = {
            "pyr": (x0, 0),
            "pv": (x0 + 1.1, -1),
            "sst": (x0 + 1.1, 1),
            "vip": (x0 + 0.2, 2.2),
        }
        for k, (x, y) in pos.items():
            ax.add_patch(
                plt.Circle((x, y), 0.34, color=colcol[k], ec="white", lw=1.5, zorder=3)
            )
            ax.text(
                x,
                y,
                k.upper(),
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )
        edges = [
            ("pyr", "pv", EXC),
            ("pyr", "sst", EXC),
            ("pv", "pyr", INH),
            ("sst", "pyr", INH),
            ("vip", "sst", INH),
        ]
        for s, d, c in edges:
            ax.annotate(
                "",
                xy=pos[d],
                xytext=pos[s],
                arrowprops=dict(
                    arrowstyle="-|>" if c == EXC else "-[",
                    color=c,
                    lw=2,
                    shrinkA=20,
                    shrinkB=20,
                    connectionstyle="arc3,rad=0.12",
                ),
            )
        ax.text(
            x0 + 0.4, -2.0, f"column {tag}", fontsize=11, fontweight="bold", ha="center"
        )
        return pos

    posA = column(0.0, "A")
    posB = column(5.0, "B")
    # shared input -> both PYR (plastic); per-column teacher -> apical; attention -> both VIP
    for pos, tag in ((posA, "A"), (posB, "B")):
        ax.annotate(
            "",
            xy=(pos["pyr"][0] - 0.3, pos["pyr"][1] - 0.3),
            xytext=(-1.5, -2.6),
            arrowprops=dict(
                arrowstyle="-|>", color=DRV, lw=2, connectionstyle="arc3,rad=0.2"
            ),
        )
        ax.annotate(
            "",
            xy=(pos["pyr"][0], pos["pyr"][1] + 0.4),
            xytext=(pos["pyr"][0], 3.3),
            arrowprops=dict(arrowstyle="-|>", color=DRV, lw=2),
        )
        ax.text(
            pos["pyr"][0], 3.45, f"teacher {tag}", fontsize=9, ha="center", color=DRV
        )
    ax.text(
        -1.6, -2.8, "shared input\n(plastic NMDA)", fontsize=9, ha="center", color=DRV
    )
    ax.annotate(
        "",
        xy=(posA["vip"][0], posA["vip"][1] + 0.4),
        xytext=(4.0, 3.6),
        arrowprops=dict(
            arrowstyle="-|>", color=DRV, lw=1.6, connectionstyle="arc3,rad=0.2"
        ),
    )
    ax.annotate(
        "",
        xy=(posB["vip"][0], posB["vip"][1] + 0.4),
        xytext=(4.0, 3.6),
        arrowprops=dict(
            arrowstyle="-|>", color=DRV, lw=1.6, connectionstyle="arc3,rad=-0.2"
        ),
    )
    ax.text(4.0, 3.75, "attention (both columns)", fontsize=9, ha="center", color=DRV)

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color=EXC, lw=2, marker=">", label="excitatory"),
            Line2D([0], [0], color=INH, lw=2, marker="o", label="inhibitory"),
            Line2D([0], [0], color=DRV, lw=2, marker=">", label="external drive"),
        ],
        loc="upper right",
        fontsize=8,
    )
    ax.set_xlim(-2.5, 8.0)
    ax.set_ylim(-3.2, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fig. 2a — two-column classifier connectivity")
    _save(fig, "fig2a")


def _save(fig, name, tight=True):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print("    saved", out)


PANELS = {
    "2a": panel_2a,
    "2b": panel_2b,
    "2c": panel_2c,
    "2d": panel_2d,
    "2e": panel_rates,
    "rates": panel_rates,
}

if __name__ == "__main__":
    which = sys.argv[1:] or ["2a", "2b", "2d", "2c"]  # 2c last (slow)
    for key in which:
        PANELS[key.lower().lstrip("fig")]()
