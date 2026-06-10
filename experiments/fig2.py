"""Reproduce Figure 2 of the paper (behavioral model) in NEST.

Builds the four panels on top of biodl.network.ClassifierNetwork (2c, 2d) and
biodl.microcircuit (2b dynamics):

  2a  two-column connectivity schematic
  2b  VIP/SST/PV/PYR population rates + calcium over train/inference (the gate)
  2c  classification accuracy vs epoch for device mismatch 0/10/20/30%
  2d  input->PYR weight matrices for columns A/B (init / trained / post-inference)

Honest status (see README in this repo): 2c reproduces the qualitative robustness
trend; 2b reproduces the disinhibition gate; 2d shows the expected structure for
the taught column but is noisy/asymmetric because the bare classifier columns lack
the motif's E/I stabilization (runaway potentiation). Every non-paper parameter is
a GUESS collected in biodl.network.Fig2Config.

Run:  python experiments/fig2.py            # all panels (2c is slow)
      python experiments/fig2.py 2c         # a single panel
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.network import ClassifierNetwork, Fig2Config  # noqa: E402

nest.set_verbosity("M_ERROR")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# the two classes: binary patterns over N inputs with an overlap region
N = 16
PAT_A = list(range(0, 10))    # inputs 0..9
PAT_B = list(range(6, 16))    # inputs 6..15  (overlap 6..9 = shared/uninformative)
MISMATCH = [0.0, 0.10, 0.20, 0.30]
EPOCHS = 8
SEEDS = 3                      # GUESS (paper averages several; downscaled)
T_PRESENT = 500.0
T_TEST = 300.0
N_TEST = 6                     # test presentations per class per epoch


def _new(mismatch, seed):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})
    cfg = Fig2Config(n_input=N, n_pyr=4, t_present=T_PRESENT, seed=seed)
    return ClassifierNetwork(cfg, mismatch=mismatch).build()


def _accuracy(net):
    correct = 0
    for _ in range(N_TEST):
        correct += net.predict(PAT_A, T_TEST) == "A"
        correct += net.predict(PAT_B, T_TEST) == "B"
    return correct / (2 * N_TEST)


# ----------------------------------------------------------------------- 2c
def panel_2c():
    print("[2c] accuracy vs epoch x device mismatch (this is the slow one)")
    acc = np.full((len(MISMATCH), SEEDS, EPOCHS + 1), np.nan)
    for mi, m in enumerate(MISMATCH):
        for s in range(SEEDS):
            net = _new(m, seed=s + 1)
            acc[mi, s, 0] = _accuracy(net)  # epoch 0 = chance-ish (random init)
            for ep in range(EPOCHS):
                net.present(PAT_A, "A", T_PRESENT)
                net.present(PAT_B, "B", T_PRESENT)
                acc[mi, s, ep + 1] = _accuracy(net)
            print(f"    mismatch={m:.0%} seed={s + 1}: final acc={acc[mi, s, -1]:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    epochs = np.arange(EPOCHS + 1)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(MISMATCH)))
    for mi, m in enumerate(MISMATCH):
        mean = np.nanmean(acc[mi], axis=0)
        sd = np.nanstd(acc[mi], axis=0)
        ax.plot(epochs, mean, "-o", color=colors[mi], lw=2, ms=4, label=f"{m:.0%} mismatch")
        ax.fill_between(epochs, mean - sd, mean + sd, color=colors[mi], alpha=0.15)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("classification accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Fig. 2c — accuracy vs epoch under device mismatch ({SEEDS} seeds, mean±sd)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig2c")


# ----------------------------------------------------------------------- 2d
def panel_2d():
    print("[2d] weight matrices (init / trained / post-inference)")
    net = _new(0.0, seed=1)
    snaps = {"random init": (net.weight_matrix("A").copy(), net.weight_matrix("B").copy())}
    for ep in range(EPOCHS):
        net.present(PAT_A, "A", T_PRESENT)
        net.present(PAT_B, "B", T_PRESENT)
    snaps["end of training"] = (net.weight_matrix("A").copy(), net.weight_matrix("B").copy())
    for _ in range(8):
        net.predict(PAT_A, T_TEST)
        net.predict(PAT_B, T_TEST)
    snaps["after inference"] = (net.weight_matrix("A").copy(), net.weight_matrix("B").copy())

    # ideal variability-free target: column responds to its own pattern's inputs
    ideal_A = np.array([15.0 if i in PAT_A else 0.0 for i in range(N)])[:, None]
    ideal_B = np.array([15.0 if i in PAT_B else 0.0 for i in range(N)])[:, None]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    cols = list(snaps.keys()) + ["ideal target"]
    for r, (label, idx) in enumerate((("A", 0), ("B", 1))):
        for cidx, cname in enumerate(cols):
            ax = axes[r, cidx]
            if cname == "ideal target":
                mat = ideal_A if idx == 0 else ideal_B
            else:
                mat = snaps[cname][idx]
            im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=15)
            if r == 0:
                ax.set_title(cname, fontsize=10)
            if cidx == 0:
                ax.set_ylabel(f"column {label}\ninput #", fontsize=9)
            ax.set_xticks([])
    fig.suptitle("Fig. 2d — input→PYR weight matrices  (rows = inputs, overlap 6–9; "
                 "honest: taught structure appears but bare columns are noisy)", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="weight w [0–15]")
    _save(fig, "fig2d", tight=False)


# ----------------------------------------------------------------------- 2b
def panel_2b():
    """Disinhibition dynamics from the single-column motif (the train/inference
    gate): attention ON -> VIP up, SST down, PYR up into the learning regime."""
    print("[2b] population rates + calcium over the attention gate")
    from biodl.microcircuit import Microcircuit

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})
    gate = {"teacher_pyr": (7, 6000), "sst_pyr": (7, 8000), "vip_sst": (7, 6000)}
    net = (Microcircuit(1, 1, 1, 1, weights=gate).build()
           .drive_profiles(np.array([0.0, 1000.0, 2000.0]), input=0.0, teacher=150.0,
                           cue=[0.0, 400.0, 0.0])
           .tonic(sst=200.0).attach_recorders())
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
    for key, c, lbl in (("pyr", "#C0392B", "PYR"), ("vip", "#3FAE3A", "VIP"),
                        ("sst", "#3FB7E8", "SST"), ("pv", "#2874A6", "PV")):
        x, y = binned(key)
        ax1.plot(x, y, color=c, lw=1.8, label=lbl)
    ax1.set_ylabel("rate (Hz)")
    ax1.legend(fontsize=8, ncol=4, loc="upper right")
    ax1.set_title("Fig. 2b — attention gate (shaded = attention ON): VIP↑ SST↓ PYR↑ into windows")
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
        pos = {"pyr": (x0, 0), "pv": (x0 + 1.1, -1), "sst": (x0 + 1.1, 1), "vip": (x0 + 0.2, 2.2)}
        for k, (x, y) in pos.items():
            ax.add_patch(plt.Circle((x, y), 0.34, color=colcol[k], ec="white", lw=1.5, zorder=3))
            ax.text(x, y, k.upper(), color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)
        edges = [("pyr", "pv", EXC), ("pyr", "sst", EXC), ("pv", "pyr", INH),
                 ("sst", "pyr", INH), ("vip", "sst", INH)]
        for s, d, c in edges:
            ax.annotate("", xy=pos[d], xytext=pos[s],
                        arrowprops=dict(arrowstyle="-|>" if c == EXC else "-[", color=c,
                                        lw=2, shrinkA=20, shrinkB=20, connectionstyle="arc3,rad=0.12"))
        ax.text(x0 + 0.4, -2.0, f"column {tag}", fontsize=11, fontweight="bold", ha="center")
        return pos

    posA = column(0.0, "A")
    posB = column(5.0, "B")
    # shared input -> both PYR (plastic); per-column teacher -> apical; attention -> both VIP
    for pos, tag in ((posA, "A"), (posB, "B")):
        ax.annotate("", xy=(pos["pyr"][0] - 0.3, pos["pyr"][1] - 0.3), xytext=(-1.5, -2.6),
                    arrowprops=dict(arrowstyle="-|>", color=DRV, lw=2, connectionstyle="arc3,rad=0.2"))
        ax.annotate("", xy=(pos["pyr"][0], pos["pyr"][1] + 0.4), xytext=(pos["pyr"][0], 3.3),
                    arrowprops=dict(arrowstyle="-|>", color=DRV, lw=2))
        ax.text(pos["pyr"][0], 3.45, f"teacher {tag}", fontsize=9, ha="center", color=DRV)
    ax.text(-1.6, -2.8, "shared input\n(plastic NMDA)", fontsize=9, ha="center", color=DRV)
    ax.annotate("", xy=(posA["vip"][0], posA["vip"][1] + 0.4), xytext=(4.0, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=DRV, lw=1.6, connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(posB["vip"][0], posB["vip"][1] + 0.4), xytext=(4.0, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=DRV, lw=1.6, connectionstyle="arc3,rad=-0.2"))
    ax.text(4.0, 3.75, "attention (both columns)", fontsize=9, ha="center", color=DRV)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=EXC, lw=2, marker=">", label="excitatory"),
                       Line2D([0], [0], color=INH, lw=2, marker="o", label="inhibitory"),
                       Line2D([0], [0], color=DRV, lw=2, marker=">", label="external drive")],
              loc="upper right", fontsize=8)
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


PANELS = {"2a": panel_2a, "2b": panel_2b, "2c": panel_2c, "2d": panel_2d}

if __name__ == "__main__":
    which = sys.argv[1:] or ["2a", "2b", "2d", "2c"]  # 2c last (slow)
    for key in which:
        PANELS[key.lower().lstrip("fig")]()
