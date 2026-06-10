"""Milestone C: the local three-factor delta-rule learning (paper Fig. 3c).

One PYR neuron with three plastic NMDA basal input synapses W1, W2, W3. Two input
patterns are presented alternately:

  * pattern A drives W1 and W2, *with* the apical teacher  -> PYR in the LTP region
  * pattern B drives W2 and W3, *without* the teacher      -> PYR in the LTD region

So W1 (only ever in A, taught) potentiates, W3 (only in B, untaught) depresses,
and W2 (in both) is driven to an intermediate value -- the uninformative shared
synapse, exactly as in Fig. 3c / Fig. 2d.

Learning needs the calcium proxy in the LTP/LTD windows: that requires
``effective_bias=True`` (which Chiara's config omits) together with the config's
own ``k_ca_th_*`` windows, and a strong apical teacher. Plastic synapses are
4-bit (w in [0,15]); the static teacher is 3-bit.

Run:  python experiments/03_learning.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.nest_setup import NEURON_MODEL, SYNAPSE_MODEL  # noqa: E402
from biodl.params import neuron_params  # noqa: E402
from biodl.sim import reset  # noqa: E402

nest.set_verbosity("M_ERROR")

N_ITER = 10  # A/B presentation pairs
T_PHASE = 1000.0  # ms per pattern presentation
IN_RATE = 10.0  # Hz, an active input channel
TEACH_RATE = 200.0  # Hz, apical teacher during pattern A
W0 = 7  # initial 4-bit weight [0,15]
IBIAS = 100.0
ETA = 0.2  # learning rate
BINARIZE = False  # graded 4-bit weights (paper Fig 2d/3c); True = binary 0/15

FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)
WCOL = {"W1": "#1E8449", "W2": "#B7950B", "W3": "#C0392B"}
NAMES = ["W1", "W2", "W3"]
PATTERN = {"A": (0, 1), "B": (1, 2)}  # which channels (0,1,2 == W1,W2,W3) are active
A_SHADE, B_SHADE = "#FCF3CF", "#EAF2F8"


def main() -> None:
    reset()

    pyr = nest.Create(NEURON_MODEL, 1)
    p = neuron_params("pyr")
    p["effective_bias"] = (
        True  # required so Isoma_ca reaches the config k_ca_th windows
    )
    pyr.set(p)
    rt = pyr[0].get("receptor_types")
    win = {
        k: pyr[0].get(k)
        for k in (
            "k_ca_th_L_minus",
            "k_ca_th_L_plus",
            "k_ca_th_H_minus",
            "k_ca_th_H_plus",
        )
    }

    wr = nest.Create("weight_recorder")
    nest.CopyModel(
        SYNAPSE_MODEL,
        "plastic",
        {
            "plastic": True,
            "binarize": BINARIZE,
            "n_bit": 4,
            "eta": ETA,
            "eta_L": ETA,
            "delay": 0.1,
            "weight_recorder": wr,
            "receptor_type": int(rt["NMDA_BASAL_SPIKES"]),
        },
    )
    nest.CopyModel(
        SYNAPSE_MODEL,
        "static",
        {
            "plastic": False,
            "binarize": True,
            "n_bit": 3,
            "eta": 0.0,
            "eta_L": 0.0,
            "delay": 0.1,
        },
    )

    # three input channels: Poisson -> parrot -> plastic NMDA synapse
    gens = nest.Create("poisson_generator", 3)
    parrots = nest.Create("parrot_neuron", 3)
    nest.Connect(gens, parrots, "one_to_one", {"weight": 1.0, "delay": 0.1})
    for par in parrots:
        nest.Connect(
            par,
            pyr,
            "one_to_one",
            {"synapse_model": "plastic", "w": W0, "Ibias": IBIAS},
        )
    parrot_id = [int(p_.get("global_id")) for p_ in parrots]

    teacher = nest.Create("poisson_generator", 1)
    nest.Connect(
        teacher,
        pyr,
        "one_to_one",
        {
            "synapse_model": "static",
            "receptor_type": int(rt["AMPA_APICAL_SPIKES"]),
            "w": 7,
            "Ibias": IBIAS,
        },
    )

    mm = nest.Create("multimeter", {"record_from": ["Isoma_ca"]})
    nest.Connect(mm, pyr)

    schedule = []  # (t_start, t_stop, pattern)
    t = 0.0
    for _ in range(N_ITER):
        for pat in ("A", "B"):
            for ch in range(3):
                gens[ch].set({"rate": IN_RATE if ch in PATTERN[pat] else 0.0})
            teacher.set({"rate": TEACH_RATE if pat == "A" else 0.0})
            nest.Simulate(T_PHASE)
            schedule.append((t, t + T_PHASE, pat))
            t += T_PHASE

    # ---- data ----
    ev = wr.get("events")
    wt = np.asarray(ev["times"])
    wsnd = np.asarray(ev["senders"])
    wval = np.asarray(ev["weights"]) / IBIAS  # weight_recorder stores w*Ibias
    tr = mm.get("events")
    t_ca, ca = np.asarray(tr["times"]) / 1e3, np.asarray(tr["Isoma_ca"])
    t_end = N_ITER * 2 * T_PHASE
    half = N_ITER * T_PHASE

    def time_avg(ch, t_from):
        """Time-weighted average weight of channel ch over [t_from, t_end] ms."""
        m = wsnd == parrot_id[ch]
        ts = np.concatenate([[0.0], wt[m], [t_end]])
        vs = np.concatenate([[W0], wval[m], [wval[m][-1] if m.any() else W0]])
        seg = np.clip(ts[1:], t_from, None) - np.clip(ts[:-1], t_from, None)
        return float(np.sum(seg * vs[:-1]) / max(np.sum(seg), 1e-9))

    means = [time_avg(ch, half) for ch in range(3)]

    # ---- figure ----
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[4.3, 1.0],
        height_ratios=[0.72, 1.5, 1.2],
        hspace=0.16,
        wspace=0.18,
    )
    ax_stim = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[1, 0], sharex=ax_stim)
    ax_ca = fig.add_subplot(gs[2, 0], sharex=ax_stim)
    ax_bar = fig.add_subplot(gs[1:, 1])

    # --- stimulus schedule (which inputs + teacher are active each phase) ---
    yrow = {"Teacher": 3, "W1": 2, "W2": 1, "W3": 0}
    rcol = {"Teacher": "#9B1B1B", **WCOL}
    for t0, t1, pat in schedule:
        items = {NAMES[c] for c in PATTERN[pat]} | (
            {"Teacher"} if pat == "A" else set()
        )
        for it in items:
            ax_stim.broken_barh(
                [(t0 / 1e3, (t1 - t0) / 1e3)],
                (yrow[it] - 0.38, 0.76),
                facecolors=rcol[it],
            )
    ax_stim.set_yticks(list(yrow.values()))
    ax_stim.set_yticklabels(list(yrow.keys()), fontsize=9)
    ax_stim.set_ylim(-0.6, 3.6)
    ax_stim.tick_params(labelbottom=False)
    ax_stim.set_title(
        "Local three-factor delta-rule learning  —  one PYR, 3 plastic inputs (Fig. 3c)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # --- weight trajectories (raw faint, 2nd-half mean bold-dotted) ---
    for ch, name in enumerate(NAMES):
        m = wsnd == parrot_id[ch]
        ax_w.step(
            np.concatenate([[0], wt[m]]) / 1e3,
            np.concatenate([[W0], wval[m]]),
            where="post",
            color=WCOL[name],
            lw=1.1,
            alpha=0.45,
        )
        ax_w.axhline(
            means[ch],
            color=WCOL[name],
            ls=(0, (1, 1)),
            lw=2.0,
            label=f"{name} (mean {means[ch]:.1f})",
        )
    ax_w.set_ylabel("plastic weight  w ∈ [0,15]")
    ax_w.set_ylim(-0.6, 15.6)
    ax_w.set_yticks([0, 5, 10, 15])
    ax_w.tick_params(labelbottom=False)
    ax_w.legend(loc="center left", fontsize=8.5, framealpha=0.9)
    ax_w.text(
        0.5,
        1.03,
        "W1 potentiates · W3 depresses · W2 (shared) → intermediate",
        transform=ax_w.transAxes,
        ha="center",
        fontsize=9.5,
        color="#333",
    )

    # --- calcium proxy with shaded LTP/LTD windows ---
    ax_ca.axhspan(
        win["k_ca_th_L_minus"], win["k_ca_th_H_minus"], color="#2980B9", alpha=0.10
    )
    ax_ca.axhspan(
        win["k_ca_th_L_plus"], win["k_ca_th_H_plus"], color="#E74C3C", alpha=0.14
    )
    ax_ca.plot(t_ca, ca, color="black", lw=0.8)
    ax_ca.text(
        t_end / 1e3,
        (win["k_ca_th_L_plus"] + win["k_ca_th_H_plus"]) / 2,
        "LTP ",
        va="center",
        ha="right",
        fontsize=8,
        color="#C0392B",
        fontweight="bold",
    )
    ax_ca.text(
        t_end / 1e3,
        win["k_ca_th_L_minus"],
        "LTD ",
        va="top",
        ha="right",
        fontsize=8,
        color="#2471A3",
        fontweight="bold",
    )
    ax_ca.set_ylabel("Isoma_ca (pA)\ncalcium proxy")
    ax_ca.set_xlabel("time (s)")
    ax_ca.set_xlim(0, t_end / 1e3)

    # --- learned-weights bar summary ---
    yb = np.arange(3)[::-1]
    ax_bar.barh(yb, means, color=[WCOL[n] for n in NAMES], height=0.62)
    ax_bar.axvline(7.5, color="grey", ls="--", lw=0.7)
    ax_bar.set_yticks(yb)
    ax_bar.set_yticklabels(NAMES, fontsize=9)
    ax_bar.set_xlim(0, 15.8)
    ax_bar.set_xticks([0, 5, 10, 15])
    ax_bar.set_xlabel("mean w (2nd half)", fontsize=9)
    ax_bar.set_title("learned weights", fontsize=10)
    for y, v in zip(yb, means):
        ax_bar.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "03_learning.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(
        f"2nd-half mean weights:  W1={means[0]:.1f}  W2={means[1]:.1f}  W3={means[2]:.1f}  "
        f"(start {W0})  -> W1 high, W3 low, W2 intermediate"
    )
    print("saved", out)


if __name__ == "__main__":
    main()
