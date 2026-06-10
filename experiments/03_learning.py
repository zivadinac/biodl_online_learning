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
own ``k_ca_th_*`` windows, and a strong apical teacher.

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
from biodl.nest_setup import NEURON_MODEL, SYNAPSE_MODEL, install_dynaple  # noqa: E402
from biodl.params import neuron_params  # noqa: E402

nest.set_verbosity("M_ERROR")

N_ITER = 10          # A/B presentation pairs
T_PHASE = 1000.0     # ms per pattern presentation
IN_RATE = 10.0       # Hz, an active input channel
TEACH_RATE = 200.0   # Hz, apical teacher during pattern A
W0 = 7               # initial 4-bit weight [0,15]
IBIAS = 100.0
ETA = 0.2            # learning rate (smaller, so graded weights can settle mid)
BINARIZE = False     # graded 4-bit weights (paper Fig 2d/3c); True = binary 0/15

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
WCOL = {"W1": "#1E8449", "W2": "#B7950B", "W3": "#C0392B"}
# pattern -> which channels are active (channel index 0,1,2 == W1,W2,W3)
PATTERN = {"A": (0, 1), "B": (1, 2)}


def main() -> None:
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})
    install_dynaple()

    pyr = nest.Create(NEURON_MODEL, 1)
    p = neuron_params("pyr")
    p["effective_bias"] = True  # required so Isoma_ca reaches the config k_ca_th windows
    pyr.set(p)
    rt = pyr[0].get("receptor_types")
    win = {k: pyr[0].get(k) for k in ("k_ca_th_L_minus", "k_ca_th_L_plus",
                                      "k_ca_th_H_minus", "k_ca_th_H_plus")}

    wr = nest.Create("weight_recorder")
    nest.CopyModel(SYNAPSE_MODEL, "plastic", {
        "plastic": True, "binarize": BINARIZE, "n_bit": 4, "eta": ETA, "eta_L": ETA,
        "delay": 0.1, "weight_recorder": wr, "receptor_type": int(rt["NMDA_BASAL_SPIKES"])})
    nest.CopyModel(SYNAPSE_MODEL, "static", {
        "plastic": False, "binarize": True, "n_bit": 3, "eta": 0.0, "eta_L": 0.0, "delay": 0.1})

    # three input channels: Poisson -> parrot -> plastic NMDA synapse
    gens = nest.Create("poisson_generator", 3)
    parrots = nest.Create("parrot_neuron", 3)
    nest.Connect(gens, parrots, "one_to_one", {"weight": 1.0, "delay": 0.1})
    for par in parrots:
        nest.Connect(par, pyr, "one_to_one", {"synapse_model": "plastic", "w": W0, "Ibias": IBIAS})
    parrot_id = [int(p_.get("global_id")) for p_ in parrots]

    teacher = nest.Create("poisson_generator", 1)
    nest.Connect(teacher, pyr, "one_to_one",
                 {"synapse_model": "static", "receptor_type": int(rt["AMPA_APICAL_SPIKES"]),
                  "w": 7, "Ibias": IBIAS})

    mm = nest.Create("multimeter", {"record_from": ["Isoma_ca"]})
    nest.Connect(mm, pyr)

    # alternate A / B presentations
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

    # weight trajectories per channel (weight_recorder stores w*Ibias)
    ev = wr.get("events")
    wt, wsnd, wval = np.asarray(ev["times"]), np.asarray(ev["senders"]), np.asarray(ev["weights"]) / IBIAS
    tr = mm.get("events")

    fig, (ax_w, ax_ca) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for (t0, t1, pat) in schedule:
        for ax in (ax_w, ax_ca):
            ax.axvspan(t0 / 1e3, t1 / 1e3, color="#FdF2C9" if pat == "A" else "#EBF1F7",
                       alpha=0.7, lw=0)

    def time_avg(ch, t_from):
        """Time-weighted average weight of channel ch over [t_from, T_SIM] ms."""
        m = wsnd == parrot_id[ch]
        ts = np.concatenate([[0.0], wt[m], [N_ITER * 2 * T_PHASE]])
        vs = np.concatenate([[W0], wval[m], [wval[m][-1] if m.any() else W0]])
        seg_t = np.clip(ts[1:], t_from, None) - np.clip(ts[:-1], t_from, None)
        return float(np.sum(seg_t * vs[:-1]) / max(np.sum(seg_t), 1e-9))

    half = N_ITER * T_PHASE  # midpoint in ms
    for ch, name in enumerate(["W1", "W2", "W3"]):
        m = wsnd == parrot_id[ch]
        ax_w.step(np.concatenate([[0], wt[m]]) / 1e3,
                  np.concatenate([[W0], wval[m]]), where="post", color=WCOL[name], lw=2, label=name)
        ax_w.axhline(time_avg(ch, half), color=WCOL[name], ls=":", lw=1.2, alpha=0.8)
    ax_w.set_ylabel("plastic weight  w [0–15]")
    ax_w.set_ylim(-0.5, 15.5)
    ax_w.set_title("Delta-rule learning: A drives W1,W2 (+teacher); B drives W2,W3 (no teacher)\n"
                   "→ W1 potentiates, W3 depresses; W2 (shared) is pushed both ways → "
                   "intermediate on average (dotted = 2nd-half mean)")
    ax_w.legend(loc="center left", fontsize=9)

    ax_ca.plot(np.asarray(tr["times"]) / 1e3, tr["Isoma_ca"], color="black", lw=0.8,
               label="Isoma_ca (calcium proxy)")
    for key, c in (("k_ca_th_L_plus", "red"), ("k_ca_th_H_plus", "red"),
                   ("k_ca_th_L_minus", "blue"), ("k_ca_th_H_minus", "blue")):
        ax_ca.axhline(win[key], color=c, ls="--", lw=0.8)
    ax_ca.set_ylabel("Isoma_ca (pA)")
    ax_ca.set_xlabel("time (s)")
    ax_ca.legend(loc="upper right", fontsize=8)
    ax_ca.text(0.01, 0.92, "yellow = pattern A (taught), blue = pattern B (untaught)",
               transform=ax_ca.transAxes, fontsize=8, color="#555")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "03_learning.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")

    half = N_ITER * T_PHASE
    print(f"2nd-half mean weights:  W1={time_avg(0, half):.1f}  W2={time_avg(1, half):.1f}  "
          f"W3={time_avg(2, half):.1f}  (start {W0})  "
          f"-> W1 high, W3 low, W2 intermediate")
    print("saved", out)


if __name__ == "__main__":
    main()
