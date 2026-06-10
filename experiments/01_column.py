"""Milestone A: the 4-neuron canonical-microcircuit column, wired and spiking.

One PYR + one PV + one SST + one VIP, driven by Poisson input + teacher +
attention cue. Confirms the connectivity is correct: every cell spikes and the
PYR integrates basal + apical currents. Saves a raster + trace figure.

Run:  python experiments/01_column.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.microcircuit import Microcircuit  # noqa: E402

nest.set_verbosity("M_ERROR")

T_SIM = 1000.0  # ms
INPUT_RATE = 50.0
TEACHER_RATE = 200.0
CUE_RATE = 100.0

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
COLORS = {"pyr": "#C0392B", "pv": "#2874A6", "sst": "#154360", "vip": "#5DADE2"}


def main() -> None:
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    net = (
        Microcircuit(n_pyr=1, n_pv=1, n_sst=1, n_vip=1)
        .build()
        .drive(input_rate=INPUT_RATE, teacher_rate=TEACHER_RATE, cue_rate=CUE_RATE)
        .attach_recorders()
    )

    nest.Simulate(T_SIM)

    rates = net.rates(T_SIM)
    print("firing rates (Hz):", {k: round(v, 1) for k, v in rates.items()})

    fig, (ax_raster, ax_pyr) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 3]
    )

    for row, key in enumerate(["pyr", "pv", "sst", "vip"]):
        sp = net.spikes(key)
        ax_raster.plot(sp["times"], [row] * len(sp["times"]), "|",
                       color=COLORS[key], markersize=12, mew=2)
    ax_raster.set_yticks(range(4))
    ax_raster.set_yticklabels([f"{k.upper()} ({rates[k]:.0f} Hz)" for k in ["pyr", "pv", "sst", "vip"]])
    ax_raster.set_ylim(-0.5, 3.5)
    ax_raster.set_title("Canonical microcircuit column (1 neuron per type) — spikes")
    ax_raster.invert_yaxis()

    tr = net.trace("pyr")
    ax_pyr.plot(tr["times"], tr["Ibasal"], color="#1E8449", label="I_basal (input + PV)")
    ax_pyr.plot(tr["times"], tr["Iapical"], color="#7B241C", label="I_apical (teacher - SST)")
    ax_pyr.plot(tr["times"], tr["Isoma_mem"], color="black", lw=0.8, label="I_soma (membrane)")
    ax_pyr.set_xlabel("time (ms)")
    ax_pyr.set_ylabel("current (pA)")
    ax_pyr.set_title("PYR somatic currents")
    ax_pyr.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "01_column.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
