"""Milestone B: the disinhibition gate (the learning on/off switch).

Input and teacher run constantly; the top-down attention cue is delivered as a
rate-vector pulse in the middle third. Cue ON -> VIP fires -> SST is suppressed ->
the PYR apical path is un-gated -> PYR firing (and its calcium proxy) rise into
the learning regime. Cue OFF -> SST clamps the apical path -> PYR firing falls.

SST is given a tonic baseline drive (``tonic(sst=...)``) so it is active by
default in this 1-neuron column, standing in for the population drive it would
receive in the full network. The gate works because PYR's apical is half-wave
rectified (``Iapical_low`` is sign-corrected in biodl/params.py).

Run:  python experiments/02_disinhibition.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.microcircuit import Microcircuit  # noqa: E402

nest.set_verbosity("M_ERROR")

T_SIM = 3000.0  # ms
CUE_ON = (1000.0, 2000.0)
INPUT_RATE = 0.0       # no bottom-up drive: PYR firing is set by the apical gate
TEACHER_RATE = 150.0   # constant top-down teacher
CUE_RATE = 400.0       # attention-cue pulse height
SST_TONIC = 200.0      # baseline drive keeping SST active when un-gated
BIN = 100.0            # ms, time-resolved rate (smoothing for a 1-cell raster)

# stronger apical gate than the column default, to make the swing legible
# (static synapses are 3-bit: w=7, strength via Ibias)
GATE_WEIGHTS = {"teacher_pyr": (7, 12900), "sst_pyr": (7, 17000), "vip_sst": (7, 12900)}

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def binned_rate(times, t_sim, bin_ms):
    edges = np.arange(0.0, t_sim + bin_ms, bin_ms)
    counts, _ = np.histogram(times, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts * 1e3 / bin_ms


def main() -> None:
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    times = np.array([0.0, CUE_ON[0], CUE_ON[1]])
    cue_profile = [0.0, CUE_RATE, 0.0]

    net = (
        Microcircuit(n_pyr=1, n_pv=1, n_sst=1, n_vip=1, weights=GATE_WEIGHTS)
        .build()
        .drive_profiles(times, input=INPUT_RATE, teacher=TEACHER_RATE, cue=cue_profile)
        .tonic(sst=SST_TONIC)
        .attach_recorders()
    )

    nest.Simulate(T_SIM)

    t_pyr, r_pyr = binned_rate(net.spikes("pyr")["times"], T_SIM, BIN)
    t_sst, r_sst = binned_rate(net.spikes("sst")["times"], T_SIM, BIN)
    t_vip, r_vip = binned_rate(net.spikes("vip")["times"], T_SIM, BIN)
    tr = net.trace("pyr")

    fig, (ax_rate, ax_ca) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax in (ax_rate, ax_ca):
        ax.axvspan(*CUE_ON, color="#F9E79F", alpha=0.55, label="attention cue ON")

    ax_rate.plot(t_pyr, r_pyr, color="#C0392B", lw=2.2, label="PYR")
    ax_rate.plot(t_vip, r_vip, color="#3FAE3A", lw=1.3, label="VIP")
    ax_rate.plot(t_sst, r_sst, color="#3FB7E8", lw=1.3, label="SST")
    ax_rate.set_ylabel("firing rate (Hz)")
    ax_rate.set_title("Disinhibition gate: cue ON un-gates PYR  (VIP -| SST -| PYR-apical)")
    ax_rate.legend(loc="upper right", fontsize=8, ncol=2)

    ax_ca.plot(tr["times"], tr["Iapical"], color="#D98880", lw=0.7, alpha=0.7, label="PYR Iapical")
    ax_ca.plot(tr["times"], tr["Isoma_ca"], color="black", lw=1.4, label="PYR Isoma_ca (calcium proxy)")
    ax_ca.set_xlabel("time (ms)")
    ax_ca.set_ylabel("current (pA)")
    ax_ca.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "02_disinhibition.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")

    def mean_in(times, lo, hi):
        return 1e3 * np.sum((times >= lo) & (times < hi)) / (hi - lo)

    pyr = net.spikes("pyr")["times"]
    print(f"PYR rate  cue-OFF: {mean_in(pyr, 0, 1000):.1f} Hz | "
          f"cue-ON: {mean_in(pyr, 1000, 2000):.1f} Hz | "
          f"cue-OFF: {mean_in(pyr, 2000, 3000):.1f} Hz")
    print("saved", out)


if __name__ == "__main__":
    main()
