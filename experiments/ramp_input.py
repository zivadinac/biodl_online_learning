"""Poisson input drive under a ramping rate.

The input rate ramps 0 -> 10 Hz over 0-5 s, then (dropping to 5 Hz) ramps
5 -> 20 Hz over 5-10 s. We plot the commanded rate profile on top and the spikes
the inhomogeneous Poisson process actually emits below: the spike density tracks
the rate. (These are the *input* spikes that drive the network, not neuron
output.) Several independent realizations are shown so the ramp is visible.

The generator's spike train is captured with parrot_neurons (which re-emit each
spike they receive), exactly as in the dynaple plasticity tutorial.

Run:  python experiments/ramp_input.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nest  # noqa: E402

nest.set_verbosity("M_ERROR")

T_SIM = 10_000.0  # ms
DT = 50.0         # ms, rate-profile resolution
N_TRIALS = 12     # independent Poisson realizations to show

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def input_ramp(times: np.ndarray) -> np.ndarray:
    """0->10 Hz over [0,5] s, then 5->20 Hz over [5,10] s (Hz at each time, ms)."""
    rate = np.empty_like(times)
    first = times <= 5000.0
    rate[first] = 0.0 + (10.0 - 0.0) * (times[first] / 5000.0)
    rate[~first] = 5.0 + (20.0 - 5.0) * ((times[~first] - 5000.0) / 5000.0)
    return rate


def main() -> None:
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    times = np.arange(0.0, T_SIM, DT)
    rates = input_ramp(times)

    # rate_times must be strictly > 0 and on the grid; nudge the first point off 0
    rate_times = times.copy()
    rate_times[0] = 0.1
    gen = nest.Create("inhomogeneous_poisson_generator", 1,
                      {"rate_times": rate_times.tolist(), "rate_values": rates.tolist()})

    parrots = nest.Create("parrot_neuron", N_TRIALS)
    nest.Connect(gen, parrots, "all_to_all", {"delay": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(parrots, sr)

    nest.Simulate(T_SIM)

    ev = sr.get("events")
    spike_t = ev["times"] / 1e3
    spike_row = ev["senders"] - parrots[0].get("global_id")

    fig, (ax_rate, ax_spk) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[1, 1.5]
    )

    ax_rate.plot(times / 1e3, rates, color="#9B1B1B", lw=2)
    ax_rate.fill_between(times / 1e3, rates, color="#9B1B1B", alpha=0.12)
    ax_rate.set_ylabel("input rate (Hz)")
    ax_rate.set_title("Ramping Poisson input drive: commanded rate (top) and emitted spikes (bottom)")
    ax_rate.axvline(5.0, color="grey", ls="--", lw=1)
    ax_rate.set_ylim(0, 22)

    ax_spk.plot(spike_t, spike_row, "|", color="#1B3B6F", markersize=7, mew=1.0)
    ax_spk.set_ylabel(f"Poisson realization (1–{N_TRIALS})")
    ax_spk.set_xlabel("time (s)")
    ax_spk.set_ylim(-0.5, N_TRIALS - 0.5)
    ax_spk.axvline(5.0, color="grey", ls="--", lw=1)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "ramp_input.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    total = len(spike_t)
    print(f"emitted {total} input spikes across {N_TRIALS} realizations "
          f"(~{total / N_TRIALS / (T_SIM / 1e3):.1f} Hz mean)")
    print("saved", out)


if __name__ == "__main__":
    main()
