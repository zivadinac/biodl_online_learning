"""Poisson input drive as a single rate step.

The input rate holds at RATE_LOW, then jumps to RATE_HIGH at STEP_TIME. We plot
the commanded rate on top and the spikes the inhomogeneous Poisson process emits
below: the spike density jumps at the step and is flat on either side. (These are
the *input* spikes that drive the network, not neuron output.) Several
independent realizations are shown.

The generator's spike train is captured with parrot_neurons (which re-emit each
spike they receive), exactly as in the dynaple plasticity tutorial.

Run:  python experiments/step_input.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import nest  # noqa: E402

nest.set_verbosity("M_ERROR")

T_SIM = 10_000.0  # ms
DT = 5.0  # ms, rate-profile resolution
N_TRIALS = 100  # independent Poisson realizations to show
STEP_TIME = 5000.0  # ms, when the rate jumps
RATE_LOW = 5.0  # Hz before the step (x)
RATE_HIGH = 50.0  # Hz after the step (y)

FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)


def input_step(times: np.ndarray) -> np.ndarray:
    """Single step: RATE_LOW before STEP_TIME, RATE_HIGH after (Hz, times in ms)."""
    return np.where(times < STEP_TIME, RATE_LOW, RATE_HIGH)


def main() -> None:
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    times = np.arange(0.0, T_SIM, DT)
    rates = input_step(times)

    # rate_times must be strictly > 0 and on the grid; nudge the first point off 0
    rate_times = times.copy()
    rate_times[0] = 0.1
    gen = nest.Create(
        "inhomogeneous_poisson_generator",
        1,
        {"rate_times": rate_times.tolist(), "rate_values": rates.tolist()},
    )

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

    ax_rate.plot(times / 1e3, rates, color="#9B1B1B", lw=2, drawstyle="steps-post")
    ax_rate.fill_between(times / 1e3, rates, color="#9B1B1B", alpha=0.12, step="post")
    ax_rate.set_ylabel("input rate (Hz)")
    ax_rate.set_title(
        f"Step Poisson input drive: {RATE_LOW:.0f} -> {RATE_HIGH:.0f} Hz "
        f"at {STEP_TIME / 1e3:.0f} s"
    )
    ax_rate.set_ylim(0, RATE_HIGH * 1.2)

    ax_spk.plot(spike_t, spike_row, "|", color="#1B3B6F", markersize=7, mew=1.0)
    ax_spk.set_ylabel(f"Poisson realization (1–{N_TRIALS})")
    ax_spk.set_xlabel("time (s)")
    ax_spk.set_ylim(-0.5, N_TRIALS - 0.5)

    for ax in (ax_rate, ax_spk):  # mark the step
        ax.axvline(STEP_TIME / 1e3, color="grey", ls="--", lw=1)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "step_input.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    total = len(spike_t)
    print(f"emitted {total} input spikes across {N_TRIALS} realizations")
    print("saved", out)


if __name__ == "__main__":
    main()
