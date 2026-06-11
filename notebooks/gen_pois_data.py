import numpy as np
import matplotlib.pyplot as plt
import nest

# Initialize NEST


def generate_poisson_spikes(frs, phase_dur_s=2., num_pres=20, seed=None):
    """ Generate time-varying Poisson distributed spike train.

        Args:
            frs - list of firing rates
            phase_dur_s - duration of spiking presentations
            num_pres - number of successive presentations of each frequency in frs
            seed - random seed, if None (default) will be choosen randomly on every call
        Return:
            spike_times - spike times in milliseconds
            sim_time - total simulation time
    """
    # reset and randomize
    nest.ResetKernel()
    if seed is None:
        seed = np.random.randint(100)
    nest.rng_seed = seed
    # generate poisson spikes
    phase_duration_ms = phase_dur_s * 1000.0
    frs = np.concatenate([[(f, 0.)] * num_pres for f in frs]).flatten()
    # Convert to NEST internal units (milliseconds and Hz)
    sim_time = len(frs) * phase_duration_ms + .1
    # Create the Inhomogeneous Poisson Input (using poisson_generator)
    # We change the rate at specific time points using 'amplitude_times' and 'amplitude_values'
    poisson_input = nest.Create("inhomogeneous_poisson_generator")
    nest.SetStatus(
        poisson_input,
        {
            "rate_times": np.arange(len(frs)) * phase_duration_ms + .1,
            "rate_values": frs,
        },
    )
    # 2. Create Recording Devices
    spike_recorder_input = nest.Create("spike_recorder")
    nest.Connect(poisson_input, spike_recorder_input)
    nest.Simulate(sim_time)
    input_spikes = nest.GetStatus(spike_recorder_input, "events")[0]
    # Extract times and senders
    spike_times = input_spikes["times"]
    return spike_times, sim_time


if __name__ == "__main__":
    input_times, sim_time = generate_poisson_spikes([5, 50, 20], 2, 20)
    input_times /= 1000  # spike times to milliseconds
    # Plotting Firing Rates Over Time (using histograms to estimate rate)
    bin_width = 1  # 1s bins for rate estimation
    bins = np.arange(0, sim_time / 1000, bin_width)
    fig, ax = plt.subplots(1, 1)
    # Plot Input Firing Rate
    ax.hist(
        input_times,
        bins=bins,
        weights=[1 / bin_width] * len(input_times),
        histtype="step",
        label="Poisson Input",
        color="black",
        linewidth=2,
    )
    # Formatting
    ax.set_title("Firing Rates Through Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated Firing Rate (Hz)")
    fig.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
