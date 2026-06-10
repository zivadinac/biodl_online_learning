# -*- coding: utf-8 -*-
#
# interactive_spikes.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Interactive input-spikes demo
-----------------------------

A single ``iaf_psc_alpha`` neuron receives a burst of input spikes delivered by
a ``spike_generator``. A matplotlib slider controls how many input spikes are
delivered; every time you move it the network is re-simulated and the plot
updates live, so you can watch the neuron's membrane potential and its output
spikes change as you feed it more (or fewer) inputs.

Run it from a normal desktop session (needs an interactive matplotlib backend):

    .venv/bin/python playground/interactive_spikes.py

See Also
~~~~~~~~

:doc:`one_neuron`

"""

###############################################################################
# First, the imports and a few constants. Input spikes are spread evenly over
# a window, so more spikes means a higher effective input rate. The synaptic
# weight is tuned so a handful of inputs are needed before the neuron fires.

import matplotlib.pyplot as plt
import nest
import numpy as np
from matplotlib.widgets import Slider

nest.set_verbosity("M_WARNING")

T_SIM = 200.0      # ms, total simulation time
T_START = 20.0     # ms, first input spike
T_END = 180.0      # ms, last input spike
T_REF = 2.0        # ms, refractory period
WEIGHT = 400.0     # pA, synaptic weight of each input spike
DT = 0.1           # ms, simulation/recording resolution

N_MIN, N_MAX, N_INIT = 0, 200, 20      # input-spikes slider: range and start
TH_MIN, TH_MAX, TH_INIT = -55.0, -40.0, -55.0  # threshold slider (mV)


###############################################################################
# Second, a helper that simulates the neuron given a number of input spikes and
# a firing threshold, and returns the membrane-potential trace, the output spike
# times, and the input spike times we injected. We reset the kernel each call so
# runs are independent.

def simulate(n_spikes, v_th):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT})

    neuron = nest.Create("iaf_psc_alpha", params={"t_ref": T_REF, "V_th": v_th})
    vm = nest.Create("voltmeter", params={"interval": DT})
    sr = nest.Create("spike_recorder")
    nest.Connect(vm, neuron)
    nest.Connect(neuron, sr)

    if n_spikes > 0:
        # Evenly spaced input spikes, aligned to the simulation resolution.
        times = np.linspace(T_START, T_END, n_spikes)
        times = np.round(times / DT) * DT
        times = np.unique(times)  # spike_generator needs strictly increasing times
        sg = nest.Create("spike_generator", params={"spike_times": times})
        nest.Connect(sg, neuron, syn_spec={"weight": WEIGHT})
    else:
        times = np.array([])

    nest.Simulate(T_SIM)

    ev = vm.events
    return ev["times"], ev["V_m"], sr.events["times"], times


###############################################################################
# Third, set up the figure: a main axis for the trace and two thin axes below
# it that hold the sliders (number of input spikes, and the firing threshold).

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.subplots_adjust(bottom=0.26)

V_reset = nest.Create("iaf_psc_alpha").get("V_reset")

n_slider = Slider(
    fig.add_axes([0.15, 0.10, 0.7, 0.04]),
    "input spikes", N_MIN, N_MAX, valinit=N_INIT, valstep=1, color="#1f77b4",
)
th_slider = Slider(
    fig.add_axes([0.15, 0.04, 0.7, 0.04]),
    "threshold (mV)", TH_MIN, TH_MAX, valinit=TH_INIT, valstep=0.5, color="#d62728",
)


###############################################################################
# Fourth, an update function that runs one simulation with the current slider
# values and (re)draws everything. It clears the main axis each call, which
# keeps the sliders untouched. Both sliders call it.

def update(_=None):
    n_spikes = int(n_slider.val)
    v_th = th_slider.val
    t, v, out_spikes, in_spikes = simulate(n_spikes, v_th)

    ax.clear()
    ax.plot(t, v, color="#1f77b4", lw=1.2, label="V_m")
    ax.axhline(v_th, color="#d62728", ls="--", lw=1, label=f"V_th ({v_th:.1f} mV)")

    # Input spikes: gray ticks along the bottom of the axis.
    ax.plot(in_spikes, np.full_like(in_spikes, V_reset - 1.0), "|",
            color="0.5", ms=10, mew=1.2, label=f"input ({len(in_spikes)})")
    # Output spikes: red dots at threshold.
    ax.plot(out_spikes, np.full_like(out_spikes, v_th), "o",
            color="#d62728", ms=6, label=f"output ({len(out_spikes)})")

    ax.set_xlim(0, T_SIM)
    ax.set_ylim(V_reset - 3, TH_MAX + 3)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("membrane potential (mV)")
    ax.set_title("iaf_psc_alpha driven by a pulse of input spikes")
    ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.canvas.draw_idle()


n_slider.on_changed(update)
th_slider.on_changed(update)
update()

plt.show()
