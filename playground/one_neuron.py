# -*- coding: utf-8 -*-
#
# one_neuron.py
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
One neuron example
------------------

This script simulates a single ``iaf_psc_alpha`` neuron driven by one
``dc_generator`` (a constant DC current source) and records its membrane
potential. Each time the neuron reaches threshold it emits a spike and is
clamped at the reset potential for an absolute refractory period ``t_ref``
before it can charge up again — that flat shelf is highlighted on the plot.

See Also
~~~~~~~~

:doc:`twoneurons`

"""

###############################################################################
# First, we import all necessary modules for simulation, analysis and
# plotting, set the verbosity to suppress info messages and reset the kernel.
# Resetting the kernel allows you to execute the script several times in a
# Python shell without interferences from previous NEST simulations.

import matplotlib.pyplot as plt
import nest

nest.set_verbosity("M_WARNING")
nest.ResetKernel()

###############################################################################
# Second, the nodes (neuron and devices) are created using ``Create``. We pick
# a deliberately long refractory period so the refractory shelf is easy to see,
# and store the returned handles in variables for later reference.

T_REF = 10.0  # ms, absolute refractory period
neuron = nest.Create("iaf_psc_alpha", params={"t_ref": T_REF, "tau_m": 10})

# One DC current source, strong enough to drive repeated spiking.
dc = nest.Create("dc_generator", params={"amplitude": 1000.0})  # pA

voltmeter = nest.Create("voltmeter", params={"interval": 0.1})
spike_recorder = nest.Create("spike_recorder")

###############################################################################
# Third, the nodes are connected. The DC source drives the neuron, the
# voltmeter observes the neuron (note the reversed direction — ``Connect``
# reflects the direction of signal flow in the kernel), and the neuron sends
# its spikes to the spike recorder.

nest.Connect(dc, neuron)
nest.Connect(voltmeter, neuron)
nest.Connect(neuron, spike_recorder)

###############################################################################
# Now we simulate the network using ``Simulate``, which takes the desired
# simulation time in milliseconds.

nest.Simulate(200.0)

###############################################################################
# Finally, we pull the recorded data and plot the membrane potential as a
# function of time. On top of the trace we draw the threshold and reset
# levels, mark every spike, and shade the ``t_ref`` window that follows each
# one — the interval during which the neuron ignores the input current.

ev = voltmeter.events
t, v = ev["times"], ev["V_m"]
spikes = spike_recorder.events["times"]

V_th = neuron.get("V_th")
V_reset = neuron.get("V_reset")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, v, color="#1f77b4", lw=1.2, label="V_m")
ax.axhline(V_th, color="#d62728", ls="--", lw=1, label=f"V_th ({V_th:.0f} mV)")
ax.axhline(V_reset, color="#2ca02c", ls=":", lw=1, label=f"V_reset ({V_reset:.0f} mV)")

for i, s in enumerate(spikes):
    ax.axvline(s, color="0.6", lw=0.8)
    ax.axvspan(
        s,
        s + T_REF,
        color="orange",
        alpha=0.2,
        label="refractory period" if i == 0 else None,
    )

ax.set_xlabel("time (ms)")
ax.set_ylabel("membrane potential (mV)")
ax.set_title("iaf_psc_alpha driven by a constant DC current — refractory period")
ax.legend(loc="lower right")
fig.tight_layout()

print(f"{len(spikes)} spikes; t_ref = {T_REF} ms")
plt.show()
