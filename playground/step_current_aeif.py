# -*- coding: utf-8 -*-
#
# step_current_aeif.py
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
Interactive step current -> aeif_cond_alpha
-------------------------------------------

Two plots sharing the same time axis:

* top    -- the injected step current over time,
* bottom -- the membrane potential of an ``aeif_cond_alpha`` neuron driven by it.

Two sliders move the current step around: one changes its amplitude (up/down)
and one changes when it switches on (left/right). Every time you move a slider
the network is re-simulated and both plots update live, so you can watch how the
adaptive-exponential neuron responds to where and how hard you push it.

Run it from a normal desktop session (needs an interactive matplotlib backend):

    .venv/bin/python playground/step_current_aeif.py

See Also
~~~~~~~~

:doc:`one_neuron`

"""

###############################################################################
# First, the imports and constants. The step has a fixed width and is moved
# left/right by its onset time; its height is the amplitude. Both are slider
# controlled.

import matplotlib.pyplot as plt
import nest
import numpy as np
from matplotlib.widgets import Slider

nest.set_verbosity("M_WARNING")

T_SIM = 300.0  # ms, total simulation time
WIDTH = 100.0  # ms, fixed width of the current step
DT = 0.1  # ms, simulation/recording resolution

AMP_MIN, AMP_MAX, AMP_INIT = 0.0, 1500.0, 600.0  # step amplitude (pA)
ONSET_MIN, ONSET_MAX, ONSET_INIT = 0.0, 180.0, 50.0  # step onset (ms)


###############################################################################
# Second, a helper that simulates the neuron for a step current of the given
# amplitude, onset and stop time. It returns the recorded membrane potential
# trace and the output spike times. The injected-current waveform is rebuilt
# analytically for the top plot.


def simulate(amplitude, onset, stop):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT})

    neuron = nest.Create("aeif_cond_alpha")
    dc = nest.Create(
        "dc_generator", params={"amplitude": amplitude, "start": onset, "stop": stop}
    )
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
    sr = nest.Create("spike_recorder")

    nest.Connect(dc, neuron)
    nest.Connect(mm, neuron)
    nest.Connect(neuron, sr)

    nest.Simulate(T_SIM)

    ev = mm.events
    return ev["times"], ev["V_m"], sr.events["times"]


###############################################################################
# Third, set up the figure: two stacked axes sharing the x (time) axis, plus
# two slider axes underneath.

fig, (ax_i, ax_v) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
fig.subplots_adjust(bottom=0.20, hspace=0.12)

ref = nest.Create("aeif_cond_alpha")
V_peak = ref.get("V_peak")
V_reset = ref.get("V_reset")
E_L = ref.get("E_L")

amp_slider = Slider(
    fig.add_axes([0.15, 0.09, 0.7, 0.03]),
    "amplitude (pA)  ↕",
    AMP_MIN,
    AMP_MAX,
    valinit=AMP_INIT,
    valstep=10,
    color="#ff7f0e",
)
onset_slider = Slider(
    fig.add_axes([0.15, 0.04, 0.7, 0.03]),
    "onset (ms)  ↔",
    ONSET_MIN,
    ONSET_MAX,
    valinit=ONSET_INIT,
    valstep=1,
    color="#1f77b4",
)


###############################################################################
# Fourth, the update function: simulate with the current slider values and
# redraw both panels. Clearing the axes keeps the sliders untouched.


def update(_=None):
    amplitude = amp_slider.val
    onset = onset_slider.val
    stop = min(onset + WIDTH, T_SIM)

    t, v, spikes = simulate(amplitude, onset, stop)

    # The multimeter samples V_m too coarsely to catch the spike peak (aeif
    # resets almost instantly), so draw the upstroke ourselves: set the sample
    # at each spike time to V_peak so the trace visibly spikes up to the marker.
    v = np.asarray(v, dtype=float).copy()
    if len(spikes):
        idx = np.clip(np.searchsorted(t, spikes), 0, len(t) - 1)
        v[idx] = V_peak

    # Injected-current waveform (a rectangular step), rebuilt analytically.
    i_trace = np.where((t >= onset) & (t < stop), amplitude, 0.0)

    # Top: current over time.
    ax_i.clear()
    ax_i.plot(t, i_trace, color="#ff7f0e", lw=1.5)
    ax_i.fill_between(t, 0, i_trace, color="#ff7f0e", alpha=0.15)
    ax_i.set_ylabel("injected current (pA)")
    ax_i.set_ylim(-50, AMP_MAX * 1.05)
    ax_i.set_title("step current  →  aeif_cond_alpha response")
    ax_i.spines[["top", "right"]].set_visible(False)

    # Bottom: membrane potential, with output spikes marked at V_peak.
    ax_v.clear()
    ax_v.plot(t, v, color="#1f77b4", lw=1.0, label="V_m")
    ax_v.plot(
        spikes,
        np.full_like(spikes, V_peak),
        "o",
        color="#d62728",
        ms=5,
        label=f"spikes ({len(spikes)})",
    )
    ax_v.set_ylabel("membrane potential (mV)")
    ax_v.set_xlabel("time (ms)")
    ax_v.set_xlim(0, T_SIM)
    ax_v.set_ylim(V_reset - 5, V_peak + 5)
    ax_v.legend(loc="upper right", frameon=False, fontsize=9)
    ax_v.spines[["top", "right"]].set_visible(False)

    fig.canvas.draw_idle()


amp_slider.on_changed(update)
onset_slider.on_changed(update)
update()

plt.show()
