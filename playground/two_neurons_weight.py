# -*- coding: utf-8 -*-
#
# two_neurons_weight.py
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
Interactive two connected neurons
---------------------------------

Like :doc:`step_current_aeif`, but now a *second* ``aeif_cond_alpha`` neuron is
connected to the first. A step current drives neuron 1 (presynaptic); its spikes
are relayed through an excitatory synapse to neuron 2 (postsynaptic). Three
panels share the same time axis:

* top    -- the injected step current,
* middle -- neuron 1 (presynaptic) membrane potential,
* bottom -- neuron 2 (postsynaptic) membrane potential.

Three sliders: the step amplitude (up/down) and onset (left/right) as before,
plus the synaptic weight. Turn the weight up and watch neuron 1's spikes start
to drive neuron 2.

Run it from a normal desktop session (needs an interactive matplotlib backend):

    .venv/bin/python playground/two_neurons_weight.py

See Also
~~~~~~~~

:doc:`step_current_aeif`

"""

###############################################################################
# First, the imports and constants.

import matplotlib.pyplot as plt
import nest
import numpy as np
from matplotlib.widgets import Slider

nest.set_verbosity("M_WARNING")

T_SIM = 300.0   # ms, total simulation time
WIDTH = 100.0   # ms, fixed width of the current step
DELAY = 1.0     # ms, synaptic delay between the two neurons
DT = 0.1        # ms, simulation/recording resolution

AMP_MIN, AMP_MAX, AMP_INIT = 0.0, 1500.0, 1000.0     # step amplitude (pA)
ONSET_MIN, ONSET_MAX, ONSET_INIT = 0.0, 180.0, 50.0  # step onset (ms)
W_MIN, W_MAX, W_INIT = 0.0, 400.0, 200.0             # synaptic weight (nS)


###############################################################################
# Second, a helper that builds the two-neuron network and simulates it. Neuron 1
# is driven by the step current; neuron 1 -> neuron 2 is an excitatory synapse
# whose weight we vary. Returns the time base, both membrane traces and both
# spike trains.

def simulate(amplitude, onset, stop, weight):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT})

    pre = nest.Create("aeif_cond_alpha")
    post = nest.Create("aeif_cond_alpha")

    dc = nest.Create("dc_generator",
                     params={"amplitude": amplitude, "start": onset, "stop": stop})
    nest.Connect(dc, pre)
    nest.Connect(pre, post, syn_spec={"weight": weight, "delay": DELAY})

    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
    nest.Connect(mm, pre)
    mm_post = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": DT})
    nest.Connect(mm_post, post)

    sr_pre = nest.Create("spike_recorder")
    sr_post = nest.Create("spike_recorder")
    nest.Connect(pre, sr_pre)
    nest.Connect(post, sr_post)

    nest.Simulate(T_SIM)

    return (mm.events["times"], mm.events["V_m"], mm_post.events["V_m"],
            sr_pre.events["times"], sr_post.events["times"])


###############################################################################
# Third, set up the figure: three stacked axes sharing the x (time) axis, plus
# three slider axes underneath.

fig, (ax_i, ax_pre, ax_post) = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))
fig.subplots_adjust(bottom=0.22, hspace=0.12)

ref = nest.Create("aeif_cond_alpha")
V_peak = ref.get("V_peak")
V_reset = ref.get("V_reset")

amp_slider = Slider(
    fig.add_axes([0.18, 0.12, 0.7, 0.025]),
    "amplitude (pA)  ↕", AMP_MIN, AMP_MAX, valinit=AMP_INIT, valstep=10, color="#ff7f0e",
)
onset_slider = Slider(
    fig.add_axes([0.18, 0.075, 0.7, 0.025]),
    "onset (ms)  ↔", ONSET_MIN, ONSET_MAX, valinit=ONSET_INIT, valstep=1, color="#7f7f7f",
)
w_slider = Slider(
    fig.add_axes([0.18, 0.03, 0.7, 0.025]),
    "weight (nS)", W_MIN, W_MAX, valinit=W_INIT, valstep=1, color="#2ca02c",
)


###############################################################################
# A small helper: the multimeter samples V_m too coarsely to catch the spike
# peak (aeif resets almost instantly), so set the sample at each spike time to
# V_peak, drawing the upstroke ourselves.

def with_peaks(t, v, spikes):
    v = np.asarray(v, dtype=float).copy()
    if len(spikes):
        idx = np.clip(np.searchsorted(t, spikes), 0, len(t) - 1)
        v[idx] = V_peak
    return v


###############################################################################
# Fourth, the update function: simulate with the current slider values and
# redraw all three panels.

def update(_=None):
    amplitude = amp_slider.val
    onset = onset_slider.val
    stop = min(onset + WIDTH, T_SIM)
    weight = w_slider.val

    t, v_pre, v_post, sp_pre, sp_post = simulate(amplitude, onset, stop, weight)
    v_pre = with_peaks(t, v_pre, sp_pre)
    v_post = with_peaks(t, v_post, sp_post)
    i_trace = np.where((t >= onset) & (t < stop), amplitude, 0.0)

    # Top: injected current.
    ax_i.clear()
    ax_i.plot(t, i_trace, color="#ff7f0e", lw=1.5)
    ax_i.fill_between(t, 0, i_trace, color="#ff7f0e", alpha=0.15)
    ax_i.set_ylabel("current (pA)")
    ax_i.set_ylim(-50, AMP_MAX * 1.05)
    ax_i.set_title(f"neuron 1  →[ weight = {weight:.0f} nS ]→  neuron 2")
    ax_i.spines[["top", "right"]].set_visible(False)

    # Middle: presynaptic neuron.
    ax_pre.clear()
    ax_pre.plot(t, v_pre, color="#1f77b4", lw=1.0)
    ax_pre.plot(sp_pre, np.full_like(sp_pre, V_peak), "o", color="#d62728", ms=4)
    ax_pre.set_ylabel("neuron 1 V_m (mV)")
    ax_pre.set_ylim(V_reset - 5, V_peak + 5)
    ax_pre.text(0.01, 0.9, f"presynaptic — {len(sp_pre)} spikes",
                transform=ax_pre.transAxes, fontsize=9, color="#1f77b4")
    ax_pre.spines[["top", "right"]].set_visible(False)

    # Bottom: postsynaptic neuron.
    ax_post.clear()
    ax_post.plot(t, v_post, color="#9467bd", lw=1.0)
    ax_post.plot(sp_post, np.full_like(sp_post, V_peak), "o", color="#d62728", ms=4)
    ax_post.set_ylabel("neuron 2 V_m (mV)")
    ax_post.set_xlabel("time (ms)")
    ax_post.set_xlim(0, T_SIM)
    ax_post.set_ylim(V_reset - 5, V_peak + 5)
    ax_post.text(0.01, 0.9, f"postsynaptic — {len(sp_post)} spikes",
                 transform=ax_post.transAxes, fontsize=9, color="#9467bd")
    ax_post.spines[["top", "right"]].set_visible(False)

    fig.canvas.draw_idle()


amp_slider.on_changed(update)
onset_slider.on_changed(update)
w_slider.on_changed(update)
update()

plt.show()
