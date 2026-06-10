# -*- coding: utf-8 -*-
#
# fi_curve.py
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
f-I curve
---------

This script measures the f-I curve (firing rate as a function of injected
current) of a single ``iaf_psc_alpha`` neuron. For each current amplitude we
run a fresh simulation, count the emitted spikes over a fixed window, and
convert that to a firing rate in spikes/s. Below the rheobase the steady-state
membrane potential never reaches threshold, so the rate is zero; above it the
rate climbs and gradually saturates toward 1 / t_ref.

See Also
~~~~~~~~

:doc:`one_neuron`

"""

###############################################################################
# First, the imports. We measure the firing rate over a long-ish window so the
# rate estimate is stable, and after discarding an initial transient.

import matplotlib.pyplot as plt
import nest
import numpy as np

nest.set_verbosity("M_WARNING")

T_SIM = 2000.0  # ms, measurement window
T_TRANSIENT = 0.0  # ms, discarded before counting spikes
T_REF = 2.0  # ms, refractory period (sets the rate ceiling 1000/t_ref)

# Current amplitudes to sweep (pA). Range chosen like the NESTML f-I tutorial
# (up to ~1 nA) with a fine step so the onset at rheobase is smooth.
amplitudes = np.linspace(0.0, 1000.0, 40)


###############################################################################
# Second, a helper that runs one simulation at a given DC amplitude and returns
# the firing rate in spikes/s. We reset the kernel each call so runs are
# independent, then count only the spikes after the transient.

def firing_rate(amplitude):
    nest.ResetKernel()

    neuron = nest.Create("iaf_psc_alpha", params={"t_ref": T_REF})
    dc = nest.Create("dc_generator", params={"amplitude": amplitude})
    sr = nest.Create("spike_recorder")

    nest.Connect(dc, neuron)
    nest.Connect(neuron, sr)

    nest.Simulate(T_SIM)

    spikes = sr.events["times"]
    counted = spikes[spikes > T_TRANSIENT]
    duration_s = (T_SIM - T_TRANSIENT) / 1000.0
    return len(counted) / duration_s


###############################################################################
# Third, sweep the amplitudes and collect the rates.

rates = np.array([firing_rate(a) for a in amplitudes])

###############################################################################
# We can also compute the analytic rheobase: the current at which the
# steady-state membrane potential V_m = I_e * R just touches threshold. For
# iaf_psc_alpha the membrane resistance is R = tau_m / C_m.

ref = nest.Create("iaf_psc_alpha")
tau_m = ref.get("tau_m")  # ms
C_m = ref.get("C_m")  # pF
E_L = ref.get("E_L")  # mV
V_th = ref.get("V_th")  # mV
R = tau_m / C_m  # GOhm (mV/pA)
rheobase = (V_th - E_L) / R  # pA

###############################################################################
# Finally, plot the f-I curve and mark the rheobase. We let the y-axis
# auto-scale to the measured rates so the curve fills the figure (the 1/t_ref
# ceiling of 500 Hz is far above this range and would otherwise flatten it).

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(amplitudes, rates, "o-", color="#1f77b4", lw=1.5, ms=4, label="$I_{inj}$ vs rate")
ax.axvline(rheobase, color="#d62728", ls="--", lw=1, label=f"rheobase ({rheobase:.0f} pA)")

ax.set_xlabel("injected current $I_{inj}$ (pA)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("f-I curve of an iaf_psc_alpha neuron")
ax.margins(x=0.02)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

print(f"rheobase ~ {rheobase:.1f} pA;  max rate {rates.max():.0f} Hz")
plt.show()
