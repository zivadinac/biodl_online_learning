# Hardware circuits: the analog VLSI implementation

Maps to Section 2.3, Fig. 3a/3b, and Methods 4.4.

## What is reused vs new

- **Reused** (previously published): the ultra-low-power AdExp-IF silicon neuron
  (Rubino et al. 2020, ref 30) and the excitatory/inhibitory DPI synapse circuits
  (Chicca 2014 ref 10, Bartolozzi & Indiveri 2007 ref 13).
- **New** (this paper's silicon contribution): the calcium-based dendritic **learning**
  circuits, in two parts:
  1. a **soma-block error circuit** that computes `delta(t)` continuously, and
  2. a per-synapse **weight-update + storage circuit** instantiated at each plastic
     (basal) NMDA synapse.

Apical and basal compartments are physically just combinations of excitatory and
inhibitory DPI synapses whose contributions are summed into `Ia` (apical) and `Ib`
(basal). The sum `Ia + Ib` drives the AdExp-IF soma firing; the difference `Ia - Ib`
drives learning.

## Soma error circuit (Fig. 3a)

Two complementary sub-circuits, one for potentiation, one for depression. They subtract
continuous-time copies of basal `Ib` and apical `Ia`, apply a tolerance, scale by the
learning rate, and gate by the calcium third factor.

Potentiation path produces `I_up`:

```
I_up  proportional to  I_eta * (Ia - Ib - I_eps),  gated by sigma_LTP
```

Depression path produces `I_dn` (complementary, active in the LTD region):

```
I_dn  proportional to  I_eta * (Ib - Ia - I_eps),  gated by sigma_LTD
```

Components:
- `I_eps` is a **tolerance / deadband current**. It is subtracted from the raw error so
  that when `Ia ~= Ib` (negligible error) no update is triggered. Larger `I_eps` widens
  the deadband. This is the analog way of ignoring noise near zero error.
- `I_eta` is the learning-rate current; multiplying by it sets the update magnitude.
- `sigma_LTP` / `sigma_LTD` are voltage variables (`V_sigma_LTP`, `V_sigma_LTD` in the
  schematic) that gate whether the path is enabled, set by whether the soma-integrated
  `Ia + Ib` pushes the neuron into the LTP or LTD calcium region.
- Using two complementary circuits (rather than one signed circuit) lets potentiation and
  depression be matched and lets the same physical layout strategy be reused for both.

The error currents `I_up` / `I_dn` are computed once per neuron at the soma and made
available to **all** of that neuron's plastic synapses.

## Synapse weight-update and storage circuit (Fig. 3b)

This is the clever mixed analog/digital part. Per plastic synapse:

1. **Sample on presynaptic spike**: when the presynaptic input `V_spk` is active, the
   circuit samples `I_up` / `I_dn` from the soma and adds/removes the corresponding charge
   on a **weight capacitor `C_w`**, moving the analog weight voltage `V_w` up or down.
   No presynaptic spike means no sampling, so the update is gated by presynaptic activity
   (the "pre" factor).

2. **Slew-limited pull to midpoint**: between updates, a transconductance amplifier in
   negative feedback (slew-rate limited) slowly drives `V_w` toward the midpoint between
   `V_dd` and ground. This is a bistability/stability mechanism: small accumulated errors
   decay unless they are reinforced strongly enough to push `V_w` to a rail. It restricts
   committed weight changes to cases where the accumulated error on `C_w` is substantial,
   which gives noise robustness.

3. **4-bit ADC + SRAM storage**: when `V_w` reaches its rail limits, a standard digital
   **4-bit counter acts as an ADC**, discretizing the weight and storing it in **SRAM**
   for long-term, non-volatile-style retention. So short-term plasticity lives in analog
   charge on `C_w`; long-term memory is the 4-bit digital value.

   Why 4 bits: prior work (Pfeil et al. 2012, ref 87) showed 4-bit synaptic resolution is
   enough for a wide range of network benchmarks. The digital storage exploits the small
   area of digital logic in advanced nodes.

The combination (analog short-term accumulation + slew pull-to-midpoint + 4-bit digital
long-term store) is the paper's mechanism for being robust to noise while still doing
fine-grained online learning.

## Technology and power

- Target node: **22 nm FDSOI** (fully depleted silicon-on-insulator).
- Transistors operate in **weak inversion (subthreshold)** for ultra-low power, with
  biologically plausible time constants.
- AdExp-IF neuron has spike-frequency adaptation.
- The energy-per-operation goal is comparable to biological neurons, far below
  synchronous digital processors, suited to always-on edge inference + learning.

## Mismatch / variability handling (the central engineering point)

- Analog elements have per-device CV ~ 0.2 from mismatch.
- The behavioral hardware-aware simulator injects mismatch into **all** parameters:
  weights, spike threshold, time constants, adaptation rates, refractory periods.
  Mismatch sweeps run at 0 / 10 / 20 / 30 percent; **20 percent is treated as a
  worst-case fabrication corner**.
- Robustness comes from: E/I balanced populations (variance suppression), the
  calcium-windowed deadband learning (ignores small noisy errors), and the 4-bit +
  pull-to-midpoint weight storage (discrete stable states).

## Validation scope (read this before trusting numbers)

- **Behavioral hardware-aware simulation**: full network, includes transistor-level
  non-idealities abstracted into the model. This is where the population robustness and
  mismatch-sweep results come from (Fig. 2c).
- **Transistor-level circuit simulation** (Fig. 3c): only a **single neuron per type**
  (1 PYR per class, 1 PV, 1 SST, 1 VIP), 3 plastic synapses each, because full-network
  SPICE-level simulation is intractable in runtime.
- There is **no fabricated-silicon measurement** in this preprint version. The current
  values on plots (Fig. 1c) are explicitly "educated estimates" of internal scaling
  factors, not measured.
