# Canonical Cortical Electronic Circuit for Neuromorphic Intelligence

Reference notes for Claude Code, distilled from:

> Maryada\*, Chiara De Luca\*, Arianna Rubino, Chenxi Wen, Matteo Cartiglia,
> Ioan-Iustin Fodorut, Melika Payvand, Giacomo Indiveri.
> "A canonical cortical electronic circuit for neuromorphic intelligence."
> bioRxiv 2025.03.28.646019, posted 2025-04-01. CC-BY 4.0. (\* equal contribution.)
> Institute of Neuroinformatics, UZH/ETH Zurich.

These docs paraphrase the paper for engineering reference. They are not a substitute
for the source. Equations are restated in clean notation; figure callouts point back to
the original.

## What this paper is

A co-design of (1) a computational model of a "canonical cortical microcircuit" and
(2) the mixed-signal analog/digital CMOS circuits that implement it, such that the model
runs natively on neuromorphic hardware and is robust to the analog device mismatch that
normally breaks such hardware. The headline trick is to treat variability as a feature,
not a bug, and to lean on three biological strategies to get reliable computation out of
unreliable substrate:

1. **Populations** of E/I neurons in a balanced regime (variance suppression).
2. **Disinhibition** (VIP -> SST -> PYR-apical) as a gating mechanism to switch
   learning on and off.
3. **Local, spike-driven plasticity** (a calcium-gated three-factor delta rule on
   dendrites).

The circuit motif is one excitatory population (pyramidal, PYR) plus three inhibitory
classes (PV, SST, VIP), with multi-compartment PYR neurons (soma + basal + apical).
It can be configured as an attractor network, a soft winner-take-all, an oscillator, or
an encoding layer, and when learning is enabled it classifies patterns and forms stable
attractors.

## Claimed firsts (use precisely if citing)

- First high-level **hardware-aware** simulation of device mismatch across **all**
  network parameters at once: weights, spike threshold, time constants, adaptation
  rates, refractory periods, then validated against transistor-level circuit simulation.
- First local synaptic-plasticity weight-update **circuit** compatible with multiple
  learning rules: Delta-rule perceptron, BCM, STDP, and BTSP. (The paper writes "BMC";
  see caveats below: it means **BCM**, Bienenstock-Cooper-Munro.)

## File map

| File | Contents |
|------|----------|
| `README.md` | This overview, contributions, caveats, glossary. |
| `architecture.md` | Cell types, compartments, synapse types, connectivity, the disinhibition gate, attractor configuration. |
| `neuron-and-learning.md` | Neuron model equations, calcium proxy, the dendritic three-factor delta learning rule, dendritic current definitions, reduction to classical delta. |
| `hardware-circuits.md` | Analog VLSI: soma error circuit, synapse weight-update + 4-bit storage circuit, technology node, mismatch handling. |
| `experiments.md` | Attractor f-f curve, binary classification of overlapping patterns, mismatch sweep results, circuit-level validation. |

## Mental model in one paragraph

Each PYR neuron continuously computes, in analog continuous time at the soma, an error
`delta(t)` proportional to (apical current - basal current), i.e. (teacher - input).
That error is broadcast to all of the neuron's plastic basal synapses but only *applied*
to a synapse when that synapse receives a presynaptic spike (`dw_i = delta(t_i)`). The
error is additionally gated by a third factor: the neuron's calcium proxy `theta(t)`
(a low-pass of its own firing rate) must lie inside an LTP or LTD window for any update
to happen at all. Disinhibition controls whether the neuron can reach those windows:
when a top-down "attention" cue drives VIP, VIP suppresses SST, which un-gates the apical
current into the soma, pushing firing rate up into the learning regions. Remove the cue
and SST clamps the apical path, firing drops below both windows, and the network is
frozen in inference mode. That is the whole learning/inference switch.

## Why the design holds up under analog mismatch

- E/I balance + recurrence makes population output variance scale better than the naive
  `1/sqrt(N)` CLT bound; the paper cites results where it scales as `1/N` in balanced
  recurrent regimes.
- The calcium-windowed gating means small noisy errors near `Ia ~= Ib` do not move
  weights (an explicit tolerance current `I_eps` enforces a deadband).
- Weights are accumulated in analog on a capacitor short-term, then **discretized to
  4 bits** and stored in SRAM long-term. The analog-to-digital step plus a slew-limited
  pull-to-midpoint on the weight capacitor gives noise robustness and bistable-style
  long-term retention.

## Caveats and corrections worth knowing before you build on this

1. **"BMC" is a typo for BCM.** The paper repeatedly abbreviates Bienenstock-Cooper-Munro
   theory as "BMC". The references list spells the author names correctly. Read every
   "BMC" as **BCM**. Figure 1e shows BCM-like weight-change-vs-postsynaptic-rate curves
   (LTD region then LTP region as rate increases), which is the BCM signature.
2. **It is a preprint, not peer reviewed** (bioRxiv, April 2025). Treat quantitative
   claims accordingly.
3. **Current scaling is indicative.** Fig. 1c axis values are explicitly stated to use
   "educated estimates" for the internal VLSI current scaling factors, not measured
   silicon values. Do not treat the pA/nA numbers as calibrated.
4. **Circuit-level results use a single neuron per type.** Fig. 3c (the SPICE-level run)
   is one PYR per class, one PV, one SST, one VIP, three plastic synapses each, because
   full-network transistor simulation is too slow. The population-level robustness claims
   come from the *behavioral* hardware-aware simulator, not from transistor-level runs.
5. **Equation cross-reference slip.** In Methods 4.1 the text refers to "eq. (7)" for the
   pair `Ia` and `Ib`; `Ia` is actually eq. (6) and `Ib` is eq. (7). Minor.
6. **Overlap formula is stated oddly.** The active-synapse count is given as
   `0.5 * (1 + 0.5 * ov) * N`. Taken literally, `ov = 1` (full overlap) gives 0.75 N
   active synapses, and `ov = 0` gives 0.5 N. Re-derive before relying on it; the figures
   were generated at `ov = 0.5`. See `experiments.md`.
7. **No measured silicon.** Everything is simulation: a behavioral hardware-aware model
   plus transistor-level circuit simulation. There is no fabricated-chip measurement in
   this version. The neuron (Rubino 2020, ref 30) and DPI synapse (refs 10, 13) circuits
   were previously published; the *learning* circuits are the new silicon contribution
   and are validated in simulation only.

## Glossary

- **PYR**: excitatory pyramidal neuron population (3 compartments: soma, basal, apical).
- **PV / SST / VIP**: inhibitory interneuron classes (parvalbumin, somatostatin,
  vasoactive intestinal peptide). Modeled with basal + soma compartments only.
- **AdExp-IF**: adaptive exponential integrate-and-fire neuron model / circuit.
- **DPI**: differential pair integrator, the current-mode synapse circuit.
- **NMDA / AMPA / GABA_A / GABA_B**: the four modeled synapse types (slow excitatory,
  fast excitatory, shunting+subtractive inhibitory, subtractive inhibitory).
- **theta(t)**: calcium proxy, a low-pass filter of the neuron's own spike train; the
  third factor that gates learning.
- **sigma_LTP / sigma_LTD**: binary gating variables, 1 inside the respective calcium
  window, 0 outside.
- **I_eps**: tolerance/deadband current subtracted from the error to ignore tiny errors.
- **I_up / I_dn**: potentiation / depression update currents produced at the soma.
- **CV**: coefficient of variation, used here as the mismatch level (mismatch sweeps go
  to ~20-30%, with 20% treated as a worst-case fabrication corner).
- **WTA**: winner-take-all (soft, implemented via PV feedback inhibition).
- **FDSOI 22 nm**: the target CMOS technology node.
