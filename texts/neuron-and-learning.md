# Neuron model and the dendritic learning rule

Maps to Methods 4.1, 4.2, 4.3 and Section 2.1 ("The dendritic delta learning rule").
All equations restated in clean notation. The `delta_{i,t}(t - t_n)` terms are spike
trains: Dirac deltas at the spike times `t_n` of presynaptic source `i`.

## Neuron model (multi-compartment AdExp-IF)

The soma integrates basal and apical dendritic currents into a membrane current
`I_soma`. (Eq. 2.)

```
tau_soma * dI_soma/dt + I_soma(t) = I_basal(t) + I_apical(t) + I_const
```

Basal dendrite, neuron j, summing over presynaptic sources i and their spikes n (Eq. 3):

```
tau_basal * dI_basal_j/dt + I_basal_j = sum_i sum_n  I_w,ij * delta_{i,t}(t - t_n)
```

Apical dendrite (Eq. 4):

```
tau_apical * dI_apical_j/dt + I_apical_j = sum_i sum_n  I_wa,ij * delta_{i,t}(t - t_n)
```

Synaptic current weight, where `I_bias` is the analog current standing in for the weight
and `w` is the 4-bit digital weight that scales it:

```
I_w = w * I_bias
```

So each synaptic event injects a quantum of charge proportional to the learned 4-bit
weight times a fixed bias current. PYR uses soma + basal + apical; interneurons use
basal + soma only.

## Calcium proxy (the third factor)

Each neuron accumulates a calcium trace `theta`, a low-pass filter of its own output
spike train (Eq. 5):

```
tau_ca * dtheta_j/dt + theta_j(t) = sum_n  beta * delta_{j,t}(t - t_n)
```

`beta` scales each spike's contribution. `theta(t)` is therefore a proxy for the neuron's
recent mean firing rate and is the gating variable for plasticity. In hardware it is
computed by a DPI current-mode integrator. Subthreshold dynamics, `theta`, and the error
term are all computed in **continuous time** by analog blocks (no clocked update).

## The dendritic three-factor delta rule

This is the heart of the paper. It is a spike-driven Hebbian rule derived from the
original Widrow-Hoff / perceptron **delta rule**, made into a **three-factor** rule by a
calcium-dependent gate.

### Error signal at the soma (Eq. 1)

Computed continuously and shared across all of a neuron's plastic synapses:

```
delta(t) = eta * (Ia - Ib) * sigma_LTP    if Ia >= Ib
         = eta * (Ia - Ib) * sigma_LTD    if Ia <  Ib
```

with the calcium-window gates:

```
sigma_LTP = 1  if  theta_LTP_minus < theta(t) < theta_LTP_plus    else 0
sigma_LTD = 1  if  theta_LTD_minus < theta(t) < theta_LTD_plus    else 0
```

where:
- `eta` is the learning rate,
- `Ia` is the total apical current, `Ib` the total basal current (defined below),
- `theta(t)` is the calcium proxy,
- `theta_LTP_+/-`, `theta_LTD_+/-` are the upper/lower stop-learning thresholds defining
  the LTP and LTD windows.

Reading of the three factors:
1. **Pre**: presence of a presynaptic spike (controls *when* the update is applied, below).
2. **Post / error**: `(Ia - Ib)`, the apical-minus-basal current, i.e. teacher-minus-input.
3. **Third factor**: `theta(t)` gating via `sigma_LTP/LTD`, i.e. the neuron's recent
   activity must sit inside the right calcium window or nothing changes ("stop-learning").

### Local per-synapse update

The continuous-time `delta(t)` is broadcast to all of the neuron's basal NMDA synapses,
but a given synapse `i` only updates **when it receives a presynaptic spike** at time
`t_i`:

```
dw_i = delta(t_i)
```

That is the locality: each synapse samples the shared soma error at its own spike times.
Worked example in Fig. 1c: weight jumps appear only at presynaptic spike times, are
proportional to the error amplitude, and only occur when `theta(t)` is inside the
matching window. The paper notes a teaching example: when `delta(t)` goes positive around
t = 250 ms but `theta(t)` is in the LTD-only region, no update happens, because the LTP
gate is closed.

### BCM consistency

Plotting mean weight change against postsynaptic firing rate gives the BCM signature: a
depression region at low/intermediate rates and a potentiation region at high rates, with
a crossover (Fig. 1e). NOTE: the paper writes this as "BMC"; the correct name is **BCM**
(Bienenstock-Cooper-Munro). The calcium windows `theta_LTD` (lower) and `theta_LTP`
(higher) are what produce the two-region BCM-like curve.

## Dendritic current definitions (Methods 4.3)

Apical current (Eq. 6), saturating at a minimum negative plateau:

```
Ia = max( I_teach - I_SST,  -I_SAT )
```

Basal current (Eq. 7):

```
Ib = I_in + I_PYR - I_PV
```

where:
- `I_teach`: EPSC from apical AMPA synapses carrying the top-down teacher/prediction.
- `I_SST`: IPSC from SST onto the apical path (the gate). Disinhibition drives this to 0.
- `I_SAT`: minimum negative plateau the apical circuit can transmit (clamp from the
  half-wave rectifier / saturation).
- `I_in`: total weighted sum over the plastic NMDA input synapses (bottom-up input).
- `I_PYR`: EPSC from recurrent PYR-PYR synapses.
- `I_PV`: IPSC from recurrent PV inhibition.

### Reduction to the classical delta rule

Two conditions make the rule exactly the textbook delta rule:
1. **Balanced regime**: `I_PYR = I_PV` (recurrent excitation cancels recurrent inhibition).
2. **Disinhibition active** (VIP attention cue on): `I_SST = 0`.

Then:

```
Ia - Ib  ->  I_teach - I_in
```

i.e. error = teacher minus input, the classical Widrow-Hoff delta. This is why the
authors can call it a "gradient-like" rule while keeping it fully local and spike-driven:
the network's own E/I balance and disinhibition gate do the work of isolating the
teacher-input difference.

## Practical notes for re-implementation

- Everything is current-mode and continuous-time. If you reimplement in software
  (e.g. a SNN simulator like NEST, which the acknowledgments mention, or in JAX/Brian2),
  the natural form is the set of leaky ODEs above plus the discrete `dw_i = delta(t_i)`
  applied at presynaptic spike events.
- The weight is **4-bit quantized** for storage even though the on-capacitor accumulation
  is analog/continuous (see `hardware-circuits.md`). Both the software model and the
  hardware discretize to 4 bits. If you skip quantization you will not reproduce the
  paper's robustness story, since the 4-bit + pull-to-midpoint scheme is part of the
  noise tolerance.
- `theta_LTP` window sits at higher rates than `theta_LTD`. Picking these windows is the
  main "hyperparameter" of the rule, alongside `eta`, `beta`, the tolerance `I_eps`
  (hardware), and the three time constants `tau_soma`, `tau_basal`, `tau_apical`,
  `tau_ca`.
