# Architecture: the canonical microcircuit motif

Maps to Fig. 1a, Section 2.1, and Methods 4.1-4.2 of the paper.

## Populations

One excitatory class and three inhibitory classes, the standard cortical "canonical
microcircuit" cell-type set:

| Class | Role | Compartments used |
|-------|------|-------------------|
| **PYR** (pyramidal, excitatory) | Carries the representation; recurrently coupled; receives bottom-up input and top-down teacher. | soma + basal + apical |
| **PV** (parvalbumin, inhibitory) | Feed-forward + recurrent inhibition. Sets E/I balance and the soft winner-take-all. | basal + soma |
| **SST** (somatostatin, inhibitory) | Gates the PYR apical dendrite. When active, strongly inhibits apical -> soma current flow. | basal + soma |
| **VIP** (vasoactive intestinal peptide, inhibitory) | Driven by a top-down "attention"/"context" cue. Disinhibits SST (suppresses it), un-gating the PYR apical path. | basal + soma |

All four classes share the same neuron model. Only PYR uses all three compartments;
interneurons use the basal compartment for all incoming connections and the soma for
dynamics.

## PYR compartments

- **Soma**: AdExp-IF dynamics. Integrates basal + apical currents and fires.
- **Basal dendrite**: receives the *bottom-up sensory input* via **plastic NMDA**
  synapses, plus recurrent PYR-PYR excitation and PV inhibition. This is where learning
  happens (the plastic synapses live here).
- **Apical dendrite**: receives the *top-down* signal via **static AMPA** synapses. This
  top-down input is the "teacher" (supervised) or "prediction / prediction error"
  (predictive coding) signal. The apical compartment applies a **half-wave rectification
  nonlinearity**: current only passes if it exceeds a threshold (see SM 0.1 in the paper).

The functional point: the soma *sums* basal and apical currents (drives firing), while
the learning block *subtracts* them (drives the error). So the two currents must be
copied into separate physical pathways. This split is both a modeling choice and a
hardware necessity.

## Synapse types (four, each with a CMOS implementation)

| Type | Kinetics | Sign | Notes |
|------|----------|------|-------|
| NMDA | slow | excitatory | the **plastic** bottom-up input synapses on the basal dendrite |
| AMPA | fast (alpha) | excitatory | **static** top-down teacher synapses on the apical dendrite |
| GABA_B | - | inhibitory, **subtractive** | standard subtractive inhibition |
| GABA_A | - | inhibitory, **subtractive + shunting** | adds shunting (divisive-ish) effect |

All four are implemented with DPI (differential pair integrator) current-mode synapse
circuits.

## Connectivity (Fig. 1a)

- **PYR -> PYR**: recurrent excitation (drives attractor / sustained activity).
- **PYR -> PV**: excites PV.
- **PV -> PYR**: feedback + feed-forward inhibition (E/I balance, soft WTA).
- **PYR -> SST**: PYR excites SST.
- **VIP -> SST**: VIP inhibits SST (this is the disinhibition link).
- **SST -> PYR apical**: SST inhibits the apical-to-soma current path (the gate).
- **Input -> PYR basal** via plastic NMDA.
- **Teacher (top-down) -> PYR apical** via static AMPA.
- **Attention/context cue (top-down) -> VIP**.

## The disinhibition gate (the learning on/off switch)

This VIP -> SST -> PYR-apical motif is the core control mechanism, and it doubles as the
training/inference selector:

- **Attention cue ON** (training mode):
  VIP fires -> VIP suppresses SST -> SST stops clamping the apical path ->
  apical (teacher) current reaches the soma -> PYR firing rate rises into the LTP/LTD
  calcium windows -> learning is enabled. In the current notation, `I_SST -> 0`, so the
  apical current `Ia` is dominated by the teacher.

- **Attention cue OFF** (inference mode):
  VIP quiet -> SST active -> SST clamps apical path -> PYR firing drops *below* both
  calcium windows -> `theta(t)` outside LTP and LTD regions -> no weight updates. The
  network computes but cannot learn. This freezes the learned weights and is the paper's
  answer to **catastrophic forgetting**: you only open the learning window when a
  top-down label/context says to.

The same cue is what makes the rule behave like a clean gradient-style delta rule (see
`neuron-and-learning.md`): with VIP-driven disinhibition `I_SST = 0`, and in the balanced
regime `I_PYR = I_PV`, the error `(Ia - Ib)` collapses to `(I_teach - I_in)`.

## Configurable operating modes

The same motif (without enabling plasticity) can act as:

- **Attractor / working-memory network**: recurrent PYR-PYR excitation balanced by
  PYR-PV feedback gives a response function with three fixed points; the nonzero stable
  fixed point sustains an ensemble's activity after the input is removed
  (Fig. 1d, stable point ~20 Hz). See `experiments.md` for how the f-f curve is measured.
- **Soft winner-take-all** for decision making (PV-mediated competition).
- **Intrinsic oscillator**.
- **Encoding layer** for amplifying / filtering / compressing high-dynamic-range inputs.

With plasticity enabled it additionally: forms robust attractors, classifies into
multiple classes, and recognizes spatio-temporal patterns.

## Robustness rationale (why populations)

Analog CMOS device mismatch typically yields CV ~ 0.2 per element. Uncoupled population
averaging gives the classical `1/sqrt(N)` standard-deviation reduction (CLT). The paper
leans on newer results showing that in **recurrent E/I balanced** networks the variance
can scale better, as `1/N`, and extends that with the local dendritic learning rule. Net
effect: the motif is designed so that mismatch across every parameter (weights,
threshold, time constants, adaptation, refractory period) averages out at the population
level rather than corrupting the computation.
