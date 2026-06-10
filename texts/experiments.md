# Experiments and results

Maps to Section 2.1 (attractor), Section 2.2 (classification), Section 2.3 (circuit-level),
and Methods 4.2.

## 1. Attractor dynamics (Fig. 1d, Methods 4.2)

Goal: show the motif forms a stable attractor / working-memory state despite full-parameter
mismatch.

Method (following the approach of Giulioni et al. 2012, ref 82):
- Model two excitatory pools and one shared inhibitory pool.
- Replace the recurrent connections *within* one excitatory population with independent
  external Poisson spike trains, so you can sweep an "effective recurrent input" rate.
- Measure that population's output firing rate as a function of the injected input rate.
- This yields a frequency-frequency (f-f) curve.

Result:
- The f-f curve has **three fixed points** (the classic recurrent-excitation +
  feedback-inhibition response). Two are stable; the nonzero stable fixed point sits at
  about **20 Hz**.
- Working-memory behavior: after an external driving input is removed, the ensemble keeps
  firing at the stable fixed point (sustained activity), confirming attractor memory.
- This holds despite mismatch in all parameters.

## 2. Binary classification of overlapping patterns (Fig. 2, Section 2.2)

The main learning demonstration.

### Task setup

- Two input patterns, each a binary vector of independent Poisson spike trains. Active
  entries fire at **50 Hz**, inactive at **5 Hz** (length-N vectors).
- The classification network is built by **duplicating** the motif into two identical
  columns, "A" and "B". The only difference between columns is the apical teacher input
  ("Teacher A" vs "Teacher B" in Fig. 2a).
- **Overlap** `ov` between the two input vectors = fraction of shared active synapses.
  The number of active synapses for a given input is stated as:

  ```
  N_active = 0.5 * (1 + 0.5 * ov) * N
  ```

  Because active-synapse count grows with overlap, the **total input current is not
  constant** across overlaps. The authors note this makes the task harder for an SNN than
  the usual setup of holding active-synapse count fixed and merely shifting which ones are
  active. The reported figures use **ov = 0.5** (50 percent overlap).

  (Caveat, see README: taken literally this formula gives 0.5 N active at ov = 0 and
  0.75 N at ov = 1; re-derive before reusing.)

### Training procedure

1. A common **attention cue** is sent to **all** VIP cells of **both** columns. This
   disinhibits SST, lets apical current reach the soma, and pushes PYR firing up into the
   LTD/LTP calcium windows. Learning is now enabled (this is the disinhibition gate from
   `architecture.md`).
2. The input pattern is presented to the PYR basal dendrites of both columns.
3. A class-specific **high teacher** signal goes only to the column matching the true
   class (teacher A for an A pattern, etc.).
4. Neurons receiving both input and high teacher are pushed into their **LTP** region:
   their stimulated synapses potentiate.
5. Neurons receiving input but no teacher fall into their **LTD** region: their
   stimulated synapses depress.
6. After ~**10 example presentations**, weights converge: high for true-class inputs, low
   for false-class inputs. The **overlapping (shared) inputs are driven to intermediate
   weights** and end up not contributing to the discrimination, which is exactly what you
   want for the uninformative shared dimensions.
7. Remove the common attention cue: SST reactivates, PYR mean rate drops below both
   windows, and the network is locked into an **inference-only** stable mode. Weights stay
   put across repeated inference (demonstrated in Fig. 2d).

### Weight matrices (Fig. 2d)

Three snapshots per column are shown: random init, end of training, after many inference
repetitions. They match an "ideal" variability-free classifier's weight matrix. Weights
are analog in change but their absolute value is **discretized to 4 bits**, in both the
software simulation and the hardware. Fig. 2b/2d data were generated at **20 percent
device mismatch** as the worst-case corner.

### Robustness result (Fig. 2c)

Accuracy vs training epoch, swept over mismatch = 0 / 10 / 20 / 30 percent, for the
50 percent overlap case. The network climbs above chance and the curves stay close
together across mismatch levels: the take-home is that classification accuracy is
**graceful** under increasing mismatch, up to and including the 20-30 percent range. (See
the figure for exact accuracy values; the preprint text does not tabulate them.)

## 3. Circuit-level (transistor) validation (Fig. 3c, Section 2.3)

Same overlapping-pattern classification, but run in **transistor-level circuit
simulation** instead of the behavioral model.

Scope: a **single neuron per type** (1 PYR per class A/B, 1 PV, 1 SST, 1 VIP), because
SPICE-level network simulation is too slow. Each PYR has **3 plastic synapses** with 4-bit
weights, `W1 W2 W3`.

Stimulus mapping (mirrors the Fig. 2d patterns):
- Pattern **A** drives synapses **W1 and W2**.
- Pattern **B** drives synapses **W2 and W3**.
- **W2 is therefore uninformative** (driven by both patterns), the analog of the
  overlapping inputs.

Timeline:
- Start: all three weights randomly initialized to the same values, both PYR neurons
  respond identically and unspecifically to both patterns (first ~2 presentations).
- t = 5 s: training turned on, teacher added to the class-matching apical compartment.
- As `theta(t)` enters LTD/LTP, weights move per the error sign:
  - PYR "A": **W1 potentiates (LTP), W3 depresses (LTD), W2 -> intermediate.**
  - PYR "B": **W3 potentiates, W1 depresses, W2 -> intermediate.**
- t = 12 s onward: both neurons are correctly tuned to their patterns. The uninformative
  W2 settles to the middle, matching the behavioral-model prediction.

This is the end-to-end consistency check: the same qualitative learning outcome appears
in the abstract model, the hardware-aware behavioral simulation, and the transistor-level
circuit simulation.

## Summary of what each experiment proves

| Experiment | Claim supported | Validation level |
|------------|-----------------|------------------|
| f-f curve (Fig. 1d) | Stable attractor / working memory survives full-parameter mismatch | behavioral model |
| Classification + mismatch sweep (Fig. 2) | Local three-factor delta rule learns separable representations; accuracy graceful to ~20-30% mismatch; disinhibition cleanly gates train vs inference | behavioral model |
| Circuit-level classification (Fig. 3c) | The actual CMOS learning circuits reproduce the learning dynamics | transistor-level SPICE, single neuron per type |
