# Reproducing Figure 2 (behavioral model) — in NEST

This reproduces Fig. 2 of *"A canonical cortical electronic circuit for
neuromorphic intelligence"* (Maryada, De Luca et al., bioRxiv 2025.03.28.646019)
**in NEST, on top of this repo's dynaple model** — not as a standalone NumPy
model. The mechanism (multi-compartment AdExp neuron + plastic three-factor delta
synapse) is the same one validated in `experiments/03_learning.py`.

```bash
python experiments/fig2.py            # all panels (2c is slow, ~10-20 min)
python experiments/fig2.py 2b 2d      # selected panels
```

Outputs: `figures/fig2a.png … fig2d.png` plus `figures/fig2_rates.png`
for the trained classifier readout trace. Code: `biodl/network.py`
(`ClassifierNetwork`, `Fig2Config`) + `experiments/fig2.py`.

## This is a qualitative reproduction

The paper is a preprint and does not publish its supplementary parameters, so a
pixel-for-pixel match is impossible. The goal is the same *mechanisms and trends*.
**Every parameter not given by the paper is a GUESS**, collected in
`biodl.network.Fig2Config` and listed below.

### Parameters guessed (calibrated for mechanism, not to match the figure)

| Param | Value | Why this value |
|---|---|---|
| `n_input` N | 16 | paper N=64; downscaled for laptop NEST runs |
| `n_pyr` per column | 4 | paper 64; downscaled |
| `plastic_ibias` | 0.3 | **dry-run calibrated**: keeps PYR sub-saturation so the teacher gates calcium and inference rate tracks weights |
| `teacher_ibias` | 300 | apical teacher (effective_bias scale; cf. 03_learning) |
| `attn_ibias` | 3.0 | attention basal drive → three-level Ca separation |
| `ltp_window` | (7500, 9000) | **calibrated** to match-train Ca (~8600) |
| `ltd_window` | (3800, 6000) | **calibrated** to non-match-train Ca (~4800) |
| `eta`, `eta_l` | 0.2 | learning rate |
| `rate_active/inactive` | 50 / 0 Hz | Active rate from the paper; inactive rate follows Chiara's `config_overlapping.yaml` (`inp_L: 0.`). The old 5 Hz background potentiated inactive synapses during teacher phases. |
| panel 2d size | `N=32`, `n_pyr=16` per class | Larger than the fast 2c sweep so the heatmaps read like the paper panel. |
| panel 2d device mismatch | 20% | Fabrication/device-style mismatch on neuron/synapse parameters, not class overlap. In the 2d panel, the class overlap is the shared input rows 12–19. |
| mismatch seeds | 3 | paper averages several; downscaled |
| epochs | 8 | paper "~10" |

The calcium windows were placed by a **dry run** (the task's prescribed method):
measured Ca for inference (~2400), non-match training (attention only, ~4800),
and match training (attention+teacher, ~8600), then set the windows so inference
sits below both (frozen), non-match in LTD (depress), match in LTP (potentiate).

## Honest status — what reproduced and what did not

**The single biggest deviation (architecture).** We first wired the *full*
PV/SST/VIP motif per column. But learning needs `effective_bias=True` so the
calcium proxy reaches the windows, and in that regime the motif's inhibition plus
the high apical gain **saturate PYR at ~370 Hz**, which pins the calcium *above*
the window (no learning) and pins the firing rate (no classification). So the
**classifier (2c/2d) uses bare PYR-per-column** — the proven learning core — with
**attention emulated as a tonic basal drive** that gates train vs inference (its
biological role is the VIP→SST→PYR disinhibition). The real motif dynamics are
shown separately in **2b**.

| Panel | Status |
|---|---|
| **2a** connectivity schematic | ✅ drawn programmatically (two columns, shared input, per-column teachers, shared attention) |
| **2b** rates + calcium over the gate | ✅ attention ON → VIP↑, SST↓, PYR↑; θ (low-pass of PYR spikes) tracks the rate. Reuses the single-column motif. θ is computed in Python (the in-NEST `Isoma_ca` needs `effective_bias=True`, which destabilises the gate weights). |
| **2c** accuracy vs epoch × mismatch | ✅ **mechanism works** — accuracy rises above chance and the mismatch curves stay clustered (graceful degradation). This is the headline robustness result. |
| **2d** weight matrices | ✅ paper-style layout: Input / Initial / Final / Test / Ideal, with A and B output blocks in the same heatmap. The NEST-trained matrix uses random 4-bit init and 20% device mismatch; repeated inference is read-only so the Test matrix stays stable. |
| **rates / 2e** PYR readout rates | ✅ after training, an extended read-only A/B input sequence with no-input gaps shows A/B PYR rates crossing with class changes and dropping when neither class is active. |

### Sanity check (delta rule reduces to classical delta)

With a single bare column, no recurrence, attention+teacher on, the error reduces
to `Ia − Ib = I_teach − I_in` (the apical teacher dominates the weak basal input),
i.e. the classical Widrow-Hoff delta rule — verified in the dry-run calcium
separation (teacher present → LTP, absent → LTD).

### Verdict on the suspect overlap formula

The paper's active-synapse count `N_active = 0.5·(1 + 0.5·ov)·N` is, taken
literally, a **count that grows with overlap** (0.5 N at ov=0, 0.75 N at ov=1) —
it does **not** express "fraction of shared active synapses," and conflates total
active count with overlap. We did not use it directly: we define the two patterns
with an explicit shared region (A = inputs 0–9, B = 6–15, shared = 6–9), which is
clearer and controllable. If forced to use the paper's variable, we'd write
`shared_active = ov · base_active` with a fixed `base_active`, keeping total
active count constant so overlap (not total drive) is the only thing varied.

### Device mismatch (panel 2c)

Multiplicative **log-normal** mismatch with coefficient of variation = m is
injected per element into the listed parameters (`_inject_mismatch` in
`network.py`): per-neuron `Isoma_th`, `t_ref`, `Isoma_dpi_tau`, `Isoma_ca_tau`,
and per-synapse `Ibias`. Lognormal keeps values positive and CV=m. 20% is the
paper's worst-case fabrication corner. 2c averages over `SEEDS` random draws.
