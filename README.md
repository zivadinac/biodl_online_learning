# biodl_online_learning

Recreating the **canonical cortical microcircuit** from

> Maryada\*, Chiara De Luca\*, A. Rubino, C. Wen, M. Cartiglia, I.-I. Fodorut,
> M. Payvand, G. Indiveri. *"A canonical cortical electronic circuit for
> neuromorphic intelligence."* bioRxiv 2025.03.28.646019. (\* equal contribution.)
> Institute of Neuroinformatics, UZH/ETH Zürich.

as a spiking network in [NEST](https://www.nest-simulator.org/), on top of the
custom **Dynap-LE** neuron/synapse model (`dynaple-model-nest`). Distilled notes
on the paper live in [`texts/`](texts/); the network configuration files we are
reproducing (from Chiara) are [`texts/config_*.yaml`](texts/).

## The idea: build up from a 4-neuron column

The motif is one excitatory population (**PYR**) and three inhibitory classes
(**PV, SST, VIP**). Rather than hardcode the paper's full network (32 PYR, 8 PV,
…), we build a **reusable microcircuit parameterised by population size** and
instantiate it small first — one neuron per type — then scale up by changing a
config.

```
attention cue ──AMPA_BASAL──▶ VIP ──GABA_B_BASAL──▶ SST ──GABA_B_APICAL──▶ PYR.apical   [the gate]
input ──NMDA_BASAL (plastic)──▶ PYR.basal        teacher ──AMPA_APICAL (static)──▶ PYR.apical
PYR ──AMPA_BASAL──▶ PV ──GABA_A_BASAL──▶ PYR.basal      PYR ──AMPA_BASAL──▶ SST
input ──AMPA_BASAL──▶ PV
```

Recurrent edges (PYR→PYR, PV→PV) are written as general connectivity with **no
autapses**, so at one neuron per type they are simply absent and switch on
automatically at population size ≥ 2.

### Milestones

| # | Experiment | Shows |
|---|------------|-------|
| A | `experiments/01_column.py` | the 4-cell column wired & spiking |
| B | `experiments/02_disinhibition.py` | the VIP→SST→PYR-apical disinhibition gate (learning on/off switch) |
| C | `experiments/03_learning.py` | the local three-factor delta rule (static figure, Fig 3c style) |
| C | `experiments/04_learning_live.py` | the same, with interactive sliders |

## Layout

```
biodl/
  config.py        load Chiara's yaml configs
  params.py        map config → dynaple_neur parameters
  microcircuit.py  Microcircuit builder: populations + connectivity, any sizes
  weights.py       float config-weight → integer w[0,15] × Ibias calibration
experiments/       runnable milestone scripts (A → B → C)
texts/             paper PDF + distilled notes + Chiara's configs
scripts/           fix-nest-wheel.sh (NEST PyPI-wheel post-install patcher)
```

## The model

All four cell types are the same multi-compartment AdExp-IF Dynap-LE neuron
(`dynaple_neur__with_dynaple_syn`); only PYR uses all three compartments
(soma + basal + apical). Receptor ports used:

| Port | Role |
|------|------|
| `NMDA_BASAL_SPIKES` | bottom-up input → PYR basal (**plastic**, learning lives here) |
| `AMPA_APICAL_SPIKES` | top-down teacher → PYR apical (static) |
| `AMPA_BASAL_SPIKES` | recurrent / inter-population excitation |
| `GABA_B_APICAL_SPIKES` | SST → PYR apical (the gate) |
| `GABA_A_BASAL_SPIKES` | PV → PYR (soma/basal inhibition) |
| `GABA_B_BASAL_SPIKES` | VIP → SST inhibition |

> **Note on model revision.** Chiara's configs name the model
> `alive_neur`/`delta_syn` (module `delta`); the `dynaple-model-nest` build we
> use names them `dynaple_neur`/`dynaple_syn` (module `dynaple_module`). Same
> circuit, a few renamed parameters (e.g. her `Isoma_ca_w` is `Isoma_ca_bias`
> here). Mappings are handled in `biodl/params.py`.

> **Open calibration.** Config weights are floats (`w_sst_pyr: 1.5`) but the
> NEST synapse takes an integer weight `w ∈ [0,15]` scaled by `Ibias`. Her
> network-builder's float→(w, Ibias) mapping is not in hand, so `biodl/weights.py`
> calibrates it empirically until each population fires in a sensible regime.

## Setup

```bash
make setup     # one-time: system C libs (brew) + patched uv sync
make build     # (re)generate + compile the dynaple NEST module, then patch
```

The `nest-simulator` PyPI wheel is not relocatable
([nest#3762](https://github.com/nest/nest-simulator/issues/3762));
`scripts/fix-nest-wheel.sh` patches it after every `uv sync`. Always use
`make sync`, never a bare `uv sync`.
