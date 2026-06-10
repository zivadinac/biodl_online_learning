"""Build the canonical cortical microcircuit on the Dynap-LE NEST model.

One excitatory population (PYR) and three inhibitory classes (PV, SST, VIP), all
the same multi-compartment ``dynaple_neur`` model, wired into the canonical motif
(see ``texts/architecture.md``). Parameterised by population size: instantiate at
one neuron per type for the minimal column, scale up by changing the sizes.

Recurrent edges (PYR->PYR, PV->PV) use ``allow_autapses=False`` general
connectivity, so at one neuron per type they are simply absent.
"""

from __future__ import annotations

import nest

from biodl.config import synapse_config
from biodl.nest_setup import NEURON_MODEL, SYNAPSE_MODEL, install_dynaple
from biodl.params import neuron_params

# builder population key -> config_neur.yaml section ("pvc" is PV).
_CFG_TYPE = {"pyr": "pyr", "pv": "pvc", "sst": "sst", "vip": "vip"}

# short edge name -> receptor port on the postsynaptic dynaple_neur.
PORT = {
    "ampa_apical": "AMPA_APICAL_SPIKES",
    "ampa_basal": "AMPA_BASAL_SPIKES",
    "nmda_basal": "NMDA_BASAL_SPIKES",
    "gaba_b_apical": "GABA_B_APICAL_SPIKES",
    "gaba_a_basal": "GABA_A_BASAL_SPIKES",
    "gaba_b_basal": "GABA_B_BASAL_SPIKES",
}

# Starter (w, Ibias) per edge. w is the integer 4-bit weight [0,15], Ibias the
# bias current it scales. These are hand-tuned starting points; biodl/weights.py
# will formalise the float-config-weight -> (w, Ibias) mapping once calibrated.
DEFAULT_WEIGHTS = {
    "input_pyr": (8, 100),    # bottom-up input -> PYR basal (NMDA, plastic in C)
    "input_pv": (10, 100),    # bottom-up input -> PV
    "teacher_pyr": (8, 100),  # top-down teacher -> PYR apical (AMPA)
    "cue_vip": (12, 100),     # attention cue -> VIP
    "pyr_pv": (8, 100),       # PYR -> PV
    "pyr_sst": (8, 100),      # PYR -> SST
    "pv_pyr": (8, 100),       # PV -| PYR (GABA_A on basal)
    "sst_pyr": (10, 100),     # SST -| PYR apical (the gate, GABA_B)
    "vip_sst": (12, 100),     # VIP -| SST (disinhibition, GABA_B)
    "pyr_pyr": (4, 100),      # recurrent excitation (size>=2)
    "pv_pv": (4, 100),        # recurrent inhibition (size>=2)
}

_STATIC = "biodl_static_syn"
_PLASTIC = "biodl_plastic_syn"

DEFAULT_RECORD = ["Isoma_mem", "Isoma_ca", "Iapical", "Ibasal", "sigma_plus", "sigma_minus"]


class Microcircuit:
    """A canonical-microcircuit column, parameterised by population size."""

    def __init__(
        self,
        n_pyr: int = 1,
        n_pv: int = 1,
        n_sst: int = 1,
        n_vip: int = 1,
        weights: dict | None = None,
        plastic_input: bool = False,
    ):
        self.sizes = {"pyr": n_pyr, "pv": n_pv, "sst": n_sst, "vip": n_vip}
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.plastic_input = plastic_input
        self.delay = float(synapse_config().get("delay", 1.0))
        self.pop: dict = {}
        self.gen: dict = {}
        self.sr: dict = {}
        self.mm: dict = {}
        self.rt = None

    # -- build ------------------------------------------------------------
    def build(self) -> "Microcircuit":
        install_dynaple()
        self._define_synapses()
        self._create_populations()
        self._connect_populations()
        return self

    def _define_synapses(self) -> None:
        existing = nest.GetKernelStatus("synapse_models")
        if _STATIC not in existing:
            nest.CopyModel(SYNAPSE_MODEL, _STATIC,
                           {"plastic": False, "binarize": True, "eta": 0.0, "eta_L": 0.0,
                            "delay": self.delay})
        if _PLASTIC not in existing:
            syn = synapse_config()
            nest.CopyModel(SYNAPSE_MODEL, _PLASTIC,
                           {"plastic": True, "binarize": bool(syn.get("binarize", True)),
                            "eta": float(syn.get("eta", 0.0)), "eta_L": float(syn.get("eta_L", 0.0)),
                            "delay": self.delay})

    def _create_populations(self) -> None:
        for key, n in self.sizes.items():
            pop = nest.Create(NEURON_MODEL, n)
            pop.set(neuron_params(_CFG_TYPE[key]))
            self.pop[key] = pop
        self.rt = self.pop["pyr"][0].get("receptor_types")

    def _syn(self, edge: str, receptor: str, model: str = _STATIC) -> dict:
        w, ibias = self.weights[edge]
        return {
            "synapse_model": model,
            "receptor_type": int(self.rt[PORT[receptor]]),
            "w": int(w),
            "Ibias": float(ibias),
        }

    def _connect_pop(self, src: str, tgt: str, edge: str, receptor: str) -> None:
        nest.Connect(self.pop[src], self.pop[tgt], "all_to_all", self._syn(edge, receptor))

    def _connect_populations(self) -> None:
        # recurrent, no autapses (only present at population size >= 2)
        if self.sizes["pyr"] >= 2:
            nest.Connect(self.pop["pyr"], self.pop["pyr"],
                         {"rule": "all_to_all", "allow_autapses": False},
                         self._syn("pyr_pyr", "ampa_basal"))
        if self.sizes["pv"] >= 2:
            nest.Connect(self.pop["pv"], self.pop["pv"],
                         {"rule": "all_to_all", "allow_autapses": False},
                         self._syn("pv_pv", "gaba_a_basal"))
        # inter-population motif
        self._connect_pop("pyr", "pv", "pyr_pv", "ampa_basal")
        self._connect_pop("pyr", "sst", "pyr_sst", "ampa_basal")
        self._connect_pop("pv", "pyr", "pv_pyr", "gaba_a_basal")
        self._connect_pop("sst", "pyr", "sst_pyr", "gaba_b_apical")  # the gate
        self._connect_pop("vip", "sst", "vip_sst", "gaba_b_basal")   # disinhibition

    # -- external drive ---------------------------------------------------
    def drive(self, input_rate: float = 0.0, teacher_rate: float = 0.0,
              cue_rate: float = 0.0) -> "Microcircuit":
        """Attach the three external Poisson drives: input, teacher, attention cue."""
        self.gen["input"] = nest.Create("poisson_generator", 1, {"rate": float(input_rate)})
        self.gen["teacher"] = nest.Create("poisson_generator", 1, {"rate": float(teacher_rate)})
        self.gen["cue"] = nest.Create("poisson_generator", 1, {"rate": float(cue_rate)})

        in_model = _PLASTIC if self.plastic_input else _STATIC
        nest.Connect(self.gen["input"], self.pop["pyr"], "all_to_all",
                     self._syn("input_pyr", "nmda_basal", model=in_model))
        nest.Connect(self.gen["input"], self.pop["pv"], "all_to_all",
                     self._syn("input_pv", "ampa_basal"))
        nest.Connect(self.gen["teacher"], self.pop["pyr"], "all_to_all",
                     self._syn("teacher_pyr", "ampa_apical"))
        nest.Connect(self.gen["cue"], self.pop["vip"], "all_to_all",
                     self._syn("cue_vip", "ampa_basal"))
        return self

    # -- recording --------------------------------------------------------
    def attach_recorders(self, record_from: list | None = None) -> "Microcircuit":
        record_from = record_from if record_from is not None else DEFAULT_RECORD
        for key, pop in self.pop.items():
            sr = nest.Create("spike_recorder")
            nest.Connect(pop, sr)
            self.sr[key] = sr
            mm = nest.Create("multimeter", {"record_from": record_from})
            nest.Connect(mm, pop)
            self.mm[key] = mm
        return self

    # -- readout ----------------------------------------------------------
    def spikes(self, key: str) -> dict:
        ev = self.sr[key].get("events")
        return {"times": ev["times"], "senders": ev["senders"]}

    def trace(self, key: str) -> dict:
        return dict(self.mm[key].get("events"))

    def rates(self, t_sim: float) -> dict:
        """Mean firing rate (Hz) per population over ``t_sim`` ms."""
        out = {}
        for key, pop in self.pop.items():
            n_spikes = len(self.sr[key].get("events")["times"])
            out[key] = 1e3 * n_spikes / (len(pop) * t_sim)
        return out
