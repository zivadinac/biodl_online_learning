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
import numpy as np

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

# (w, Ibias) per edge. w is the integer 4-bit weight [0,15], Ibias the bias
# current it scales; postsynaptic drive tracks the product w*Ibias. These are
# calibrated for the *1-neuron-per-type* column: with no population to sum over,
# a single synapse must carry the full drive, so Ibias is far larger than the
# per-synapse population weights in config_overlapping.yaml. The apical (teacher)
# port is ~10x more potent than the basal ports, hence its smaller Ibias.
# biodl/weights.py will formalise the float-config -> (w, Ibias) mapping.
# Static (non-plastic) synapses are 3-bit, so w in [0,7]; the postsynaptic drive
# is set by the product w*Ibias (n_bit only clips the max weight, it does not
# rescale). Every static edge uses w=7 (full 3-bit) and tunes strength via Ibias.
DEFAULT_WEIGHTS = {
    "input_pyr": (
        7,
        4300,
    ),  # bottom-up input -> PYR basal (NMDA, plastic in C -> 4-bit)
    "input_pv": (7, 2150),  # bottom-up input -> PV
    "teacher_pyr": (7, 8600),  # top-down teacher -> PYR apical (AMPA), drives the
    #                            rectified apical (see params.NEGATE / Iapical_low)
    "cue_vip": (7, 4300),  # attention cue -> VIP
    "pyr_pv": (7, 2600),  # PYR -> PV
    "pyr_sst": (7, 2600),  # PYR -> SST
    "pv_pyr": (7, 2150),  # PV -| PYR (GABA_A on basal)
    "sst_pyr": (7, 3400),  # SST -| PYR apical (the gate, GABA_B)
    "vip_sst": (7, 5400),  # VIP -| SST (disinhibition, GABA_B)
    "pyr_pyr": (6, 1000),  # recurrent excitation (size>=2)
    "pv_pv": (6, 1000),  # recurrent inhibition (size>=2)
}

# Canonical-motif edges, the single source of truth for both the builder and the
# connectivity graph. Each: (src, dst, weight_key, receptor, sign).
# sign is "+" excitatory / "-" inhibitory (determined by the receptor; kept for plots).
MOTIF_EDGES = [
    ("pyr", "pv", "pyr_pv", "ampa_basal", "+"),
    ("pyr", "sst", "pyr_sst", "ampa_basal", "+"),
    ("pv", "pyr", "pv_pyr", "gaba_a_basal", "-"),
    ("sst", "pyr", "sst_pyr", "gaba_b_apical", "-"),  # the gate
    ("vip", "sst", "vip_sst", "gaba_b_basal", "-"),  # disinhibition
]

# Recurrent edges — present only at population size >= 2 (no autapses).
RECURRENT_EDGES = [
    ("pyr", "pyr", "pyr_pyr", "ampa_basal", "+"),
    ("pv", "pv", "pv_pv", "gaba_a_basal", "-"),
]

# External Poisson drives: (source, dst, weight_key, receptor, sign).
DRIVE_EDGES = [
    ("input", "pyr", "input_pyr", "nmda_basal", "+"),  # plastic in learning expts
    ("input", "pv", "input_pv", "ampa_basal", "+"),
    ("teacher", "pyr", "teacher_pyr", "ampa_apical", "+"),
    ("cue", "vip", "cue_vip", "ampa_basal", "+"),
]

_STATIC = "biodl_static_syn"
_PLASTIC = "biodl_plastic_syn"

DEFAULT_RECORD = [
    "Isoma_mem",
    "Isoma_ca",
    "Iapical",
    "Ibasal",
    "sigma_plus",
    "sigma_minus",
]


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
        pyr_params: dict | None = None,
        connect_recurrent: bool = True,
    ):
        self.sizes = {"pyr": n_pyr, "pv": n_pv, "sst": n_sst, "vip": n_vip}
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.plastic_input = plastic_input
        self.pyr_params = pyr_params or {}
        self.connect_recurrent = connect_recurrent
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
            nest.CopyModel(
                SYNAPSE_MODEL,
                _STATIC,
                {
                    "plastic": False,
                    "binarize": True,
                    "n_bit": 3,  # static synapses: 3-bit
                    "eta": 0.0,
                    "eta_L": 0.0,
                    "delay": self.delay,
                },
            )
        if _PLASTIC not in existing:
            syn = synapse_config()
            nest.CopyModel(
                SYNAPSE_MODEL,
                _PLASTIC,
                {
                    "plastic": True,
                    "binarize": bool(syn.get("binarize", True)),
                    "n_bit": 4,  # plastic synapses: 4-bit
                    "eta": float(syn.get("eta", 0.0)),
                    "eta_L": float(syn.get("eta_L", 0.0)),
                    "delay": self.delay,
                },
            )

    def _create_populations(self) -> None:
        for key, n in self.sizes.items():
            if n <= 0:  # absent population (e.g. a bare-PYR classifier column)
                continue
            pop = nest.Create(NEURON_MODEL, n)
            params = neuron_params(_CFG_TYPE[key])
            if key == "pyr":
                params.update(self.pyr_params)
            pop.set(params)
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
        nest.Connect(
            self.pop[src], self.pop[tgt], "all_to_all", self._syn(edge, receptor)
        )

    def _connect_populations(self) -> None:
        # inter-population motif (skip edges touching an absent population)
        for src, dst, edge, receptor, _sign in MOTIF_EDGES:
            if src in self.pop and dst in self.pop:
                self._connect_pop(src, dst, edge, receptor)
        # recurrent, no autapses (only present at population size >= 2)
        for src, dst, edge, receptor, _sign in RECURRENT_EDGES:
            if self.connect_recurrent and src in self.pop and self.sizes[src] >= 2:
                nest.Connect(
                    self.pop[src],
                    self.pop[dst],
                    {"rule": "all_to_all", "allow_autapses": False},
                    self._syn(edge, receptor),
                )

    # -- external drive ---------------------------------------------------
    def drive(
        self, input_rate: float = 0.0, teacher_rate: float = 0.0, cue_rate: float = 0.0
    ) -> "Microcircuit":
        """Attach the three external drives as *constant-rate* Poisson processes."""
        for name, rate in (
            ("input", input_rate),
            ("teacher", teacher_rate),
            ("cue", cue_rate),
        ):
            self.gen[name] = nest.Create("poisson_generator", 1, {"rate": float(rate)})
        self._connect_drives()
        return self

    def drive_profiles(
        self, times, input=None, teacher=None, cue=None
    ) -> "Microcircuit":
        """Attach the three drives as *time-varying* Poisson processes.

        ``times`` is a 1-D vector of times (ms) at which each rate takes effect
        (piecewise-constant: value ``i`` holds from ``times[i]`` until ``times[i+1]``).
        Each of ``input`` / ``teacher`` / ``cue`` is, independently:

        * a 1-D rate vector (Hz), same length as ``times`` — time-varying drive;
        * a scalar — constant rate;
        * a callable ``rate(t_ms) -> Hz`` — sampled at ``times``;
        * ``None`` — silent.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        for name, profile in (("input", input), ("teacher", teacher), ("cue", cue)):
            self.gen[name] = self._poisson_from_profile(times, profile)
        self._connect_drives()
        return self

    def _poisson_from_profile(
        self, times: np.ndarray, profile
    ) -> "nest.NodeCollection":
        if profile is None:
            return nest.Create("poisson_generator", 1, {"rate": 0.0})
        if callable(profile):
            values = np.asarray([float(profile(t)) for t in times], dtype=float)
        elif np.isscalar(profile):
            return nest.Create("poisson_generator", 1, {"rate": float(profile)})
        else:
            values = np.asarray(profile, dtype=float)
        if values.shape != times.shape:
            raise ValueError(
                f"rate vector length {values.shape} != times length {times.shape}"
            )
        # inhomogeneous_poisson_generator needs strictly-increasing rate_times on
        # the simulation grid and > 0; nudge the first point off 0 if needed.
        res = nest.GetKernelStatus("resolution")
        rt = np.round(times / res) * res
        rt[0] = max(rt[0], res)
        for i in range(1, len(rt)):  # enforce strictly increasing on the grid
            if rt[i] <= rt[i - 1]:
                rt[i] = rt[i - 1] + res
        return nest.Create(
            "inhomogeneous_poisson_generator",
            1,
            {"rate_times": rt.tolist(), "rate_values": values.tolist()},
        )

    def _connect_drives(self) -> None:
        for src, dst, edge, receptor, _sign in DRIVE_EDGES:
            if dst not in self.pop:
                continue
            model = (
                _PLASTIC if (edge == "input_pyr" and self.plastic_input) else _STATIC
            )
            nest.Connect(
                self.gen[src],
                self.pop[dst],
                "all_to_all",
                self._syn(edge, receptor, model=model),
            )

    def tonic(self, w: int = 7, ibias: float = 2600.0, **rates) -> "Microcircuit":
        """Add a constant baseline excitatory (AMPA_BASAL) drive to populations.

        In a 1-neuron-per-type column a cell lacks the upstream population it would
        be driven by in the full network. ``tonic(sst=200)`` gives SST a standing
        drive so it is tonically active and can then be gated by VIP — the
        disinhibition gate needs SST active by default. ``rates`` maps a population
        key (``"pyr"``/``"pv"``/``"sst"``/``"vip"``) to a rate in Hz.
        """
        for key, rate in rates.items():
            g = nest.Create("poisson_generator", 1, {"rate": float(rate)})
            self.gen[f"tonic_{key}"] = g
            nest.Connect(
                g,
                self.pop[key],
                "all_to_all",
                {
                    "synapse_model": _STATIC,
                    "receptor_type": int(self.rt[PORT["ampa_basal"]]),
                    "w": int(w),
                    "Ibias": float(ibias),
                },
            )
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
