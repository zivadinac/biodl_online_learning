"""Two-column classifier for reproducing Fig. 2 (behavioral model) in NEST.

Each column is a population of PYR neurons (the dynaple multi-compartment model)
sharing an N-dimensional bottom-up input on **plastic NMDA basal synapses**, with
a **distinct per-column apical teacher**. Training presents a pattern to both
columns and switches the *matching* column's teacher on: that column's calcium
proxy rises into the LTP window (potentiation), while the non-matching column
(pattern but no teacher) sits in the LTD window (depression) -- exactly the
paper's teacher-gated three-factor delta rule. Readout = the column whose PYR
population fires faster.

ARCHITECTURE NOTE (honest deviation). We tried wiring the *full* PV/SST/VIP motif
per column, but under ``effective_bias=True`` (required so the calcium proxy
reaches the learning windows) the motif's inhibition + the high apical gain
saturate PYR at ~370 Hz, which pins the calcium above the window (no learning)
and pins the firing rate (no classification). So the classifier uses the *bare*
PYR-per-column form -- the proven learning core (cf. experiments/03_learning.py).
The PV/SST/VIP disinhibition dynamics are shown separately (the 2b panel reuses
the single-column motif from biodl/microcircuit.py). Everything below the dashed
line in Fig2Config is a GUESS not given by the paper.

The teacher-on (attention/train) vs teacher-off (inference) switch is the same
gate the disinhibition motif implements biologically; here it is applied directly
as the train/inference control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nest
import numpy as np

from biodl.microcircuit import Microcircuit
from biodl.nest_setup import SYNAPSE_MODEL, install_dynaple

_PLASTIC = "fig2_plastic_syn"
_STATIC = "fig2_static_syn"


@dataclass
class Fig2Config:
    """Knobs for the Fig. 2 reproduction. Items below the line are GUESSes (not
    given by the paper; chosen for mechanism, calibrated from dry runs)."""

    n_input: int = 16  # paper N=64; downscaled for laptop NEST runs (GUESS)
    n_pyr: int = 4  # PYR per column (paper 64) (GUESS)
    rate_active: float = 50.0  # Hz, from the paper
    # Chiara's config_overlapping.yaml uses inp_L=0. Background inactive spikes
    # otherwise potentiate "inactive" synapses during teacher phases.
    rate_inactive: float = 0.0

    # ---- GUESS (calibrated against the calcium values the neurons reach) ----
    plastic_ibias: float = 0.3  # weak input bias: keeps PYR sub-saturation so the
    #                              teacher can gate calcium and inference rate tracks weights
    teacher_ibias: float = 300.0  # apical teacher strength (3-bit static)
    teacher_rate: float = 200.0  # Hz
    attn_ibias: float = 3.0  # attention basal drive (emulates disinhibition gate)
    attn_rate: float = 200.0  # Hz
    eta: float = 0.2  # learning rate (LTP)
    eta_l: float = 0.2  # learning rate (LTD)
    w0: int = 7  # initial 4-bit weight (mid)
    # calcium LTP/LTD windows, calibrated from a dry run to the three-level
    # separation: inference (~2400, below both) < non-match train (attn only, ~4800,
    # LTD) < match train (attn+teacher, ~8600, LTP). Overrides config k_ca_th_*.
    ltp_window: tuple = (7500.0, 9000.0)
    ltd_window: tuple = (3800.0, 6000.0)
    t_present: float = 800.0  # ms per training presentation
    t_infer: float = 1000.0  # ms per inference presentation
    seed: int = 1


class ClassifierNetwork:
    """Two PYR columns A/B sharing an N-dim plastic input, with per-column teachers."""

    COLUMNS = ("A", "B")

    def __init__(self, cfg: Fig2Config | None = None, mismatch: float = 0.0):
        self.cfg = cfg or Fig2Config()
        self.mismatch = float(mismatch)
        self.pyr: dict[str, object] = {}
        self.teacher: dict[str, object] = {}
        self.plastic_conn: dict[str, object] = {}
        self.sr: dict[str, object] = {}
        self.input_gen = None
        self.parrots = None
        self.rt = None
        self.attention = None

    # -- build ------------------------------------------------------------
    def build(self) -> "ClassifierNetwork":
        install_dynaple()
        c = self.cfg
        for name, plastic in ((_PLASTIC, True), (_STATIC, False)):
            if name not in nest.synapse_models:
                nest.CopyModel(
                    SYNAPSE_MODEL,
                    name,
                    {
                        "plastic": plastic,
                        "binarize": False if plastic else True,
                        "n_bit": 4 if plastic else 3,
                        "eta": c.eta if plastic else 0.0,
                        "eta_L": c.eta_l if plastic else 0.0,
                        "delay": 0.1,
                    },
                )

        win = {
            "k_ca_th_L_plus": c.ltp_window[0],
            "k_ca_th_H_plus": c.ltp_window[1],
            "k_ca_th_L_minus": c.ltd_window[0],
            "k_ca_th_H_minus": c.ltd_window[1],
        }
        # bare Microcircuit columns: no PV/SST/VIP (absent) and NO recurrent
        # PYR->PYR (connect_recurrent=False) -- recurrence explodes Ibasal and
        # flips the learning sign for these independent classifier columns.
        pyr_params = {"effective_bias": True, **win}
        for name in self.COLUMNS:
            col = Microcircuit(
                n_pyr=c.n_pyr, n_pv=0, n_sst=0, n_vip=0,
                pyr_params=pyr_params, connect_recurrent=False,
            ).build()
            self.pyr[name] = col.pop["pyr"]
        self.rt = self.pyr["A"][0].get("receptor_types")

        # shared N-dim input -> every PYR of both columns (plastic NMDA)
        self.input_gen = nest.Create(
            "poisson_generator", c.n_input, {"rate": c.rate_inactive}
        )
        self.parrots = nest.Create("parrot_neuron", c.n_input)
        nest.Connect(self.input_gen, self.parrots, "one_to_one", {"delay": 0.1})
        for name in self.COLUMNS:
            nest.Connect(
                self.parrots,
                self.pyr[name],
                "all_to_all",
                {
                    "synapse_model": _PLASTIC,
                    "receptor_type": int(self.rt["NMDA_BASAL_SPIKES"]),
                    "w": c.w0,
                    "Ibias": c.plastic_ibias,
                },
            )
        for name in self.COLUMNS:
            self.plastic_conn[name] = self._column_plastic_connections(name)

        # per-column apical teacher (static AMPA)
        for name in self.COLUMNS:
            g = nest.Create("poisson_generator", 1, {"rate": 0.0})
            nest.Connect(
                g,
                self.pyr[name],
                "all_to_all",
                {
                    "synapse_model": _STATIC,
                    "receptor_type": int(self.rt["AMPA_APICAL_SPIKES"]),
                    "w": 7,
                    "Ibias": c.teacher_ibias,
                },
            )
            self.teacher[name] = g
            sr = nest.Create("spike_recorder")
            nest.Connect(self.pyr[name], sr)
            self.sr[name] = sr

        # shared attention drive (basal) -> both columns; the train/inference gate.
        # On in training it lifts PYR into the LTD/LTP calcium regime; off in
        # inference PYR sits below both windows so weights are frozen.
        self.attention = nest.Create("poisson_generator", 1, {"rate": 0.0})
        for name in self.COLUMNS:
            nest.Connect(
                self.attention,
                self.pyr[name],
                "all_to_all",
                {
                    "synapse_model": _STATIC,
                    "receptor_type": int(self.rt["AMPA_BASAL_SPIKES"]),
                    "w": 7,
                    "Ibias": c.attn_ibias,
                },
            )

        if self.mismatch > 0:
            self._inject_mismatch()
        return self

    def _column_plastic_connections(self, column: str):
        return nest.GetConnections(
            source=self.parrots, target=self.pyr[column], synapse_model=_PLASTIC
        )

    def _set_input_plasticity(self, enabled: bool) -> None:
        for name in self.COLUMNS:
            self._column_plastic_connections(name).set({"plastic": bool(enabled)})

    def randomize_input_weights(self, seed: int | None = None) -> "ClassifierNetwork":
        """Random 4-bit initial weights on the plastic input synapses."""
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        for name in self.COLUMNS:
            conns = self._column_plastic_connections(name)
            conns.set([{"w": int(v)} for v in rng.integers(0, 16, size=len(conns))])
        return self

    def _inject_mismatch(self) -> None:
        """Multiplicative lognormal mismatch (CV=self.mismatch) on per-neuron
        threshold/refractory/time-constants and per-synapse bias current."""
        rng = np.random.default_rng(self.cfg.seed + int(self.mismatch * 1000))
        sigma = np.sqrt(np.log(1 + self.mismatch**2))

        def jit(n):
            return rng.lognormal(-0.5 * sigma * sigma, sigma, size=n)

        keys = ["Isoma_th", "t_ref", "Isoma_dpi_tau", "Isoma_ca_tau"]
        for name in self.COLUMNS:
            pop = self.pyr[name]
            base = pop.get(keys)
            for k in keys:
                nominal = np.atleast_1d(base[k]).astype(float)
                pop.set([{k: float(v * j)} for v, j in zip(nominal, jit(len(pop)))])
        for name in self.COLUMNS:
            conns = self._column_plastic_connections(name)
            ib = np.atleast_1d(conns.get("Ibias")).astype(float)
            conns.set([{"Ibias": float(v * j)} for v, j in zip(ib, jit(len(ib)))])

    # -- protocol ---------------------------------------------------------
    def set_pattern(self, active) -> None:
        active = set(int(i) for i in active)
        self.input_gen.set(
            [
                {
                    "rate": self.cfg.rate_active
                    if i in active
                    else self.cfg.rate_inactive
                }
                for i in range(self.cfg.n_input)
            ]
        )

    def set_teacher(self, column: str | None) -> None:
        for name in self.COLUMNS:
            self.teacher[name].set(
                {"rate": self.cfg.teacher_rate if name == column else 0.0}
            )

    def set_attention(self, on: bool) -> None:
        self.attention.set({"rate": self.cfg.attn_rate if on else 0.0})

    def present(self, active, teacher: str | None, t: float) -> None:
        """One training presentation: attention ON (gate open), teacher on the
        matching column."""
        self.set_pattern(active)
        self.set_teacher(teacher)
        self.set_attention(True)
        self._set_input_plasticity(True)
        nest.Simulate(t)

    # -- readout ----------------------------------------------------------
    def infer_rates(self, active, t: float | None = None) -> dict[str, float]:
        t = t if t is not None else self.cfg.t_infer
        self.set_pattern(active)
        self.set_teacher(None)
        self.set_attention(False)  # gate closed -> PYR below windows -> weights frozen
        for name in self.COLUMNS:
            self.sr[name].set({"n_events": 0})
        self._set_input_plasticity(False)
        try:
            nest.Simulate(t)
        finally:
            self._set_input_plasticity(True)
        return {
            name: 1e3 * self.sr[name].get("n_events") / (len(self.pyr[name]) * t)
            for name in self.COLUMNS
        }

    def predict(self, active, t: float | None = None) -> str:
        r = self.infer_rates(active, t)
        return "A" if r["A"] >= r["B"] else "B"

    def weight_matrix(self, column: str) -> np.ndarray:
        """input(N) x PYR weight matrix for a column (4-bit w values)."""
        conns = self._column_plastic_connections(column)
        w = np.asarray(conns.get("w"), dtype=float)
        src = np.asarray(conns.get("source"))
        tgt = np.asarray(conns.get("target"))
        in_ids, pyr_ids = sorted(set(src)), sorted(set(tgt))
        ix = {v: i for i, v in enumerate(in_ids)}
        jx = {v: j for j, v in enumerate(pyr_ids)}
        mat = np.zeros((len(in_ids), len(pyr_ids)))
        for s, tg, val in zip(src, tgt, w):
            mat[ix[s], jx[tg]] = val
        return mat
