"""Single full-motif EMG classifier.

This module provides a one-PYR A-detector wrapped by the full PV/SST/VIP motif
and an sklearn-style client class for binary rest-vs-fist EMG. The public
classifier mirrors :class:`biodl.decoder.NeuromorphicClassifier`, but its readout
is one PYR firing rate: high means rest (class 0), low means fist (class 1).

NEST is intentionally imported only inside the network builder methods. Importing
``SingleMotifClassifier`` from a GUI process stays pure Python/numpy; simulation
work is delegated to the existing ``biodl.decoder`` subprocess worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from biodl.decoder import _get_worker

_RATE_SCALE_PERCENTILE = 75.0
_EPS = 1e-12

_STATIC_SYN = "single_motif_static_syn"
_PLASTIC_SYN = "single_motif_plastic_syn"


@dataclass(frozen=True)
class SingleMotifConfig:
    """Calibration knobs for the single full-motif detector."""

    n_input: int = 8
    seed: int = 1
    epochs: int = 1
    t_train: float = 200.0
    t_infer: float = 300.0
    rate_active: float = 10.0
    teacher_rate: float = 200.0
    attention_rate: float = 400.0
    sst_tonic_rate: float = 200.0
    rate_inactive: float = 0.0
    w0: int = 7
    plastic_ibias: float = 30.0
    eta: float = 0.2
    eta_l: float = 0.2
    teacher_ibias: float = 6000.0
    ltd_low: float = 6500.0
    ltd_high: float = 7800.0
    ltp_low: float = 8000.0
    ltp_high: float = 9000.0
    iapical_low: float = 5000.0
    cue_ibias: float = 6000.0
    sst_tonic_ibias: float = 2600.0
    pyr_pv_ibias: float = 900.0
    pv_pyr_ibias: float = 200.0
    pyr_sst_ibias: float = 2600.0
    sst_pyr_ibias: float = 8000.0
    vip_sst_ibias: float = 6000.0


class SingleMotifNetwork:
    """One PYR plus PV/SST/VIP, with plastic basal input and apical teacher."""

    def __init__(self, cfg: SingleMotifConfig):
        self.cfg = cfg
        self.pop: dict[str, object] = {}
        self.input_gen = None
        self.parrots = None
        self.teacher = None
        self.attention = None
        self.sst_tonic = None
        self.spike_rec = None
        self.plastic_conn = None
        self.rt: dict[str, int] | None = None

    def build(self) -> "SingleMotifNetwork":
        from biodl.sim import reset

        reset(seed=self.cfg.seed)
        self._copy_synapses()
        self._create_populations()
        self._connect_motif()
        self._connect_plastic_input()
        self._connect_teacher_attention()
        self._attach_readout()
        return self

    def _copy_synapses(self) -> None:
        import nest
        from biodl.nest_setup import SYNAPSE_MODEL

        c = self.cfg
        nest.CopyModel(
            SYNAPSE_MODEL,
            _PLASTIC_SYN,
            {
                "plastic": True,
                "binarize": False,
                "n_bit": 4,
                "eta": c.eta,
                "eta_L": c.eta_l,
                "delay": 0.1,
            },
        )
        nest.CopyModel(
            SYNAPSE_MODEL,
            _STATIC_SYN,
            {
                "plastic": False,
                "binarize": True,
                "n_bit": 3,
                "eta": 0.0,
                "eta_L": 0.0,
                "delay": 0.1,
            },
        )

    def _create_populations(self) -> None:
        import nest
        from biodl.nest_setup import NEURON_MODEL
        from biodl.params import neuron_params

        for key, cfg_key in (
            ("pyr", "pyr"),
            ("pv", "pvc"),
            ("sst", "sst"),
            ("vip", "vip"),
        ):
            pop = nest.Create(NEURON_MODEL, 1)
            params = neuron_params(cfg_key)
            if key == "pyr":
                params.update(
                    {
                        "effective_bias": True,
                        "k_ca_th_L_minus": self.cfg.ltd_low,
                        "k_ca_th_H_minus": self.cfg.ltd_high,
                        "k_ca_th_L_plus": self.cfg.ltp_low,
                        "k_ca_th_H_plus": self.cfg.ltp_high,
                        "Iapical_low": self.cfg.iapical_low,
                    }
                )
            pop.set(params)
            self.pop[key] = pop
        self.rt = self.pop["pyr"][0].get("receptor_types")

    def _syn(
        self, receptor: str, ibias: float, w: int = 7, model: str = _STATIC_SYN
    ) -> dict:
        from biodl.microcircuit import PORT

        assert self.rt is not None
        return {
            "synapse_model": model,
            "receptor_type": int(self.rt[PORT[receptor]]),
            "w": int(w),
            "Ibias": float(ibias),
        }

    def _connect_motif(self) -> None:
        import nest
        from biodl.microcircuit import MOTIF_EDGES

        c = self.cfg
        ibias_by_edge = {
            "pyr_pv": c.pyr_pv_ibias,
            "pv_pyr": c.pv_pyr_ibias,
            "pyr_sst": c.pyr_sst_ibias,
            "sst_pyr": c.sst_pyr_ibias,
            "vip_sst": c.vip_sst_ibias,
        }
        for src, dst, edge, receptor, _sign in MOTIF_EDGES:
            nest.Connect(
                self.pop[src],
                self.pop[dst],
                "one_to_one",
                self._syn(receptor, ibias_by_edge[edge]),
            )

    def _connect_plastic_input(self) -> None:
        import nest

        assert self.rt is not None
        c = self.cfg
        self.input_gen = nest.Create(
            "poisson_generator", c.n_input, {"rate": c.rate_inactive}
        )
        self.parrots = nest.Create("parrot_neuron", c.n_input)
        nest.Connect(
            self.input_gen, self.parrots, "one_to_one", {"weight": 1.0, "delay": 0.1}
        )
        nest.Connect(
            self.parrots,
            self.pop["pyr"],
            "all_to_all",
            {
                "synapse_model": _PLASTIC_SYN,
                "receptor_type": int(self.rt["NMDA_BASAL_SPIKES"]),
                "w": c.w0,
                "Ibias": c.plastic_ibias,
            },
        )
        self.plastic_conn = nest.GetConnections(
            source=self.parrots, target=self.pop["pyr"], synapse_model=_PLASTIC_SYN
        )

    def _connect_teacher_attention(self) -> None:
        import nest

        c = self.cfg
        self.teacher = nest.Create("poisson_generator", 1, {"rate": 0.0})
        nest.Connect(
            self.teacher,
            self.pop["pyr"],
            "one_to_one",
            self._syn("ampa_apical", c.teacher_ibias),
        )
        self.attention = nest.Create("poisson_generator", 1, {"rate": 0.0})
        nest.Connect(
            self.attention,
            self.pop["vip"],
            "one_to_one",
            self._syn("ampa_basal", c.cue_ibias),
        )
        self.sst_tonic = nest.Create("poisson_generator", 1, {"rate": c.sst_tonic_rate})
        nest.Connect(
            self.sst_tonic,
            self.pop["sst"],
            "one_to_one",
            self._syn("ampa_basal", c.sst_tonic_ibias),
        )

    def _attach_readout(self) -> None:
        import nest

        self.spike_rec = nest.Create("spike_recorder")
        nest.Connect(self.pop["pyr"], self.spike_rec)

    def _set_input_plasticity(self, enabled: bool) -> None:
        assert self.plastic_conn is not None
        self.plastic_conn.set({"plastic": bool(enabled)})

    def set_rates(self, rates) -> None:
        rates = np.asarray(rates, dtype=float)
        if rates.shape != (self.cfg.n_input,):
            raise ValueError(f"rates must have length {self.cfg.n_input}")
        self.input_gen.set([{"rate": float(r)} for r in rates])

    def set_pattern(self, active) -> None:
        active = set(int(i) for i in active)
        self.input_gen.set(
            [
                {"rate": self.cfg.rate_active if i in active else self.cfg.rate_inactive}
                for i in range(self.cfg.n_input)
            ]
        )

    def set_teacher(self, on: bool) -> None:
        self.teacher.set({"rate": self.cfg.teacher_rate if on else 0.0})

    def set_attention(self, on: bool) -> None:
        self.attention.set({"rate": self.cfg.attention_rate if on else 0.0})

    def present(
        self,
        active=None,
        rates=None,
        teacher_on: bool = False,
        attention: bool = True,
        t: float | None = None,
    ) -> None:
        import nest

        if rates is not None:
            self.set_rates(rates)
        else:
            self.set_pattern(active or [])
        self.set_teacher(teacher_on)
        self.set_attention(attention)
        self._set_input_plasticity(True)
        nest.Simulate(float(self.cfg.t_train if t is None else t))

    def infer_rate(self, active=None, rates=None, t: float | None = None) -> float:
        import nest

        duration = float(self.cfg.t_infer if t is None else t)
        if rates is not None:
            self.set_rates(rates)
        else:
            self.set_pattern(active or [])
        self.set_teacher(False)
        self.set_attention(False)
        self.spike_rec.set({"n_events": 0})
        self._set_input_plasticity(False)
        try:
            nest.Simulate(duration)
        finally:
            self._set_input_plasticity(True)
        return float(1e3 * self.spike_rec.get("n_events") / duration)

    def weight_vector(self) -> np.ndarray:
        import nest

        assert self.parrots is not None
        weights = [
            float(
                np.atleast_1d(
                    nest.GetConnections(self.parrots[i], self.pop["pyr"]).get("w")
                )[0]
            )
            for i in range(self.cfg.n_input)
        ]
        return np.asarray(weights, dtype=float)


class SingleMotifClassifier:
    """Binary rest-vs-fist classifier backed by one full-motif PYR detector."""

    classes_ = np.array([0, 1])

    def __init__(
        self,
        n_features: int,
        n_pyr: int | None = None,
        epochs: int = 1,
        active_frac: float = 0.625,
        t_present: float = 200.0,
        t_infer: float = 300.0,
        seed: int = 1,
        encode: str = "topk",
        rate_active: float = 10.0,
        threshold: float | None = None,
        motif_config: SingleMotifConfig | None = None,
        **motif_kwargs,
    ):
        self.n_features = int(n_features)
        self.n_pyr = 1
        self._requested_n_pyr = None if n_pyr is None else int(n_pyr)
        self.epochs = int(epochs)
        self.active_frac = float(active_frac)
        self.t_present = float(t_present)
        self.t_infer = float(t_infer)
        self.seed = int(seed)
        self.encode = encode
        self.rate_active = float(rate_active)
        base_cfg = motif_config or SingleMotifConfig(n_input=self.n_features)
        updates = {
            "n_input": self.n_features,
            "seed": self.seed,
            "epochs": self.epochs,
            "t_train": self.t_present,
            "t_infer": self.t_infer,
            "rate_active": self.rate_active,
            **motif_kwargs,
        }
        self.motif_config = replace(base_cfg, **updates)
        self.threshold_ = None if threshold is None else float(threshold)
        self.train_rest_rate_ = 0.0
        self.train_fist_rate_ = 0.0
        self.rate_margin_ = 0.0
        self._rate_scale: float | None = None
        self._fitted = False
        self.last_rate = 0.0

    def _encode(self, x) -> list[int]:
        x = np.asarray(x, dtype=float)
        k = max(1, int(round(self.active_frac * len(x))))
        return np.argsort(x)[-k:].tolist()

    def _fit_rate_scale(self, X: np.ndarray) -> float:
        pos = np.clip(X - np.median(X, axis=1, keepdims=True), 0.0, None)
        active = pos[pos > 0.0]
        if active.size == 0:
            return 1.0
        return max(float(np.percentile(active, _RATE_SCALE_PERCENTILE)), _EPS)

    def _encode_rates(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pos = np.clip(x - np.median(x), 0.0, None)
        if self._rate_scale is None:
            mx = float(pos.max())
            return (pos / mx) * self.rate_active if mx > 0 else np.zeros_like(x)
        rates = (pos / self._rate_scale) * self.rate_active
        return np.clip(rates, 0.0, self.rate_active)

    def _drive(self, x) -> dict:
        if self.encode == "rate":
            return {"rates": self._encode_rates(x)}
        return {"active": self._encode(x)}

    def fit(self, X, y) -> "SingleMotifClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n_samples, {self.n_features})")
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError(
                "SingleMotifClassifier is binary: labels must be 0 (rest) / 1 (fist)"
            )
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError("fit requires at least one rest sample and one fist sample")
        if self.encode not in {"topk", "rate"}:
            raise ValueError("encode must be 'topk' or 'rate'")

        if self.encode == "rate":
            self._rate_scale = self._fit_rate_scale(X)
        drives = [self._drive(x) for x in X]
        train_rates = np.asarray(
            _get_worker().request(
                "single_motif_fit", asdict(self.motif_config), drives, y.tolist()
            ),
            dtype=float,
        )
        self.train_rest_rate_ = float(np.mean(train_rates[y == 0]))
        self.train_fist_rate_ = float(np.mean(train_rates[y == 1]))
        self.rate_margin_ = self.train_rest_rate_ - self.train_fist_rate_
        if self.threshold_ is None:
            self.threshold_ = 0.5 * (self.train_rest_rate_ + self.train_fist_rate_)
        self.last_rate = float(train_rates[-1])
        self._fitted = True
        return self

    def _proba_from_rate(self, rate: float) -> tuple[float, float]:
        assert self.threshold_ is not None
        scale = max(abs(self.rate_margin_) / 6.0, 1.0)
        z = np.clip((self.threshold_ - float(rate)) / scale, -60.0, 60.0)
        p_fist = float(1.0 / (1.0 + np.exp(-z)))
        return 1.0 - p_fist, p_fist

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted or self.threshold_ is None:
            raise RuntimeError("call fit() before predict")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        worker = _get_worker()
        out = []
        for x in X:
            rate = float(
                worker.request("single_motif_predict", self._drive(x), self.t_infer)
            )
            self.last_rate = rate
            out.append(self._proba_from_rate(rate))
        return np.asarray(out, dtype=float)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] > proba[:, 0]).astype(int)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    def __getstate__(self):
        return self.__dict__.copy()
