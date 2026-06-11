"""A scikit-learn-style EMG gesture classifier backed by the neuromorphic
two-column spiking network (``biodl.network.ClassifierNetwork``).

Wraps the spiking classifier behind the same ``fit`` / ``predict`` /
``predict_proba`` / ``score`` interface a MyoGestic model uses, so it drops into
the galvani pipeline like any sklearn model. Binary gestures only (e.g. rest vs
fist): **class 0 → column A, class 1 → column B**.

Each EMG feature window (one value per channel, e.g. RMS) is encoded to the
network's binary input by activating its most-active channels (top
``active_frac``). Two gestures that recruit different channel sets therefore
produce different input patterns, and the three-factor delta rule learns to make
the corresponding column fire faster. Inference is the same patterns with the
teacher off (weights frozen); the predicted class is the faster-firing column.

Requires NEST + the compiled dynaple module (the workshop venv).
"""

from __future__ import annotations

import numpy as np

from biodl.network import ClassifierNetwork, Fig2Config
from biodl.sim import reset


class NeuromorphicClassifier:
    """Two-column spiking classifier with an sklearn-style API (binary)."""

    classes_ = np.array([0, 1])

    def __init__(self, n_features: int, n_pyr: int = 4, epochs: int = 5,
                 active_frac: float = 0.5, t_present: float = 300.0,
                 t_infer: float = 200.0, seed: int = 1,
                 encode: str = "topk", rate_active: float = 50.0):
        self.n_features = int(n_features)
        self.n_pyr = int(n_pyr)
        self.epochs = int(epochs)
        self.active_frac = float(active_frac)
        self.t_present = float(t_present)
        self.t_infer = float(t_infer)
        self.seed = int(seed)
        self.encode = encode  # "topk" (binary) or "rate" (amplitude rate-coding)
        self.rate_active = float(rate_active)
        self.net: ClassifierNetwork | None = None

    # -- encoding -----------------------------------------------------------
    def _encode(self, x) -> list[int]:
        """Feature window -> indices of the most-active channels (the pattern)."""
        x = np.asarray(x, dtype=float)
        k = max(1, int(round(self.active_frac * len(x))))
        return np.argsort(x)[-k:].tolist()

    def _encode_rates(self, x) -> np.ndarray:
        """Amplitude rate-coding: each channel -> Poisson rate (Hz) proportional
        to its feature (normalised to the strongest channel)."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, None)
        mx = x.max()
        return (x / mx) * self.rate_active if mx > 0 else np.zeros_like(x)

    def _drive(self, x) -> dict:
        return {"rates": self._encode_rates(x)} if self.encode == "rate" else {"active": self._encode(x)}

    # -- sklearn API --------------------------------------------------------
    def fit(self, X, y) -> "NeuromorphicClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("NeuromorphicClassifier is binary: labels must be 0 (rest) / 1 (fist)")
        reset(seed=self.seed)
        self.net = (ClassifierNetwork(Fig2Config(n_input=self.n_features, n_pyr=self.n_pyr,
                                                 seed=self.seed))
                    .build()
                    .randomize_input_weights(seed=self.seed))
        # Train on each class's *canonical* pattern (the mean feature window),
        # not every recorded window: with NEST in a real-time GUI thread, looping
        # all windows x epochs hangs the UI. The class-defining channels dominate
        # the mean, so the canonical pattern generalises to noisy test windows.
        drives = {int(c): self._drive(X[y == c].mean(axis=0))
                  for c in (0, 1) if (y == c).any()}
        for _ in range(self.epochs):
            for label, drive in drives.items():
                self.net.present(teacher="A" if label == 0 else "B", t=self.t_present, **drive)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = []
        for x in X:
            r = self.net.infer_rates(t=self.t_infer, **self._drive(x))
            a, b = r["A"], r["B"]
            tot = a + b
            out.append([0.5, 0.5] if tot == 0 else [a / tot, b / tot])
        return np.asarray(out)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        # tie -> class 0 (rest / column A), matching ClassifierNetwork.predict ("A if A>=B")
        return (proba[:, 1] > proba[:, 0]).astype(int)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))
