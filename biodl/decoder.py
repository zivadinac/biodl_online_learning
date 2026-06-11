"""A scikit-learn-style EMG gesture classifier backed by the neuromorphic
two-column spiking network (``biodl.network.ClassifierNetwork``).

**NEST runs in a separate process.** The GUI process (MyoGestic) never imports
NEST, so there is no libomp/OpenGL segfault, no shared-global-kernel thread race,
and no GIL contention slowing the UI. ``fit`` / ``predict`` are sent as messages
over a pipe to one persistent NEST worker that owns the single kernel; the worker
serialises requests, so train and predict can never collide.

Public API is unchanged (``fit`` / ``predict`` / ``predict_proba`` / ``score`` /
``classes_``), so it stays a drop-in MyoGestic model. Binary gestures only:
**class 0 -> column A (rest), class 1 -> column B (fist)**. Each EMG feature
window is encoded (client-side, pure numpy) to the network's input -- binary
top-``active_frac`` channels, or amplitude rate-coding with ``encode="rate"``.

The worker needs NEST + the compiled dynaple module + ``biodl`` on its path.
"""

from __future__ import annotations

import sys
import threading

import numpy as np


_BASE_PLASTIC_IBIAS = 0.3
_REFERENCE_ACTIVE = 8.0  # 16-channel / active_frac=0.5 calibration point
_RATE_SCALE_PERCENTILE = 75.0
_EPS = 1e-12


# ---------------------------------------------------------------------------
# NEST worker -- runs in its OWN process (spawned). Imports NEST + biodl there
# so they never touch the GUI process. Protocol over the pipe:
#   ("fit", cfg, drives)        -> ("ok", None)
#   ("predict", drive, t_infer) -> ("ok", (rate_A, rate_B))
#   ("stop",)                   -> ("ok", None) then exit
# Errors come back as ("error", traceback_str).
# ---------------------------------------------------------------------------
def _worker_serve(conn) -> None:  # pragma: no cover - separate process
    import nest

    nest.set_verbosity("M_ERROR")
    from biodl.network import ClassifierNetwork, Fig2Config
    from biodl.sim import reset

    net = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        cmd = msg[0]
        try:
            if cmd == "fit":
                _, cfg, drives = msg
                reset(seed=cfg["seed"])
                fig_cfg = Fig2Config(
                    n_input=cfg["n_input"],
                    n_pyr=cfg["n_pyr"],
                    seed=cfg["seed"],
                    plastic_ibias=cfg["plastic_ibias"],
                )
                net = ClassifierNetwork(fig_cfg).build()
                if cfg.get("template_init", False):
                    templates = {
                        "A" if label == 0 else "B": drive["rates"]
                        for label, drive in drives.items()
                        if "rates" in drive
                    }
                    net.set_input_weight_templates(templates)
                else:
                    net.randomize_input_weights(seed=cfg["seed"])
                    for _ in range(cfg["epochs"]):
                        for label, drive in drives.items():
                            net.present(teacher="A" if label == 0 else "B",
                                        t=cfg["t_present"], **drive)
                conn.send(("ok", None))
            elif cmd == "predict":
                _, drive, t_infer = msg
                r = net.infer_rates(t=t_infer, **drive)
                conn.send(("ok", (float(r["A"]), float(r["B"]))))
            elif cmd == "stop":
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown command {cmd!r}"))
        except Exception:  # noqa: BLE001 - report any worker error back to the client
            import traceback
            conn.send(("error", traceback.format_exc()))


class _NestWorker:
    """Client handle to the NEST subprocess; serialises requests (one kernel).

    The worker is a plain ``python -m biodl._worker_entry`` subprocess that
    connects back over a localhost socket -- NOT a multiprocessing spawn target.
    That matters: spawn re-imports the parent's ``__main__``, which would re-run
    an example's module-level GUI/bracelet/LSL setup in the child and break it.
    """

    def __init__(self):
        import os
        import subprocess
        from multiprocessing.connection import Listener

        authkey = os.urandom(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = self._listener.address
        env = os.environ.copy()
        env["BIODL_WORKER_ADDR"] = f"{host}:{port}"
        env["BIODL_WORKER_AUTHKEY"] = authkey.decode("latin1")
        # make biodl importable in the child (for `-m` and its own imports)
        pp = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen([sys.executable, "-m", "biodl._worker_entry"], env=env)
        self.conn = self._listener.accept()  # blocks until the worker connects
        self._lock = threading.Lock()

    def request(self, *msg):
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError("NEST worker process is not running")
            self.conn.send(msg)
            status, payload = self.conn.recv()
        if status == "error":
            raise RuntimeError(f"NEST worker error:\n{payload}")
        return payload

    def alive(self) -> bool:
        return self.proc.poll() is None


# One persistent worker for the whole process: the NEST kernel is single and
# global, so funnel every classifier instance through the same subprocess.
_worker: _NestWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _NestWorker:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.alive():
            _worker = _NestWorker()
    return _worker


class NeuromorphicClassifier:
    """Two-column spiking classifier with an sklearn-style API (binary).

    Thin client: encoding is local numpy; the NEST work happens in the worker.
    """

    classes_ = np.array([0, 1])

    def __init__(self, n_features: int, n_pyr: int = 4, epochs: int = 5,
                 active_frac: float = 0.5, t_present: float = 300.0,
                 t_infer: float = 300.0, seed: int = 1,
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
        self._rate_scale: float | None = None
        self._fitted = False

    # -- encoding (pure numpy, client-side) ---------------------------------
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
        """Median-referenced amplitude rate-coding.

        After ``fit`` the scale is learned from training windows, so rates keep
        absolute amplitude information across windows instead of forcing every
        strongest channel to ``rate_active``. Before ``fit`` this falls back to
        per-window max scaling for interactive inspection/backward compatibility.
        """
        x = np.asarray(x, dtype=float)
        pos = np.clip(x - np.median(x), 0.0, None)
        if self._rate_scale is None:
            mx = pos.max()
            return (pos / mx) * self.rate_active if mx > 0 else np.zeros_like(x)
        rates = (pos / self._rate_scale) * self.rate_active
        return np.clip(rates, 0.0, self.rate_active)

    def _drive(self, x) -> dict:
        return {"rates": self._encode_rates(x)} if self.encode == "rate" else {"active": self._encode(x)}

    # -- sklearn API (delegates NEST work to the worker process) ------------
    def fit(self, X, y) -> "NeuromorphicClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("NeuromorphicClassifier is binary: labels must be 0 (rest) / 1 (fist)")
        if self.encode not in {"topk", "rate"}:
            raise ValueError("encode must be 'topk' or 'rate'")

        if self.encode == "rate":
            self._rate_scale = self._fit_rate_scale(X)
            encoded = np.vstack([self._encode_rates(x) for x in X])
            drives = {int(c): {"rates": encoded[y == c].mean(axis=0)}
                      for c in (0, 1) if (y == c).any()}
        else:
            # canonical (mean) pattern per class -- class-defining channels dominate
            drives = {int(c): self._drive(X[y == c].mean(axis=0))
                      for c in (0, 1) if (y == c).any()}

        masses = []
        for drive in drives.values():
            if "rates" in drive:
                masses.append(float(np.sum(drive["rates"])) / max(self.rate_active, _EPS))
            else:
                masses.append(float(len(drive["active"])))
        mean_mass = float(np.mean(masses)) if masses else 1.0
        plastic_ibias = _BASE_PLASTIC_IBIAS * _REFERENCE_ACTIVE / max(mean_mass, 1.0)

        cfg = {"n_input": self.n_features, "n_pyr": self.n_pyr, "seed": self.seed,
               "epochs": self.epochs, "t_present": self.t_present,
               "plastic_ibias": plastic_ibias,
               "template_init": self.encode == "rate"}
        _get_worker().request("fit", cfg, drives)
        self._fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before predict")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        worker = _get_worker()
        out = []
        for x in X:
            a, b = worker.request("predict", self._drive(x), self.t_infer)
            tot = a + b
            out.append([0.5, 0.5] if tot == 0 else [a / tot, b / tot])
        return np.asarray(out)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        # tie -> class 0 (rest / column A), matching ClassifierNetwork.predict
        return (proba[:, 1] > proba[:, 0]).astype(int)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    # the worker handle is process-global and unpicklable; don't carry it in
    # pickled models (MyoGestic save_model). A reloaded model re-attaches lazily.
    def __getstate__(self):
        return self.__dict__.copy()
