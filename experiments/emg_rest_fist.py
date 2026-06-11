"""Neuromorphic EMG decoder demo: rest vs fist (synthetic, no hardware).

Proves the spiking two-column classifier (biodl.network.ClassifierNetwork),
wrapped behind a scikit-learn API (biodl.decoder.NeuromorphicClassifier), can
learn to distinguish two EMG gestures from channel features. This is the
standalone proof; the galvani example wires the same model into MyoGestic.

Synthetic EMG: each window is one RMS value per channel.
  * "rest"  recruits the extensor group  (channels 0-9, moderate)
  * "fist"  recruits the flexor group    (channels 6-15, strong)  -- overlap 6-9
plus baseline + per-window gain noise, so the two gestures are separable but
share channels (exactly the overlapping-pattern problem the network is built for).

Run:  python experiments/emg_rest_fist.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biodl.decoder import NeuromorphicClassifier  # noqa: E402

N_CH = 16
REST_CH = np.arange(0, 10)
FIST_CH = np.arange(6, 16)
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def synthetic_emg(n_per_class, seed=0):
    """Return (X, y): X = (2*n_per_class, N_CH) RMS features, y in {0:rest, 1:fist}."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, chans, amp in ((0, REST_CH, 0.55), (1, FIST_CH, 0.9)):
        for _ in range(n_per_class):
            f = rng.uniform(0.05, 0.15, N_CH)                      # baseline RMS
            f[chans] += rng.uniform(0.6, 1.0, len(chans)) * amp    # gesture activation
            f *= rng.uniform(0.8, 1.2, N_CH)                       # per-window gain noise
            X.append(f)
            y.append(label)
    X, y = np.asarray(X), np.asarray(y)
    p = rng.permutation(len(X))
    return X[p], y[p]


def main():
    Xtr, ytr = synthetic_emg(n_per_class=8, seed=1)
    Xte, yte = synthetic_emg(n_per_class=6, seed=99)

    clf = NeuromorphicClassifier(n_features=N_CH, n_pyr=4, epochs=6, active_frac=0.5)
    print("training neuromorphic decoder on rest vs fist ...")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(np.mean(pred == yte))
    print(f"test accuracy: {acc:.0%}  ({np.sum(pred == yte)}/{len(yte)})")

    # ---- figure ----
    fig, (ax_emg, ax_res) = plt.subplots(1, 2, figsize=(11, 4.2),
                                         gridspec_kw={"width_ratios": [1.2, 1]})
    rest_mean = Xtr[ytr == 0].mean(0)
    fist_mean = Xtr[ytr == 1].mean(0)
    ch = np.arange(N_CH)
    ax_emg.bar(ch - 0.2, rest_mean, 0.4, color="#2874A6", label="rest")
    ax_emg.bar(ch + 0.2, fist_mean, 0.4, color="#C0392B", label="fist")
    ax_emg.axvspan(5.5, 9.5, color="grey", alpha=0.12)
    ax_emg.text(7.5, ax_emg.get_ylim()[1] * 0.95, "overlap", ha="center", fontsize=8, color="#555")
    ax_emg.set_xlabel("EMG channel")
    ax_emg.set_ylabel("mean RMS feature")
    ax_emg.set_title("Synthetic EMG: rest vs fist channel activation")
    ax_emg.legend(fontsize=9)

    proba = clf.predict_proba(Xte)
    order = np.argsort(yte)
    ax_res.bar(np.arange(len(yte)), proba[order, 1],
               color=["#C0392B" if yte[order][i] == 1 else "#2874A6" for i in range(len(yte))])
    ax_res.axhline(0.5, color="grey", ls="--", lw=1)
    ax_res.set_ylim(0, 1)
    ax_res.set_xlabel("test window (sorted by true class)")
    ax_res.set_ylabel("P(fist)  =  column-B firing share")
    ax_res.set_title(f"Decoder output — test accuracy {acc:.0%}")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "emg_rest_fist.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
