"""Tests for biodl's deterministic logic and network faithfulness.

These freeze the regressions we hit (and fixed) by hand this project: the
Iapical_low sign-flip, the calcium-parameter rename, the edge-data consistency,
and that the wired network actually matches the declared connectivity.

Runnable two ways (no framework required):
    python tests/test_biodl.py     # plain runner, prints PASS/FAIL
    pytest tests/                  # if pytest is installed
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biodl.config import network_config, neuron_config, synapse_config  # noqa: E402
from biodl.microcircuit import (  # noqa: E402
    DEFAULT_WEIGHTS,
    DRIVE_EDGES,
    MOTIF_EDGES,
    PORT,
    RECURRENT_EDGES,
)
from biodl.params import CELL_TYPES, neuron_params  # noqa: E402


# -- params.py --------------------------------------------------------------
def test_params_negate_iapical_low():
    """Iapical_low must be sign-flipped: our dynaple_neur floors at -Iapical_low,
    Chiara's config writes the negative plateau directly (else the gate dies)."""
    raw = neuron_config()["pyr"]["Iapical_low"]
    assert neuron_params("pyr")["Iapical_low"] == -raw


def test_params_rename_calcium():
    out = neuron_params("pyr")
    assert "Isoma_ca_bias" in out, "config Isoma_ca_w must map to Isoma_ca_bias"
    assert "Isoma_ca_w" not in out, "the pre-rename key must not leak through"


def test_params_unknown_type_raises():
    try:
        neuron_params("nope")
    except ValueError:
        return
    raise AssertionError("unknown cell type should raise ValueError")


def test_params_all_cell_types_resolve():
    for ct in CELL_TYPES:
        assert isinstance(neuron_params(ct), dict)


# -- config.py --------------------------------------------------------------
def test_configs_load():
    assert "all" in neuron_config()
    assert "net" in network_config()
    assert "delay" in synapse_config()


# -- microcircuit edge data -------------------------------------------------
def test_edge_data_consistency():
    """Every motif/recurrent/drive edge references a real receptor port and a
    weight that actually exists in DEFAULT_WEIGHTS, with a valid sign."""
    for edges in (MOTIF_EDGES, RECURRENT_EDGES, DRIVE_EDGES):
        for src, dst, wkey, receptor, sign in edges:
            assert receptor in PORT, f"{receptor!r} is not a known receptor port"
            assert wkey in DEFAULT_WEIGHTS, f"{wkey!r} missing from DEFAULT_WEIGHTS"
            assert sign in ("+", "-"), f"bad sign {sign!r} on {src}->{dst}"


# -- live network faithfulness (needs NEST) ---------------------------------
def test_network_matches_edge_data():
    """The 1-neuron column wires exactly the motif + drive edges (no recurrence
    at N=1, no autapses) -- the faithfulness check, frozen as a regression."""
    import nest

    from biodl.microcircuit import Microcircuit
    from biodl.sim import reset

    reset(seed=1)
    net = Microcircuit(1, 1, 1, 1).build().drive(input_rate=10.0)

    idname = {}
    for key, pop in net.pop.items():
        for nid in pop.tolist():
            idname[nid] = key
    for key, gen in net.gen.items():
        for nid in gen.tolist():
            idname[nid] = key

    actual = set()
    for c in nest.GetConnections():
        s, t = c.source, c.target
        if s in idname and t in idname:
            actual.add((idname[s], idname[t]))

    expected = {(s, d) for s, d, *_ in MOTIF_EDGES} | {(s, d) for s, d, *_ in DRIVE_EDGES}
    assert actual == expected, f"wired {sorted(actual)} != declared {sorted(expected)}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
