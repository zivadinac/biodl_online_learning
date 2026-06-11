"""Map Chiara's ``config_neur.yaml`` onto the ``dynaple_neur`` model parameters.

The config targets a sibling model revision (``alive_neur``/``delta_syn``); a few
parameter names differ from the ``dynaple_neur`` build we use. Known renames are
in ``RENAME``. Each cell type's parameters are the shared ``all`` block updated
with the type-specific overrides.
"""

from biodl.config import neuron_config

# config_neur.yaml uses these per-type sections; "pvc" is PV.
CELL_TYPES = ("pyr", "sst", "pvc", "vip")

# config name -> dynaple_neur name (verified against the compiled model).
RENAME = {
    "Isoma_ca_w": "Isoma_ca_bias",  # "CA jump height, on post"
}

# Parameters whose SIGN must flip between Chiara's revision and dynaple_neur.
# The dynaple_neur model clamps the apical current at -Iapical_low
# (Iapical = max(Iampa_apical - Igaba_b_apical, -Iapical_low), .nestml line ~320),
# whereas her sibling revision floors at Iapical_low directly. So her negative
# plateau (config pyr Iapical_low = -1000) must be negated here to keep the floor
# negative; left as -1000 it becomes a +1000 pA floor that PINS the apical and
# completely defeats the SST -> PYR-apical disinhibition gate (confirmed by probe
# and by Codex). Negating reproduces the paper's Ia = max(I_teach - I_SST, -I_SAT).
# FLAG FOR CHIARA: please confirm the Iapical_low sign convention in your model.
NEGATE = ("Iapical_low",)


def neuron_params(cell_type: str, cfg: dict | None = None) -> dict:
    """Resolved dynaple_neur parameter dict for one cell type."""
    if cell_type not in CELL_TYPES:
        raise ValueError(
            f"unknown cell type {cell_type!r}, expected one of {CELL_TYPES}"
        )
    cfg = cfg if cfg is not None else neuron_config()
    merged = dict(cfg.get("all", {}))
    merged.update(cfg.get(cell_type, {}))
    out = {}
    for key, value in merged.items():
        name = RENAME.get(key, key)
        out[name] = -value if name in NEGATE else value
    return out
