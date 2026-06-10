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


def neuron_params(cell_type: str, cfg: dict | None = None) -> dict:
    """Resolved dynaple_neur parameter dict for one cell type."""
    if cell_type not in CELL_TYPES:
        raise ValueError(f"unknown cell type {cell_type!r}, expected one of {CELL_TYPES}")
    cfg = cfg if cfg is not None else neuron_config()
    merged = dict(cfg.get("all", {}))
    merged.update(cfg.get(cell_type, {}))
    return {RENAME.get(k, k): v for k, v in merged.items()}
