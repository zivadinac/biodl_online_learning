"""Load the Dynap-LE NEST module, working around the non-relocatable wheel.

The ``nest-simulator`` PyPI wheel (3.10.0rc*) ships ``build_info["prefix"]`` and
``nest-config`` with the wheel-build machine's temporary paths frozen in, so a
bare ``nest.Install("dynaple_module")`` fails with "file not found" even when the
compiled ``.so`` is present (nest-simulator#3762). The reliable incantation is to
point ``build_info["prefix"]`` at the real install dir and load the ``.so`` by
absolute path.
"""

import os

import nest

MODULE = "dynaple_module"
NEURON_MODEL = "dynaple_neur__with_dynaple_syn"
SYNAPSE_MODEL = "dynaple_syn__with_dynaple_neur"


def _module_so() -> str:
    """Absolute path to the compiled module .so, trying known locations."""
    nest_dir = os.path.dirname(nest.__file__)
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    candidates = [
        os.path.join(nest_dir, "lib", "nest", f"{MODULE}.so"),
        os.path.join(repo, "..", "dynaple-model-nest", "neuron_model", "target", f"{MODULE}.so"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"{MODULE}.so not found in {candidates}. Run `make build` to compile it."
    )


def install_dynaple() -> None:
    """Install the Dynap-LE module into the running kernel. Idempotent."""
    nest.build_info["prefix"] = os.path.dirname(nest.__file__)
    try:
        nest.Install(_module_so())
    except Exception as exc:  # noqa: BLE001 - NEST raises if already loaded
        if "loaded already" not in str(exc) and "already" not in str(exc).lower():
            raise
