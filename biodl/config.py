"""Load the network configuration files (from Chiara) in ``texts/``."""

import os

import yaml

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "texts"
)


def _load(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def neuron_config() -> dict:
    """Per-cell-type neuron parameters (``all`` block + ``pyr``/``sst``/``pvc``/``vip``)."""
    return _load("config_neur.yaml")


def synapse_config() -> dict:
    """Global synapse parameters (delay, learning rates, binarize)."""
    return _load("config_syn.yaml")


def network_config() -> dict:
    """Network: population sizes, connection weights, input rates, sim settings."""
    return _load("config_overlapping.yaml")
