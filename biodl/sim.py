"""A scoped simulation context that owns the NEST kernel lifecycle.

The NEST kernel is global, mutable state. Every caller must remember to
``nest.ResetKernel()``, set the resolution, and install the dynaple module
*before* building anything — and forgetting a step silently inherits the previous
run's neurons, connections and clock. Most of the hard-to-find bugs in this repo
came from exactly that.

``Simulation`` makes the ownership explicit and un-forgettable::

    from biodl.sim import Simulation
    from biodl.microcircuit import Microcircuit

    with Simulation(resolution=0.1, seed=1) as sim:
        net = Microcircuit(1, 1, 1, 1).build().drive(input_rate=50).attach_recorders()
        sim.run(1000.0)
        rates = net.rates(1000.0)        # recorders are still readable here
    # outside the block the kernel is left as-is until the next Simulation()

It replaces the ``ResetKernel`` / ``SetKernelStatus`` / ``install_dynaple``
boilerplate with one line, adds optional deterministic seeding, and scopes the
run so two experiments in one process can't bleed into each other.
"""

from __future__ import annotations

import nest

from biodl.nest_setup import install_dynaple


def reset(
    resolution: float = 0.1, seed: int | None = None, verbosity: str = "M_ERROR"
) -> None:
    """Reset + configure the NEST kernel and install the dynaple module.

    The imperative one-liner equivalent of entering a :class:`Simulation` block.
    Replaces the ``nest.ResetKernel(); nest.SetKernelStatus({"resolution": ...})``
    (and easy-to-forget ``install_dynaple()``) boilerplate every script repeated.
    Use :class:`Simulation` for scoped ``with`` blocks; ``reset()`` for flat scripts.
    """
    nest.set_verbosity(verbosity)
    nest.ResetKernel()
    status: dict = {"resolution": float(resolution)}
    if seed is not None:
        status["rng_seed"] = int(seed)
    nest.SetKernelStatus(status)
    install_dynaple()


class Simulation:
    """Context manager that resets and configures the NEST kernel on entry."""

    def __init__(
        self,
        resolution: float = 0.1,
        seed: int | None = None,
        verbosity: str = "M_ERROR",
    ):
        self.resolution = float(resolution)
        self.seed = seed
        self.verbosity = verbosity

    def __enter__(self) -> "Simulation":
        reset(self.resolution, self.seed, self.verbosity)
        return self

    def run(self, t_ms: float) -> None:
        """Advance the simulation by ``t_ms`` milliseconds."""
        nest.Simulate(float(t_ms))

    @property
    def time(self) -> float:
        """Current biological time (ms)."""
        return nest.GetKernelStatus("biological_time")

    def __exit__(self, *exc) -> bool:
        return False  # don't suppress exceptions
