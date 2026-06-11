# workshop — convenience targets.
#
# The nest-simulator PyPI wheel needs post-sync patching to compile NESTML
# modules (see scripts/fix-nest-wheel.sh and
# https://github.com/nest/nest-simulator/issues/3762). These targets keep the
# venv reproducible: always `make sync`, never a bare `uv sync`.

VENV_PY := $(CURDIR)/.venv/bin/python
DYNAPLE := ../dynaple-model-nest

.PHONY: setup sync build tutorial clean

setup:           ## one-time: system C libs, then a patched sync
	brew install gsl boost cmake open-mpi libomp libtool
	$(MAKE) sync

sync:            ## uv sync, then re-apply the nest-wheel workarounds
	uv sync
	bash scripts/fix-nest-wheel.sh

build: sync      ## (re)generate + compile the dynaple NEST module
	cd $(DYNAPLE)/neuron_model && $(VENV_PY) generate_neuron_model.py
	bash scripts/fix-nest-wheel.sh

tutorial:        ## run the plastic-synapse tutorial, produces the plot PNG
	$(VENV_PY) $(DYNAPLE)/tutorial/run_tutorial.py

clean:           ## remove the compiled module build output
	rm -rf $(DYNAPLE)/neuron_model/target
