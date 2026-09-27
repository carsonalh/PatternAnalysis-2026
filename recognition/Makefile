.PHONY: all
all:

.PHONY: install-kernel
install-kernel:
	uv run ipython kernel install --user --env VIRTUAL_ENV .venv --name=Time_Series_Generation_s4702231

.PHONY: run-jupyter
run-jupyter:
	PYTHONPATH="$(shell realpath .)$${PYTHONPATH:+:$$PYTHONPATH}" \
	uv run --with jupyter jupyter lab --ServerApp.root_dir="$(shell realpath .)"
