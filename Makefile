PYTHON ?= python
export HF_HOME ?= $(CURDIR)/artifacts/hf-cache
export HF_DATASETS_CACHE ?= $(CURDIR)/artifacts/hf-cache/datasets
export UV_CACHE_DIR ?= $(CURDIR)/artifacts/uv-cache
export TMPDIR ?= $(CURDIR)/artifacts/tmp
export PIP_CACHE_DIR ?= $(CURDIR)/artifacts/pip-cache

.PHONY: setup check audit-data prepare-data bootstrap benchmark train-pilot train eval demo api release-check release
setup:
	$(PYTHON) -m pip install -e ".[data,train,eval,serve,dev]"
check:
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src
	$(PYTHON) -m pytest -q
	$(PYTHON) -m masriswitch.cli validate-config
audit-data:
	$(PYTHON) -m masriswitch.cli audit-data
prepare-data:
	$(PYTHON) -m masriswitch.cli prepare-data
bootstrap:
	$(PYTHON) -m masriswitch.cli bootstrap
benchmark:
	$(PYTHON) -m masriswitch.cli benchmark
train-pilot:
	$(PYTHON) -m masriswitch.cli train-pilot
train:
	$(PYTHON) -m masriswitch.cli train
eval:
	$(PYTHON) -m masriswitch.cli eval
demo:
	$(PYTHON) -m masriswitch.cli demo
api:
	$(PYTHON) -m masriswitch.cli api
release-check:
	$(PYTHON) -m masriswitch.cli release-check
release:
	$(PYTHON) -m masriswitch.cli release
