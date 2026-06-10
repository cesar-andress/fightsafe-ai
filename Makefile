# FightSafe AI — developer and reproduction tasks (GNU Make 3.81+)
# Requires: Python 3.12, pip install -e ".[dev]"

.PHONY: install format lint typecheck test test-unit test-integration test-e2e \
	coverage pre-commit ci clean \
	fusion-pdf fusion-assets fusion-all fusion-all-force fusion-clean \
	reproduce-fusion reproduce-sinica reproduce-sports reproduce-all \
	verify-repro check-paper-update

PYTHON ?= python3.12
PIP := $(PYTHON) -m pip
export PYTHONPATH := $(abspath src):$(abspath .)

# Companion manuscript directories (monorepo layout: sibling folders)
FUSION_DIR ?= ../fusion2026
SINICA_DIR ?= ../sinica2026
SPORTS_DIR ?= ../sports

RUFF_CFG := --config pyproject.toml
MYPY := $(PYTHON) -m mypy --config-file pyproject.toml --show-error-codes
PYTEST := $(PYTHON) -m pytest
PYTEST_UNIT := $(PYTEST) tests/unit
PYTEST_INT := $(PYTEST) tests/integration
PYTEST_E2E := $(PYTEST) tests/e2e
CI_PYTEST_COV := --cov=src/fightsafe_ai --cov-report=term-missing --cov-fail-under=74
COV_LOCAL := $(CI_PYTEST_COV) --cov-report=html --cov-report=xml
PDFLAGS ?= -interaction=nonstopmode -halt-on-error

install:
	$(PIP) install -U pip wheel
	$(PIP) install -e ".[dev]"

format:
	ruff format $(RUFF_CFG) .

lint:
	ruff check $(RUFF_CFG) .

typecheck:
	$(MYPY) src tests

test:
	$(PYTEST)

test-unit:
	$(PYTEST_UNIT)

test-integration:
	$(PYTEST_INT)

test-e2e:
	$(PYTEST_E2E)

coverage:
	$(PYTEST) $(COV_LOCAL)
	@echo "Coverage HTML: htmlcov/index.html"

pre-commit:
	pre-commit run --all-files

ci:
	ruff format --check .
	ruff check .
	$(MYPY) src tests
	$(PYTEST) $(CI_PYTEST_COV)

clean:
	rm -rf build dist .eggs
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov
	rm -f .coverage .coverage.* coverage.xml
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.egg-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true

## --- fusion2026 (Information Fusion manuscript) --------------------------------

fusion-pdf:
	test -f "$(FUSION_DIR)/main.tex" || (echo "Missing $(FUSION_DIR)/main.tex" && exit 1)
	cd "$(FUSION_DIR)" && pdflatex $(PDFLAGS) main.tex
	cd "$(FUSION_DIR)" && bibtex main
	cd "$(FUSION_DIR)" && pdflatex $(PDFLAGS) main.tex
	cd "$(FUSION_DIR)" && pdflatex $(PDFLAGS) main.tex

fusion-assets:
	$(PYTHON) -m fightsafe_ai.paper.build_all \
		--paper-dir "$(FUSION_DIR)" \
		--output-dir outputs/evaluation/boxingvi_batch

fusion-all:
	$(PYTHON) -m fightsafe_ai.paper.build_all \
		--dataset-root data/boxingvi \
		--video-ids V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 \
		--paper-dir "$(FUSION_DIR)" \
		--output-dir outputs/evaluation/boxingvi_batch \
		--strike-percentile 85 \
		--strike-merge-frames 8 \
		--tolerance-seconds 0.5 \
		--compare-baselines \
		--run-tests \
		--compile

fusion-all-force:
	$(PYTHON) -m fightsafe_ai.paper.build_all \
		--dataset-root data/boxingvi \
		--video-ids V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 \
		--paper-dir "$(FUSION_DIR)" \
		--output-dir outputs/evaluation/boxingvi_batch \
		--strike-percentile 85 \
		--strike-merge-frames 8 \
		--tolerance-seconds 0.5 \
		--compare-baselines \
		--run-tests \
		--compile \
		--force

fusion-clean:
	rm -f "$(FUSION_DIR)"/*.aux "$(FUSION_DIR)"/*.bbl "$(FUSION_DIR)"/*.blg \
		"$(FUSION_DIR)"/*.log "$(FUSION_DIR)"/*.out \
		"$(FUSION_DIR)"/*.toc "$(FUSION_DIR)"/*.lof "$(FUSION_DIR)"/*.lot \
		"$(FUSION_DIR)"/*.fls "$(FUSION_DIR)"/*.fdb_latexmk "$(FUSION_DIR)"/main.pdf

reproduce-fusion:
	bash scripts/reproduce_fusion2026.sh

## --- sinica2026 (TapKO / HITL manuscript) ------------------------------------

reproduce-sinica:
	bash scripts/reproduce_sinica2026.sh

## --- sports (FightSafe-Bench manuscript) -------------------------------------

reproduce-sports:
	bash scripts/reproduce_sports.sh

reproduce-all:
	bash scripts/reproduce_all.sh

verify-repro:
	$(PYTHON) scripts/verify_paper_outputs.py --paper all

## --- maintenance ---------------------------------------------------------------

PAPER_CHECK_FLAGS ?=
check-paper-update:
	FUSION_MAIN="$(FUSION_DIR)/main.tex" $(PYTHON) tools/check_paper_update.py $(PAPER_CHECK_FLAGS)

# Backward-compatible aliases (deprecated)
paper: fusion-pdf
paper-assets: fusion-assets
paper-all: fusion-all
paper-all-force: fusion-all-force
paper-clean: fusion-clean
