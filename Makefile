# ============================================================
# Makefile -- reproduce all experiments and figures.
#
# Usage:
#   make venv           Create virtual environment and install dependencies.
#   make data           Download real-world datasets (Ames Housing, Abalone).
#   make simulations    Run all Monte Carlo experiments (saves CSVs).
#   make realworld      Run real-world regression experiments.
#   make figures-py     Generate quick-check Python figures.
#   make figures-r      Generate publication-quality R/ggplot figures.
#   make all            Full pipeline: venv + data + simulations + figures.
#   make clean          Remove generated results and figures.
# ============================================================

VENV     := .venv
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
RSCRIPT  := Rscript

.PHONY: all venv data simulations realworld figures-py figures-r clean cleanall help

# ---- Default target ----
all: venv data simulations realworld figures-py figures-r
	@echo "Done. Results in results/, figures in figures/."

# ---- Create venv and install package ----
venv: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@touch $(VENV)/bin/activate

# ---- Download real-world datasets ----
data: venv
	$(PYTHON) scripts/download_data.py

# ---- Run parametric simulations (writes CSVs to results/) ----
simulations: venv
	$(PYTHON) scripts/run_gaussian.py
	$(PYTHON) scripts/run_poisson.py
	$(PYTHON) scripts/run_uniform.py
	$(PYTHON) scripts/run_fourier.py
	$(PYTHON) scripts/run_correlated.py

# ---- Run real-world experiments ----
realworld: venv data
	$(PYTHON) scripts/run_realworld.py
	$(PYTHON) scripts/export_residuals_csv.py

# ---- Python quick-check figures (writes to figures/) ----
figures-py: venv
	$(PYTHON) scripts/plot_all.py

# ---- R publication figures (writes to figures/) ----
figures-r:
	$(RSCRIPT) R/plot_gaussian.R
	$(RSCRIPT) R/plot_poisson.R
	$(RSCRIPT) R/plot_uniform.R
	$(RSCRIPT) R/plot_regression.R
	$(RSCRIPT) R/plot_realworld.R

# ---- Clean generated files ----
clean:
	rm -f results/*.csv results/*.npz
	rm -f figures/*.pdf figures/*.png

# ---- Remove everything including venv and cached data ----
cleanall: clean
	rm -rf $(VENV)
	rm -f data/*.npy

# ---- Help ----
help:
	@echo "Targets:"
	@echo "  make venv          Create venv and install package"
	@echo "  make data          Download real-world datasets"
	@echo "  make simulations   Run parametric Monte Carlo experiments"
	@echo "  make realworld     Run real-world regression experiments"
	@echo "  make figures-py    Python quick-check figures"
	@echo "  make figures-r     R publication figures"
	@echo "  make all           Full pipeline"
	@echo "  make clean         Remove generated files"
	@echo "  make cleanall      Remove generated files + venv + data cache"
