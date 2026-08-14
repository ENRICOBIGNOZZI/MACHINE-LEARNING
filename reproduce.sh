#!/usr/bin/env bash
set -euo pipefail

python code/run_simulations.py
python code/run_certification_efficiency.py --reuse-repetitions
python code/run_first_order.py
python code/run_electricity.py
python code/run_co2.py
python code/run_triggered_recertification.py --reuse-repetitions
python code/make_revision_tables.py
python code/make_top_paper_figures.py

# main.bbl is included in the release; BibTeX is therefore not required.
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
