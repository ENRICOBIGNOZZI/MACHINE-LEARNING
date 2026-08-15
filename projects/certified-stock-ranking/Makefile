.PHONY: test synthetic paper clean

test:
	PYTHONPATH=src pytest

synthetic:
	PYTHONPATH=src python scripts/run_synthetic_demo.py
	mkdir -p paper/figures
	cp results/synthetic/concept_partial_order.pdf paper/figures/concept_partial_order.pdf
	cp results/synthetic/certification_frontier.pdf paper/figures/certification_frontier.pdf

paper: synthetic
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null
	cd paper && (bibtex main >/dev/null 2>&1 || bibtex8 main >/dev/null)
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null

clean:
	cd paper && rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.toc main.synctex.gz
	rm -rf .pytest_cache
