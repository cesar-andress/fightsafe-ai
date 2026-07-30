# Build the new manuscript from this directory only.
# Prefer latexmk when available.

TEX := main.tex
PDF := main.pdf

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(TEX) refs.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX); \
	else \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
		bibtex main || true; \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -c $(TEX); \
	fi
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.lof *.lot \
		*.nav *.out *.run.xml *.snm *.spl *.synctex.gz *.toc *.vrb *.xdv

distclean: clean
	rm -f $(PDF)
