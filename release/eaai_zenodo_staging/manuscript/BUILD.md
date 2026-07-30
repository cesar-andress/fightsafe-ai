# Manuscript build (optional)

From this directory:

```bash
cd manuscript
make
# or: latexmk -pdf -interaction=nonstopmode main.tex
```

`figures/`, `tables/` and `supplementary/` are symlinks to the package-root directories so `\includegraphics` / `\input` paths match the paper1 layout.

Requires TeX Live with `elsarticle`, `booktabs`, `tabularx`, `hyperref`, `microtype`.
Auxiliary files are build artefacts and are not part of the deposit.
The appendix is printed after the bibliography.
