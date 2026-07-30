# Manuscript build (optional)

From package root:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode main.tex
```

Requires a TeX Live installation with `elsarticle`, `booktabs`, `tabularx`, `hyperref`, `microtype`.
Auxiliary files are build artefacts and are not part of the deposit.
