# paper/ — EAAI manuscript

**Title:** Engineering an Interpretable Temporal Event Pipeline with Explicit Channel Availability: A Combat-Sports Case Study  
**Journal:** Engineering Applications of Artificial Intelligence

## Build

```bash
# From repository root (recommended): regenerate tables/figures first
python3.12 scripts/generate_eaai_assets.py

cd paper
latexmk -pdf -interaction=nonstopmode main.tex
# Bibliography: bibtex main
```

Canonical numerical source: `../canonical_results/run_20260730_005150/`.

Appendix tables are printed after the bibliography.
