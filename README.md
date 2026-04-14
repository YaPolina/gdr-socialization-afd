# Does Socialization Under the GDR Increase AfD Vote Probability?

Master's thesis project using the German General Social Survey (ALLBUS) 2023 and causal-inference methods (AIPW with overlap weights) to estimate the effect of growing up in the former GDR on support for the AfD.

## Data

The analysis uses **ALLBUS 2023** microdata, available by request from GESIS -- Leibniz Institute for the Social Sciences: https://www.gesis.org (Study ZA8830, DOI: https://doi.org/10.4232/1.14544).

The raw dataset is not included due to licensing restrictions. After obtaining it, place the file at `data/raw/ALLBUS_2023.dta`.

## Repository Structure

```
scripts/                 Pipeline scripts (run sequentially)
  step1_build_model_afd.py         Data cleaning and sample construction
  step2_ate_overlap_eastyouth.py   AIPW + overlap weight estimation (GroupKFold cross-fitting)
  step3_balance_overlap_diagnostics.py  Covariate balance (SMD) diagnostics
  step4_inference_multicluster_eastyouth.py  Cluster bootstrap inference (xs11, land)
  step5_rosenbaum_delta_curve.py   Rosenbaum-style sensitivity analysis
  step6_spatial_placebo.py         Spatial placebo tests (1000 permutations)
  step7_cohort_heterogeneity.py    Cohort heterogeneity by age at reunification
  step8_thesis_figures.py          All thesis figures (fig01--fig12)
  step9_thesis_tables.py           Thesis summary tables (table1--table6)

src/                     Shared modules
  clean.py               Data cleaning and recoding utilities
  config.py              Paths, variable lists, covariate specifications
  io.py                  I/O helpers
  audit.py               Variable audit and checks

data/
  raw/                   Raw ALLBUS data (not tracked; see README inside)
  derived/               Intermediate datasets (not tracked; reproducible from step1)
  external/              germany_states.geojson for map figures

outputs/
  figures/               Thesis figures (fig01--fig12, fig_cohort_age1990)
  tables/                Summary tables (table1--table6)
    placebo/             Spatial placebo draws and results
```

## Reproducibility

### 1. Set up environment

```bash
poetry install
poetry shell
```

### 2. Run the pipeline

Run each step sequentially:

```bash
poetry run python scripts/step1_build_model_afd.py
poetry run python scripts/step2_ate_overlap_eastyouth.py
poetry run python scripts/step3_balance_overlap_diagnostics.py
poetry run python scripts/step4_inference_multicluster_eastyouth.py
poetry run python scripts/step5_rosenbaum_delta_curve.py
poetry run python scripts/step6_spatial_placebo.py --all
poetry run python scripts/step7_cohort_heterogeneity.py
poetry run python scripts/step8_thesis_figures.py
poetry run python scripts/step9_thesis_tables.py
```

### 3. Outputs

Figures are saved to `outputs/figures/`, summary tables to `outputs/tables/`.
