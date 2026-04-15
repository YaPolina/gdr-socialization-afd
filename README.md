# Does Socialization Under the GDR Increase AfD Vote Probability?

Master's thesis project using the German General Social Survey (ALLBUS) 2023 and causal inference methods (AIPW with overlap weights) to estimate the effect of growing up in the former GDR on support for the AfD.

## Data

The analysis uses **ALLBUS 2023** microdata, available by request from GESIS - Leibniz Institute for the Social Sciences: https://www.gesis.org (Study ZA8830, DOI: https://doi.org/10.4232/1.14544).

The raw dataset is not included due to licensing restrictions. After obtaining it, place the file at `data/raw/ALLBUS_2023.dta`.

## Repository Structure

```
├── scripts/                 Pipeline scripts (run sequentially)
│   ├── step1_build_model_afd.py
│   ├── step2_ate_overlap_eastyouth.py
│   ├── step3_balance_overlap_diagnostics.py
│   ├── step4_inference_multicluster_eastyouth.py
│   ├── step5_rosenbaum_delta_curve.py
│   ├── step6_spatial_placebo.py
│   ├── step7_cohort_heterogeneity.py
│   ├── step8_thesis_figures.py
│   └── step9_thesis_tables.py
│
├── src/                     Shared modules
│   ├── clean.py             Data cleaning and recoding utilities
│   ├── config.py            Paths, variable lists, covariate specifications
│   ├── io.py                I/O helpers
│   └── audit.py             Variable audit and checks
│
├── data/
│   ├── raw/                 Raw ALLBUS data (not tracked; see README inside)
│   └── external/            germany_states.geojson for map figures
│
└── outputs/
    ├── figures/             Thesis figures
    └── tables/              Summary tables
        └── placebo/         Spatial placebo draws and results
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
# Step 1: Clean raw ALLBUS data, recode variables, construct the analytic sample
poetry run python scripts/step1_build_model_afd.py

# Step 2: Estimate the ATE of GDR socialisation on AfD vote probability
#          via AIPW with overlap weights (GroupKFold cross-fitting)
poetry run python scripts/step2_ate_overlap_eastyouth.py

# Step 3: Check covariate balance: compute standardised mean differences (SMD)
#          before and after overlap weighting
poetry run python scripts/step3_balance_overlap_diagnostics.py

# Step 4: Cluster bootstrap inference with two-way clustering (xs11 × land)
poetry run python scripts/step4_inference_multicluster_eastyouth.py

# Step 5: Rosenbaum-style sensitivity analysis: compute how large an unmeasured
#          confounder would need to be to nullify the estimated effect
poetry run python scripts/step5_rosenbaum_delta_curve.py

# Step 6: Spatial placebo tests: randomly reassign East/West labels 1 000 times
#          and re-estimate to build a null distribution (--all runs all permutations)
poetry run python scripts/step6_spatial_placebo.py --all

# Step 7: Cohort heterogeneity: re-estimate the effect separately by age
#          at reunification to test whether socialisation intensity matters
poetry run python scripts/step7_cohort_heterogeneity.py

# Step 8: Generate all thesis figures from saved estimates
poetry run python scripts/step8_thesis_figures.py

# Step 9: Generate thesis summary tables
poetry run python scripts/step9_thesis_tables.py
```

### 3. Outputs

Figures are saved to `outputs/figures/`, summary tables to `outputs/tables/`.
