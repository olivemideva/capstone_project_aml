# Kenyan demographic reconstruction and county prioritisation

This capstone will explore reconstructing incomplete county-level demographic profiles and
using them to compare simulated prioritisation strategies. The application is planned to use
FastAPI, a purple-themed Streamlit interface with light and dark modes, and Docker.

## Status

The local data pipeline and tree-model baselines are implemented. The final model, API,
interface, containers, and deployment have not yet been implemented.

## Data

The planned source is the [HDX HAPI population dataset](https://data.humdata.org/dataset/hdx-hapi-population).
Download the global non-HRP population CSV separately and place it at
`data/raw/hdx_hapi_population_global_non_hrp.csv`. Local files in `data/` are ignored by Git.
The supplied extract contains Kenya's 47 admin-1 counties for the 2019 reference period. A
different extract can be specified with `--input`; it must contain all 47 county codes and one
reference period. Inspect its source and reference period before comparing results.

Run the standalone loader from the repository root to verify the local file:

```powershell
python -m src.data.load
```

For another file, use `python -m src.data.load --input path/to/file.csv`. The loader checks
required columns and the 47 Kenya admin-level-1 county codes, and logs the file's SHA-256,
source shape and selected Kenya shape. The cleaner uses the same loader and performs additional
demographic and reference-period checks.

## Setup

Create and activate a Python virtual environment, then install the dependencies needed for
development:

```powershell
python -m pip install -r requirements-api.txt -r requirements-ui.txt -r requirements-dev.txt -r requirements-data.txt -r requirements-model.txt -r requirements-optimization.txt
```

The model dependencies support the tree baselines and SHAP explanations.

## Cleaning and exploratory analysis

Run these commands from the repository root:

```powershell
python -m src.data.clean
python -m src.data.validate
python -m src.data.missingness
python -m src.data.eda
```

Cleaning filters the global file to Kenya admin level 1, validates county identifiers and
demographic categories, and produces one row per county in
`data/processed/kenya_county_population.csv`. Missing observations remain missing. Columns use
`population_{f|m|all}_{age}` with underscores in age ranges, `100_plus` for ages 100 and above,
and `all` for total age. Male, female, and combined-sex values are retained as reported; their
relationships are checked rather than assumed.

Validation writes `reports/consistency_checks.csv` and exits with an error if any complete
components do not add up. Checks with missing components are labelled incomplete, not
consistent. Missingness reports and interactive HTML figures are written to `reports/`. These
generated files and the processed dataset are ignored by Git and can be recreated locally.

## Features, splits and masking

After cleaning, run:

```powershell
python -m src.features.prepare
```

This writes nine experiments (three patterns and rates) for each county-grouped
train/validation/test split under `data/processed/experiments/`. The default seed is 42;
`--seed`, `--input`, and `--output-dir` can be used to change the defaults. With 47 counties,
the split has 33 training, 7 validation and 7 test counties. Each experiment writes separate
`inputs.csv`, `mask.csv` and `targets.csv` files, plus a shared `summary.json` recording the
requested rates and actual masked cell fractions. Targets are for scoring only, never predictors.
Do not combine differently masked versions of the same county across splits or experiments.

Random masking selects 10%, 20% or 40% of observed cells. Age-block masking selects that
percentage of available county-age blocks, hiding both available sex cells; county-block
masking selects that percentage of counties and hides all available age-sex cells in each.
Counts are rounded to the nearest whole unit, so actual cell fractions can differ, particularly
in the small validation and test splits. County-block cases have no observed demographic
inputs: reconstruction needs justified external information or an explicit baseline.

Features use only the 42 five-year/open-ended age-by-sex cells, not the source aggregate
totals, to avoid direct leakage through redundant totals. The denominator for shares is the sum
of these 42 cells, which can differ from a reported source total. County names and codes in the
output are identifiers for joining and auditing, not demographic model predictors. Male/female sex ratio, child
(0–14), working-age (15–64) and older (65+) shares, and dependency ratio are computed *after*
masking; incomplete components or zero denominators yield missing features. The audit checks
split isolation, masking, target separation and feature recomputation. It does not prove that
future model training is leakage-free: fit imputers, scalers and tuning only on training data,
and keep test data untouched until final evaluation.

## Tree baselines

After cleaning, run `python -m src.models.baselines`. Random Forest and XGBoost are tuned
independently for each masking pattern and rate using three-fold county-grouped CV on the
training split. Each training example predicts one masked age/sex cell from the remaining
masked profile and the target age/sex; missing inputs use a fixed -1 sentinel (populations are
non-negative). County identifiers, source totals and held-out values are not predictors. CV
uses only counties that actually have masked targets: at 10% county-block masking this means
only three training counties, so those results are particularly uncertain.

The script writes 18 model artifacts to `models/baselines/`, a validation results table and
training-sample SHAP plots to `reports/baselines/`, and input checksum/seed metadata in
`run.json`. These generated outputs are ignored by Git and can be recreated locally. The test
split is not evaluated here and must remain untouched until the final model is selected.
Consistency compares completed atomic cells against the original reported totals, including
existing source discrepancies; exact agreement is not assumed. MAE, RMSE and R² cover only
artificially held-out cells, not every cell in a reconstructed profile. Never load joblib
artifacts from untrusted sources.

## Masked autoencoder and selection

Install `requirements-model.txt`, then run these commands from the repository root after
cleaning the data:

```powershell
python -m src.models.baselines
python -m src.models.autoencoder
python -m src.models.compare_models
```

The autoencoder takes 42 masked age-sex values and 42 observed-value indicators. It predicts
non-negative age-sex cells, optimising masked-cell error plus a tunable consistency penalty.
The penalty uses reported age and gender totals **as training labels only**, not as inference
inputs. Consequently the autoencoder uses extra supervision that the tree baselines do not;
comparisons reflect that difference. No reported totals from validation or test counties are
used to predict their held-out cells. Validation consistency metrics compare predictions against
reported totals afterward, which themselves have documented source inconsistencies.

Hyperparameters are chosen with three-fold county-grouped CV on masked training counties; the
normalisation scale is fitted separately inside each training fold. The exact Phase 4 masks are
recreated using the same seed. The comparison rejects mismatched input checksums, seeds or
validation-mask fingerprints; rerun both training commands if either input or mask logic changes.

Selection chooses the family with the lowest **unweighted mean validation MAE across the nine
masking scenarios**, not the single easiest scenario. It saves that family's nine scenario-specific
artifacts, their comparison table and selection metadata to `models/selected/`. A caller must
select the corresponding pattern and rate; this is not yet a general-purpose deployed model.
The selection command refuses to overwrite a nonempty output directory; use `--output-dir` to
create a new selection. The final test split has not been evaluated. The validation set has only
seven counties and CV for 10% county-block has only three eligible training counties, so rankings
are preliminary. Autoencoder candidate artifacts and reports are regenerable and ignored by Git;
selected artifacts are intended to be tracked. Only load trusted model artifacts.

## Simulated county prioritisation

After cleaning, run:

```powershell
python -m src.optimization.compare
```

The default simulation uses the **observed** complete 2019 county profiles, not reconstructed
data. It selects up to ten counties, with one simulated unit of cost per selected county; there
is no currency or real allocation implied. The budget, seed, generations and input file can be
specified with `--budget`, `--seed`, `--generations` and `--input`.

Indicators use the observed age-sex cells: children are ages 0–14, working age 15–64, elderly
65+, and dependency burden is (children + elderly) / working-age population. Incomplete profiles
or a zero working-age denominator are rejected. The objectives maximise the selected counties'
share of all children, share of all elderly people, and share of the sum of county dependency
ratios. The third objective is an **illustrative burden-based equity proxy**, not a proven
measure of fairness, need, service delivery or deprivation. All objectives treat selecting a
county as covering its entire indicated population; this is a simplifying assumption, not a
service-coverage estimate. Age-cell totals may differ from source aggregate totals.

The weighted-sum baseline solves each of 66 fixed weight combinations exactly for the equal-cost
budget. Binary NSGA-III approximates a nondominated front; solutions are not guaranteed globally
Pareto-optimal. Both methods use the same objectives and budget. Their hypervolumes use the same
fixed `[0.01, 0.01, 0.01]` reference point in minimisation coordinates (objectives are negated).
The result depends on the weight grid, stochastic search, seed and hypervolume reference point;
it does not establish real-world superiority. This phase deliberately does not use or evaluate
the selected reconstruction models.

Generated `indicators.csv`, `tradeoffs.csv`, `summary.json` and an interactive `tradeoffs.html`
are saved to `reports/prioritisation/` and ignored by Git. County selections are stored as JSON
arrays of county codes in the table. The decision maker must assess assumptions and trade-offs.

## Reproduction and deployment

### FastAPI service

Install the requirements files listed in Setup. The API requires the selected tree-model files
under `models/selected/` and the derived observed indicators in
`models/observed_county_indicators.csv`. Both are included in the repository, so the raw CSV
is not required to serve predictions. Only load model files from trusted sources; joblib
deserialization executes code. The current service supports the selected Random Forest or
XGBoost artifacts, not an autoencoder-selected family. The serialized models were trained
with scikit-learn 1.9.1, pinned in `requirements-model.txt`; if changing that version, retrain
and reselect artifacts because scikit-learn does not guarantee cross-version compatibility.

Start the API from the project root:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive request schemas. `/health` confirms that
startup artifacts loaded, `/counties` lists county identifiers, and `/indicators/{county}`
returns observed 2019 indicators by county code. `/reconstruct` requires all 42 age-sex keys,
with `null` for missing values, plus an explicitly selected `pattern` (`random`, `age-block`
or `county-block`) and `rate` (10, 20 or 40). It preserves observed cells and returns the
predicted keys separately; predictions are experimental and the selected scenario is not
automatically inferred or validated against the submitted mask. Input values must be finite and
between zero and one trillion.

`/prioritise` accepts an equal-cost simulated budget, three objective weights for the
weighted-sum baseline, a random seed and a generation count (1–200). It returns the weighted
solution and an approximate NSGA-III front over the *observed* indicators, not reconstructed
profiles. At most one optimisation runs per API process at a time; a simultaneous request
receives HTTP 503. These are decision-support simulations, not actual allocations. Indicators
can be regenerated from a processed dataset matching `models/selected/selection.json` using
`python -m src.api.export_indicators`.

CORS is disabled unless `AML_CORS_ORIGINS` is set to a comma-separated list of trusted browser
origins. `AML_MODEL_DIR` and `AML_INDICATORS_PATH` override the default artifact locations.
`/metrics` exposes Prometheus request counts, errors (including 4xx), request latency and
successful reconstruction counts. Metrics are in-process and reset on restart; persistent
history requires an external collector. The API currently has no authentication or public
rate limiting and should not be treated as a production service.

### End-to-end pipeline

Place the raw CSV as described above, install the requirement files, then run from the project
root:

```powershell
python run_pipeline.py
```

`config.yaml` sets the input, output root, fixed seed, simulated budget and optimisation
generations. Each invocation creates a new timestamped directory under
`data/processed/pipeline-runs/`; use `--run-name NAME` for a specific new directory or
`--config PATH` for another configuration. Existing run directories are never overwritten.
Changing the input or output paths to absolute paths is supported.

The pipeline loads and cleans the data, records the consistency report, prepares masks, trains
both model families, selects using validation results, and optimises **observed** profiles. It
logs a warning and continues when source totals are inconsistent or incomplete; the standalone
`src.data.validate` command remains strict. Check `reports/consistency_checks.csv` inside the
run directory. The manifest is written only on successful completion and records hashes, seed,
configuration, and check counts. Failed runs may leave partial files without a manifest; choose
a new run name for a retry. The pipeline does not overwrite the previously committed
`models/selected/` artifacts, and it does not evaluate the held-out test split.

Docker, deployment, architecture, and public service links will be documented as those
components are completed.
