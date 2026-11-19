# Kenyan demographic reconstruction and county prioritisation

This capstone will explore reconstructing incomplete county-level demographic profiles and
using them to compare simulated prioritisation strategies. The application is planned to use
FastAPI, a purple-themed Streamlit interface with light and dark modes, and Docker.

## Status

The local data cleaning and exploratory analysis scripts are implemented. The models, API,
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
python -m pip install -r requirements-api.txt -r requirements-ui.txt -r requirements-dev.txt -r requirements-data.txt
```

Additional modelling dependencies will be added when those components are implemented.

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

## Reproduction and deployment

Pipeline, Docker, deployment, architecture, and public service links will be documented as
those components are completed.
