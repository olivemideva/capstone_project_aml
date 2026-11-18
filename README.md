# Kenyan demographic reconstruction and county prioritisation

This capstone will explore reconstructing incomplete county-level demographic profiles and
using them to compare simulated prioritisation strategies. The application is planned to use
FastAPI, a purple-themed Streamlit interface with light and dark modes, and Docker.

## Status

This repository is a scaffold. The data pipeline, models, API, interface, containers, and
deployment have not yet been implemented.

## Data

The planned source is the [HDX HAPI population dataset](https://data.humdata.org/dataset/hdx-hapi-population).
Download the relevant data separately and place it in `data/raw/`. Local files in `data/` are
ignored by Git. The loader and exact reproduction instructions will be added after the dataset
has been inspected.

## Setup

Create and activate a Python virtual environment, then install the dependencies needed for
development:

```powershell
python -m pip install -r requirements-api.txt -r requirements-ui.txt -r requirements-dev.txt
```

Additional modelling and data dependencies will be added when those components are implemented.

## Reproduction and deployment

Pipeline, Docker, deployment, architecture, and public service links will be documented as
those components are completed.
