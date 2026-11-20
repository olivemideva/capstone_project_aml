"""Shared notebook workflow and provenance-checked pipeline access."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px

from run_pipeline import read_config, resolved_path, run_pipeline
from src.data.analysis import exploratory_figures, exploratory_tables
from src.data.clean import clean_population
from src.data.load import load_kenya
from src.models.autoencoder import sha256_file
from src.models.diagnostics import evaluate_selected

ROOT = Path(__file__).resolve().parents[1]


def source_and_profiles(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = load_kenya(resolved_path(config["input"], ROOT))
    return source, clean_population(source)


def source_tables(source: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overview = pd.DataFrame([{
        "kenya_rows": len(source),
        "columns": len(source.columns),
        "counties": source["admin1_code"].nunique(),
        "duplicate_rows": int(source.duplicated().sum()),
        "reference_period_start": ", ".join(
            sorted(source["reference_period_start"].dropna().unique())
        ),
        "reference_period_end": ", ".join(sorted(source["reference_period_end"].dropna().unique())),
    }])
    return {
        "source_overview": overview,
        "source_fields": pd.DataFrame({
            "field": source.columns,
            "dtype": source.dtypes.astype(str).to_numpy(),
            "missing": source.isna().sum().to_numpy(),
            "distinct": source.nunique(dropna=True).to_numpy(),
        }),
        "source_categories": source.groupby(
            ["gender", "age_range"], dropna=False
        ).size().reset_index(name="rows"),
    }


def obtain_run(config_path: Path, existing: Path | None = None) -> Path:
    config_path = config_path.resolve()
    config = read_config(config_path)
    if existing is None:
        name = "notebook-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = resolved_path(config["output_root"], ROOT) / name
        run_pipeline(config, config_path, run_dir)
        return run_dir
    run_dir = resolved_path(str(existing), ROOT)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("Existing run is incomplete or was created with another configuration")
    if manifest["input_sha256"] != sha256_file(resolved_path(config["input"], ROOT)):
        raise ValueError("Existing run was created from another raw input")
    processed = run_dir / "data" / "kenya_county_population.csv"
    if manifest["processed_sha256"] != sha256_file(processed):
        raise ValueError("Existing run's processed data checksum does not match")
    return run_dir


def run_outputs(run_dir: Path) -> dict:
    processed = run_dir / "data" / "kenya_county_population.csv"
    model_dir = run_dir / "models" / "selected"
    selection = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (run_dir / "data" / "experiments" / "summary.json").read_text(encoding="utf-8")
    )
    optimisation = json.loads(
        (run_dir / "reports" / "prioritisation" / "summary.json").read_text(encoding="utf-8")
    )
    return {
        "manifest": manifest,
        "experiments": pd.DataFrame(summary["experiments"]),
        "selection": selection,
        "comparison": pd.read_csv(model_dir / "comparison.csv"),
        "diagnostics": evaluate_selected(processed, model_dir),
        "optimisation": optimisation,
        "tradeoffs": pd.read_csv(run_dir / "reports" / "prioritisation" / "tradeoffs.csv"),
    }


def model_figures(outputs: dict) -> dict:
    comparison = outputs["comparison"]
    diagnostics = outputs["diagnostics"].copy()
    diagnostics["scenario"] = diagnostics["pattern"] + " " + (
        diagnostics["rate"] * 100
    ).astype(int).astype(str) + "%"
    return {
        "validation": px.bar(
            comparison, x="pattern", y="mae", color="model", facet_col="rate",
            barmode="group", title="Validation MAE by model and masking scenario",
            color_discrete_sequence=["#7C3AED", "#A78BFA", "#4C1D95"],
        ),
        "generalisation": px.bar(
            diagnostics.melt(
                id_vars="scenario", value_vars=["cv_mae", "mae"],
                var_name="estimate", value_name="error_mae",
            ), x="scenario", y="error_mae", color="estimate", barmode="group",
            title="Training grouped-CV and validation MAE (different county samples)",
            color_discrete_sequence=["#7C3AED", "#A78BFA"],
        ),
        "costs": px.bar(
            diagnostics.melt(
                id_vars="scenario",
                value_vars=["cost_equal", "cost_under_double", "cost_over_double"],
                var_name="assumption", value_name="cost_per_cell",
            ), x="scenario", y="cost_per_cell", color="assumption", barmode="group",
            title="Illustrative asymmetric error cost per masked cell",
            color_discrete_sequence=["#7C3AED", "#A78BFA", "#4C1D95"],
        ),
        "tradeoffs": px.scatter_3d(
            outputs["tradeoffs"], x="child_coverage", y="elderly_coverage",
            z="dependency_burden_coverage", color="method", hover_data=["selected_count"],
            title="Simulated prioritisation trade-offs",
            color_discrete_sequence=["#7C3AED", "#A78BFA"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = obtain_run(args.config, args.run_dir)
    source, profiles = source_and_profiles(read_config(args.config))
    sources = source_tables(source)
    tables = exploratory_tables(profiles)
    outputs = run_outputs(run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in {**sources, **tables}.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    for name in ("experiments", "comparison", "diagnostics", "tradeoffs"):
        outputs[name].to_csv(args.output_dir / f"{name}.csv", index=False)
    for name in ("manifest", "selection", "optimisation"):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(outputs[name], indent=2) + "\n", encoding="utf-8"
        )
    for name, figure in {**exploratory_figures(tables), **model_figures(outputs)}.items():
        figure.write_html(args.output_dir / f"{name}.html", include_plotlyjs="cdn")
    print(f"Analysis reports saved to {args.output_dir}")


__all__ = [
    "ROOT", "read_config", "source_and_profiles", "source_tables", "exploratory_tables",
    "exploratory_figures", "obtain_run", "run_outputs", "model_figures", "evaluate_selected",
]

if __name__ == "__main__":
    main()
