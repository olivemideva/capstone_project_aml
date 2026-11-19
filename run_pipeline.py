"""Run the reproducible local pipeline into a new, isolated output directory."""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.clean import clean_population
from src.data.load import load_kenya
from src.data.validate import validate_profiles
from src.models.autoencoder import sha256_file
from src.optimization.indicators import county_indicators

LOGGER = logging.getLogger("pipeline")
ROOT = Path(__file__).resolve().parent


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or set(config) != {
        "seed", "input", "output_root", "prioritisation"
    }:
        raise ValueError("Config must contain seed, input, output_root and prioritisation")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    for key in ("input", "output_root"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
    optimisation = config["prioritisation"]
    if not isinstance(optimisation, dict) or set(optimisation) != {"budget", "generations"}:
        raise ValueError("prioritisation must contain budget and generations")
    for key in ("budget", "generations"):
        if type(optimisation[key]) is not int or optimisation[key] < 1:
            raise ValueError(f"prioritisation.{key} must be a positive integer")
    return config


def resolved_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def command(module: str, *args: str) -> None:
    LOGGER.info("Running %s", module)
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True)


def run_pipeline(config: dict, config_path: Path, run_dir: Path) -> None:
    input_path = resolved_path(config["input"], ROOT)
    if not input_path.is_file():
        raise FileNotFoundError(f"Population CSV not found: {input_path}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if input_path == run_dir or run_dir in input_path.parents:
        raise ValueError("Run directory cannot contain the input CSV")

    seed = str(config["seed"])
    LOGGER.info("Loading local data from %s", input_path)
    raw_hash = sha256_file(input_path)
    profiles = clean_population(load_kenya(input_path))
    county_indicators(profiles)
    if config["prioritisation"]["budget"] > len(profiles):
        raise ValueError("Prioritisation budget exceeds the number of counties")

    run_dir.mkdir(parents=True, exist_ok=False)
    data = run_dir / "data"
    reports = run_dir / "reports"
    models = run_dir / "models"
    data.mkdir()
    reports.mkdir()
    processed = data / "kenya_county_population.csv"
    profiles.to_csv(processed, index=False)
    checks = validate_profiles(profiles)
    checks.to_csv(reports / "consistency_checks.csv", index=False)
    inconsistent = int(checks["status"].eq("inconsistent").sum())
    incomplete = int(checks["status"].eq("incomplete").sum())
    if inconsistent or incomplete:
        LOGGER.warning(
            "Source totals: %d inconsistent and %d incomplete checks; see %s",
            inconsistent, incomplete, reports / "consistency_checks.csv",
        )
    command(
        "src.features.prepare", "--input", processed, "--output-dir", data / "experiments",
        "--seed", seed,
    )
    for module, name in (
        ("src.models.baselines", "baselines"), ("src.models.autoencoder", "autoencoders")
    ):
        command(
            module, "--input", processed, "--model-dir", models / name,
            "--report-dir", reports / name, "--seed", seed,
        )
    command(
        "src.models.compare_models", "--baseline-report", reports / "baselines",
        "--autoencoder-report", reports / "autoencoders", "--model-root", models,
        "--output-dir", models / "selected",
    )
    command(
        "src.optimization.compare", "--input", processed, "--output-dir",
        reports / "prioritisation", "--seed", seed,
        "--budget", str(config["prioritisation"]["budget"]),
        "--generations", str(config["prioritisation"]["generations"]),
    )
    manifest = {
        "status": "complete",
        "input_sha256": raw_hash,
        "processed_sha256": sha256_file(processed),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "counties": len(profiles),
        "inconsistent_checks": inconsistent,
        "incomplete_checks": incomplete,
        "optimisation_input": "observed profiles",
        "test_evaluated": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Pipeline complete: %s", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--run-name", help="Unique directory name; defaults to a UTC timestamp"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config_path = args.config.resolve()
    config = read_config(config_path)
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if name in {".", ".."} or not name or Path(name).name != name or "/" in name or "\\" in name:
        parser.error("--run-name must be a single directory name")
    output_root = resolved_path(config["output_root"], ROOT)
    run_pipeline(config, config_path, output_root / name)


if __name__ == "__main__":
    main()
