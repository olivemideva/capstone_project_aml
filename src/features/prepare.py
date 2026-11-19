"""Create reproducible county splits and masked reconstruction experiments."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.clean import PROCESSED_PATH
from src.features.audit import audit_masked_inputs, audit_splits
from src.features.engineer import ATOMIC_COLUMNS, engineer_features
from src.features.masking import PATTERNS, RATES, simulate_mask
from src.features.split import split_counties


def prepare_experiments(
    profiles: pd.DataFrame, output_dir: Path, seed: int = 42
) -> list[dict]:
    if profiles["admin1_code"].duplicated().any():
        raise ValueError("Preparation requires one profile per county")
    splits = split_counties(profiles, seed)
    audit_splits(splits)
    summary = []
    for pattern in PATTERNS:
        for rate in RATES:
            for name, split in splits.items():
                inputs, mask, targets = simulate_mask(split, pattern, rate, seed)
                features = engineer_features(inputs)
                audit_masked_inputs(split, inputs, mask, targets, features)
                directory = output_dir / pattern / str(int(rate * 100)) / name
                directory.mkdir(parents=True, exist_ok=True)
                identifiers = split[["admin1_code", "admin1_name"]]
                pd.concat([identifiers, inputs, features], axis=1).to_csv(
                    directory / "inputs.csv", index=False
                )
                pd.concat([identifiers[["admin1_code"]], mask], axis=1).to_csv(
                    directory / "mask.csv", index=False
                )
                pd.concat([identifiers[["admin1_code"]], targets], axis=1).to_csv(
                    directory / "targets.csv", index=False
                )
                observed = int(split[ATOMIC_COLUMNS].notna().sum().sum())
                masked = int(mask.to_numpy().sum())
                summary.append({
                    "pattern": pattern, "requested_rate": rate, "split": name,
                    "counties": len(split), "observed_cells": observed,
                    "masked_cells": masked, "actual_cell_fraction": masked / observed,
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"seed": seed, "experiments": summary}, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/experiments"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_experiments(pd.read_csv(args.input), args.output_dir, args.seed)
    print(f"Saved {len(summary)} split experiments and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
