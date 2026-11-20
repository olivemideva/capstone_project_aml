"""Export observed county indicators for API distribution."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.clean import PROCESSED_PATH
from src.data.load import COUNTY_CODES
from src.optimization.indicators import county_indicators


def export(source: Path, selection: Path, output: Path) -> None:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    selection_meta = json.loads(selection.read_text(encoding="utf-8"))
    if digest.hexdigest() != selection_meta["input_sha256"]:
        raise ValueError("Processed input does not match the selected model's training data")
    indicators = county_indicators(pd.read_csv(source))
    if set(indicators["admin1_code"]) != COUNTY_CODES:
        raise ValueError("Expected the 47 Kenya county codes")
    output.parent.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(output, index=False)
    print(f"Exported {len(indicators)} observed county indicators to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--selection", type=Path, default=Path("models/selected/selection.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/observed_county_indicators.csv")
    )
    args = parser.parse_args()
    export(args.input, args.selection, args.output)


if __name__ == "__main__":
    main()
