"""Load Kenya county population records from a local HDX HAPI CSV."""

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/hdx_hapi_population_global_non_hrp.csv")
REQUIRED_COLUMNS = [
    "location_code", "admin_level", "admin1_code", "admin1_name", "gender",
    "age_range", "population", "reference_period_start", "reference_period_end",
]
COUNTY_CODES = {f"KE{i:03}" for i in range(1, 48)}
LOGGER = logging.getLogger(__name__)


def load_kenya(path: Path) -> pd.DataFrame:
    """Load Kenya admin-1 rows, preserving the source columns and missing values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Population CSV not found: {path}")

    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)

    columns = pd.read_csv(path, nrows=0).columns
    missing = set(REQUIRED_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    frames = []
    total_rows = 0
    for chunk in pd.read_csv(path, dtype=str, chunksize=100_000):
        total_rows += len(chunk)
        location = chunk["location_code"].str.strip().str.upper()
        level = chunk["admin_level"].str.strip()
        frames.append(chunk.loc[(location == "KEN") & (level == "1")])

    rows = pd.concat(frames, ignore_index=True)
    if rows.empty:
        raise ValueError(f"No Kenya admin-level-1 population records found in {path}")
    codes = rows["admin1_code"].astype("string").str.strip().str.upper()
    if codes.isna().any() or codes.eq("").any():
        raise ValueError("Kenya admin-level-1 records contain missing county codes")
    actual_codes = set(codes)
    if actual_codes != COUNTY_CODES:
        raise ValueError(
            "Expected all 47 Kenya county codes; "
            f"missing={sorted(COUNTY_CODES - actual_codes)}, "
            f"unexpected={sorted(actual_codes - COUNTY_CODES)}"
        )
    LOGGER.info(
        "Source: %s; SHA-256: %s; source shape: (%d, %d); "
        "Kenya admin-1 shape: %s; counties: %d",
        path, checksum.hexdigest(), total_rows, len(columns), rows.shape, len(actual_codes),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_kenya(args.input)


if __name__ == "__main__":
    main()
