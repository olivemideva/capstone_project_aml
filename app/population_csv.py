"""Read and reconstruct uploaded county population tables."""

import csv
import io
import math

import pandas as pd

from app import api_client
from src.features.engineer import ATOMIC_COLUMNS

MAX_ROWS = 100
MAX_BYTES = 5 * 1024 * 1024
MISSING_VALUES = {"", "na", "n/a", "nan", "null"}


def read_population_csv(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_BYTES:
        raise ValueError("The file must be 5 MB or smaller.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the CSV as UTF-8 and try again.") from exc
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if len(row) != len(header):
                raise ValueError(f"CSV line {reader.line_num} has the wrong number of columns.")
            if count > MAX_ROWS:
                raise ValueError(f"Please upload at most {MAX_ROWS} rows at a time.")
    except (StopIteration, csv.Error) as exc:
        raise ValueError("The file needs a valid CSV header.") from exc
    if len(header) != len(set(header)):
        raise ValueError("The file contains duplicate column names.")
    missing_columns = [column for column in ATOMIC_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(
            f"The file is missing {len(missing_columns)} required population columns. "
            "Use the template to check their names."
        )
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Unable to read the CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The file needs at least one county row.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Please upload at most {MAX_ROWS} rows at a time.")
    for column in ATOMIC_COLUMNS:
        values = frame[column].str.strip()
        missing = values.str.lower().isin(MISSING_VALUES)
        numeric = pd.to_numeric(values.where(~missing), errors="coerce")
        invalid = ~missing & (numeric.isna() | ~numeric.map(math.isfinite))
        invalid |= ~missing & ((numeric < 0) | (numeric > 1e12))
        if invalid.any():
            row = int(invalid.to_numpy().nonzero()[0][0]) + 1
            raise ValueError(
                f"Data row {row}, column {column}: use a number between 0 and "
                "1,000,000,000,000, or leave it blank if missing."
            )
        frame[column] = numeric
    return frame


def reconstruct_rows(frame: pd.DataFrame, pattern: str, rate: int) -> tuple[pd.DataFrame, list]:
    completed = frame.copy()
    changes = []
    for index, row in frame.iterrows():
        missing = [column for column in ATOMIC_COLUMNS if pd.isna(row[column])]
        if not missing:
            continue
        population = {
            column: None if pd.isna(row[column]) else float(row[column])
            for column in ATOMIC_COLUMNS
        }
        try:
            result = api_client.reconstruct(
                {"population": population, "pattern": pattern, "rate": rate}
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Could not estimate data row {index + 1}: {exc}") from exc
        for column in missing:
            completed.at[index, column] = result["population"][column]
            changes.append((index, column, result["population"][column]))
    return completed, changes
