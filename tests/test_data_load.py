import hashlib
import logging

import pandas as pd
import pytest

from src.data.load import COUNTY_CODES, REQUIRED_COLUMNS, load_kenya


@pytest.fixture
def source_rows():
    return pd.DataFrame([
        {
            "location_code": "KEN", "admin_level": "1", "admin1_code": code,
            "admin1_name": f"County {code}", "gender": "f", "age_range": "all",
            "population": "1", "reference_period_start": "2019-01-01",
            "reference_period_end": "2019-12-31", "extra_column": "preserved",
        }
        for code in sorted(COUNTY_CODES)
    ])


def test_load_filters_and_logs_shape_and_checksum(tmp_path, source_rows, caplog):
    other_country = source_rows.iloc[[0]].copy()
    other_country["location_code"] = "UGA"
    other_level = source_rows.iloc[[0]].copy()
    other_level["admin_level"] = "0"
    path = tmp_path / "population.csv"
    pd.concat([source_rows, other_country, other_level]).to_csv(path, index=False)
    with caplog.at_level(logging.INFO, logger="src.data.load"):
        result = load_kenya(path)
    assert len(result) == 47
    assert result["extra_column"].eq("preserved").all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() in caplog.text
    assert "source shape: (49, 10)" in caplog.text
    assert "Kenya admin-1 shape: (47, 10)" in caplog.text
    assert "counties: 47" in caplog.text


def test_missing_file_and_columns_are_reported(tmp_path, source_rows):
    with pytest.raises(FileNotFoundError, match="Population CSV not found"):
        load_kenya(tmp_path / "missing.csv")
    path = tmp_path / "invalid.csv"
    source_rows.drop(columns="age_range").to_csv(path, index=False)
    with pytest.raises(ValueError, match="age_range"):
        load_kenya(path)


def test_rejects_missing_or_unexpected_county(tmp_path, source_rows):
    path = tmp_path / "incomplete.csv"
    source_rows.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="KE047"):
        load_kenya(path)
    source_rows.loc[0, "admin1_code"] = "KE999"
    source_rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="KE999"):
        load_kenya(path)


def test_rejects_no_kenya_admin1_rows(tmp_path, source_rows):
    path = tmp_path / "other_level.csv"
    source_rows["admin_level"] = "0"
    source_rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match="No Kenya admin-level-1"):
        load_kenya(path)


def test_required_columns_match_fixture(source_rows):
    assert set(REQUIRED_COLUMNS).issubset(source_rows.columns)
