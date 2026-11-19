import pandas as pd
import pytest

from src.data.clean import COUNTY_CODES, POPULATION_COLUMNS, clean_population, load_kenya


@pytest.fixture
def source_rows():
    return pd.DataFrame([
        {
            "location_code": "KEN", "admin_level": "1", "admin1_code": code,
            "admin1_name": f"County {code}", "gender": gender, "age_range": age,
            "population": str(population), "reference_period_start": "2019-01-01",
            "reference_period_end": "2019-12-31",
        }
        for code in sorted(COUNTY_CODES)
        for gender, age, population in [
            ("f", "all", 4), ("m", "all", 6), ("all", "all", 10),
            ("f", "0-4", 4), ("m", "0-4", 6), ("all", "0-4", 10),
        ]
    ])


def test_clean_preserves_missing_columns_and_normalizes(source_rows):
    source_rows.loc[0, "admin1_name"] = "  County   KE001  "
    source_rows.loc[0, "gender"] = " F "
    result = clean_population(source_rows)
    assert len(result) == 47
    assert result.loc[0, "admin1_name"] == "County KE001"
    assert result.loc[0, "population_f_all"] == 4
    assert pd.isna(result.loc[0, "population_f_5_9"])
    assert list(result.columns[2:]) == POPULATION_COLUMNS


def test_clean_rejects_duplicates_and_multiple_periods(source_rows):
    with pytest.raises(ValueError, match="Duplicate"):
        clean_population(pd.concat([source_rows, source_rows.iloc[[0]]]))
    source_rows.loc[0, "reference_period_start"] = "2020-01-01"
    with pytest.raises(ValueError, match="reference period"):
        clean_population(source_rows)


def test_clean_rejects_invalid_values_and_missing_county(source_rows):
    source_rows.loc[0, "population"] = "-1"
    with pytest.raises(ValueError, match="non-negative"):
        clean_population(source_rows)
    with pytest.raises(ValueError, match="47 county codes"):
        clean_population(source_rows[source_rows.admin1_code != "KE001"])


def test_load_filters_global_rows(tmp_path, source_rows):
    other = source_rows.iloc[[0]].copy()
    other.loc[:, "location_code"] = "UGA"
    path = tmp_path / "population.csv"
    pd.concat([source_rows, other]).to_csv(path, index=False)
    assert len(load_kenya(path)) == len(source_rows)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_kenya(tmp_path / "missing.csv")