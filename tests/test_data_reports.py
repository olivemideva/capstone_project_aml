import pandas as pd
import pytest

from src.data.clean import POPULATION_COLUMNS
from src.data.eda import write_figures
from src.data.missingness import missingness_reports
from src.data.validate import validate_profiles


@pytest.fixture
def profiles():
    row = {column: 0.0 for column in POPULATION_COLUMNS}
    row.update({
        "admin1_code": "KE001", "admin1_name": "County",
        "population_f_all": 4.0, "population_m_all": 6.0,
        "population_all_all": 10.0, "population_f_0_4": 4.0,
        "population_m_0_4": 6.0, "population_all_0_4": 10.0,
    })
    return pd.DataFrame([row])


def test_validation_detects_inconsistent_and_incomplete(profiles):
    checks = validate_profiles(profiles)
    assert checks.status.eq("consistent").all()
    profiles.loc[0, "population_f_0_4"] = 5
    profiles.loc[0, "population_m_5_9"] = float("nan")
    checks = validate_profiles(profiles).set_index("check")
    assert checks.loc["gender_0-4", "status"] == "inconsistent"
    assert checks.loc["gender_0-4", "difference"] == -1
    assert checks.loc["gender_5-9", "status"] == "incomplete"
    assert checks.loc["age_m", "status"] == "incomplete"


def test_missingness_and_figures(profiles, tmp_path):
    profiles.loc[0, "population_f_0_4"] = float("nan")
    county, column = missingness_reports(profiles)
    assert county.loc[0, "missing_count"] == 1
    assert column.set_index("column").loc["population_f_0_4", "missing_fraction"] == 1
    write_figures(profiles, tmp_path)
    assert (tmp_path / "county_population.html").is_file()
    assert (tmp_path / "county_missingness.html").is_file()


def test_reports_reject_missing_columns(profiles):
    with pytest.raises(ValueError, match="Missing profile columns"):
        missingness_reports(profiles.drop(columns="population_f_all"))
    with pytest.raises(ValueError, match="Missing profile columns"):
        validate_profiles(profiles.drop(columns="population_f_all"))