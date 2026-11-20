import pandas as pd

from src.analysis_workflow import model_figures


def test_model_figures_accept_diagnostic_columns():
    outputs = {
        "comparison": pd.DataFrame([{
            "pattern": "random", "rate": 0.1, "mae": 2.0, "model": "random_forest",
        }]),
        "diagnostics": pd.DataFrame([{
            "pattern": "random", "rate": 0.1, "cv_mae": 3.0, "mae": 2.0,
            "cost_equal": 2.0, "cost_under_double": 3.0, "cost_over_double": 3.0,
        }]),
        "tradeoffs": pd.DataFrame([{
            "child_coverage": 0.4, "elderly_coverage": 0.3,
            "dependency_burden_coverage": 0.2, "method": "weighted_sum",
            "selected_count": 1,
        }]),
    }
    assert set(model_figures(outputs)) == {"validation", "generalisation", "costs", "tradeoffs"}
