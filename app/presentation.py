"""Plain-language labels for information shown in the dashboard."""

from src.data.clean import population_column
from src.features.engineer import AGES

POPULATION_LABELS = {
    population_column(sex, age): f"{label}, ages {age.replace('+', ' and over')}"
    for age in AGES
    for sex, label in (("f", "Female"), ("m", "Male"))
}

INDICATOR_LABELS = {
    "admin1_code": "County code",
    "admin1_name": "County",
    "children": "Children aged 0–14",
    "working_age": "People aged 15–64",
    "elderly": "People aged 65 and over",
    "total": "Total population",
    "child_share": "Share of population aged 0–14",
    "elderly_share": "Share of population aged 65 and over",
    "dependency_ratio": "Dependants per working-age person",
    "source": "Data source",
}

METHOD_LABELS = {
    "weighted_sum": "Priority-weighted selection",
    "nsga3": "Alternative balanced selection",
}
