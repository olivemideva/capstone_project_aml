"""Request and response models for the demographic decision-support API."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.features.engineer import ATOMIC_COLUMNS


class ReconstructionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    population: dict[str, float | None]
    pattern: Literal["random", "age-block", "county-block"]
    rate: Literal[10, 20, 40]

    @field_validator("population", mode="before")
    @classmethod
    def validate_population(cls, value):
        if not isinstance(value, dict) or set(value) != set(ATOMIC_COLUMNS):
            raise ValueError("population must contain exactly the 42 age-sex keys")
        for item in value.values():
            if item is not None and (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or item > 1e12
                or not math.isfinite(item)
                or item < 0
            ):
                raise ValueError(
                    "population values must be finite numbers from 0 to 1e12, or null"
                )
        if all(item is not None for item in value.values()):
            raise ValueError("population must contain at least one missing value")
        return value


class ReconstructionResponse(BaseModel):
    population: dict[str, float]
    predicted_fields: list[str]
    model_family: str
    pattern: str
    rate: int
    note: str


class County(BaseModel):
    admin1_code: str
    admin1_name: str


class Indicators(County):
    children: float
    working_age: float
    elderly: float
    total: float
    child_share: float
    elderly_share: float
    dependency_ratio: float
    source: str


class PrioritiseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget: Annotated[int, Field(strict=True, ge=1, le=47)] = 10
    weights: tuple[float, float, float] = (1, 1, 1)
    generations: Annotated[int, Field(strict=True, ge=1, le=200)] = 50
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)] = 42

    @model_validator(mode="after")
    def validate_weights(self):
        if (
            not all(math.isfinite(x) and x >= 0 for x in self.weights)
            or not math.isfinite(sum(self.weights))
            or sum(self.weights) <= 0
        ):
            raise ValueError("weights must be finite, non-negative and not all zero")
        return self


class Solution(BaseModel):
    method: str
    selected_counties: list[County]
    selected_count: int
    child_coverage: float
    elderly_coverage: float
    dependency_burden_coverage: float


class PrioritiseResponse(BaseModel):
    budget: int
    weights: tuple[float, float, float]
    seed: int
    generations: int
    source: str
    note: str
    solutions: list[Solution]
