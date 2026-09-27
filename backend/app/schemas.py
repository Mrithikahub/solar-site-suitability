"""Request / response models for the Solar Site Suitability API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ACRE_M2 = 4046.8564224


class EnergyParams(BaseModel):
    area_m2: float = Field(ACRE_M2, gt=0, le=1e9, description="PV array area in m² (default 1 acre)")
    efficiency: float = Field(0.20, gt=0, le=0.5, description="Module efficiency (0-1)")
    performance_ratio: float = Field(0.75, gt=0, le=1, description="System performance ratio (0-1)")


class PredictRequest(EnergyParams):
    lat: float = Field(..., ge=-90, le=90, examples=[9.35])
    lon: float = Field(..., ge=-180, le=180, examples=[78.38])


class Site(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: str | None = Field(None, max_length=120)


class CompareRequest(EnergyParams):
    sites: list[Site] = Field(..., min_length=2, max_length=5)

    @field_validator("sites")
    @classmethod
    def distinct(cls, v: list[Site]) -> list[Site]:
        keys = {(round(s.lat, 4), round(s.lon, 4)) for s in v}
        if len(keys) != len(v):
            raise ValueError("sites must be distinct locations")
        return v


Scope = Literal["global", "tamil_nadu"]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
