"""Pydantic request/response models and ORM -> dict serializers."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .db import Arm, Experiment, Version


class EventIn(BaseModel):
    arm_id: int
    impressions: int = Field(ge=0, le=100_000_000)
    conversions: int = Field(ge=0, le=100_000_000)


class EventBatch(BaseModel):
    events: list[EventIn] = Field(min_length=1, max_length=50)


class ExperimentCreate(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    metric: str | None = Field(default=None, max_length=80)


def variant_out(v) -> dict:
    return {"id": v.id, "label": v.label, "genre": v.genre, "angle": v.angle, "text": v.text,
            "rationale": v.rationale, "hypothesis": v.hypothesis, "primary_metric": v.primary_metric}


def version_out(v: Version, cached: bool | None = None, include_analysis: bool = True) -> dict:
    out = {
        "id": v.id, "episode_id": v.episode_id, "episode_title": v.episode.title,
        "version_number": v.version_number, "genre": v.genre, "engine": v.engine,
        "has_video": bool(v.video_path), "created_at": v.created_at.isoformat() if isinstance(v.created_at, datetime) else None,
        "scores": v.analysis["scores"], "variants": [variant_out(x) for x in v.variants],
    }
    if include_analysis:
        out["analysis"] = v.analysis
    if cached is not None:
        out["cached"] = cached
    return out


def arm_out(a: Arm) -> dict:
    return {"id": a.id, "label": a.label, "is_control": a.is_control, "variant_id": a.variant_id,
            "impressions": a.impressions, "conversions": a.conversions}


def experiment_out(e: Experiment) -> dict:
    return {"id": e.id, "version_id": e.version_id, "name": e.name, "metric": e.metric,
            "created_at": e.created_at.isoformat(), "arms": [arm_out(a) for a in e.arms]}
