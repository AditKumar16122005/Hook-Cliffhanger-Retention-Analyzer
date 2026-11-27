"""Analysis pipeline: parse -> heuristics -> (optional) multimodal LLM -> blend -> persist."""
from __future__ import annotations

import hashlib
import logging
import os

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config, llm, video
from .db import Episode, HookVariant, Version
from .heuristics import analyze_script, dropoff_from_scores, ttfc_score
from .rewrites import normalize_genre, offline_rewrites
from .script_parser import ParsedScript, parse_script, rescale

log = logging.getLogger("hooklab.service")


class AnalysisError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def content_hash(script_text: str | None, video_hash: str | None, genre: str) -> str:
    h = hashlib.sha256()
    h.update((script_text or "").encode())
    h.update(b"\x00")
    h.update((video_hash or "").encode())
    h.update(b"\x00")
    h.update(genre.encode())
    return h.hexdigest()


def _blend(h: int | None, l: int | None, w: float) -> int:
    if h is not None and l is not None:
        return round(w * l + (1 - w) * h)
    return int(h if h is not None else l)  # type: ignore[arg-type]


def _build_result(parsed: ParsedScript | None, heur: dict | None, ai: llm.LLMAnalysis | None,
                  have_frames: bool, warnings: list[str]) -> dict:
    w = config.LLM_WEIGHT
    hook = _blend(heur["hook"]["score"] if heur else None, ai.hook_score if ai else None, w)
    pacing = _blend(heur["pacing"]["score"] if heur else None, ai.pacing_score if ai else None, w)
    cliff = _blend(heur["cliffhanger"]["score"] if heur else None, ai.cliffhanger_score if ai else None, w)

    if ai and have_frames and ai.conflict_time_s is not None:
        ttfc = float(ai.conflict_time_s)
    elif heur and heur["ttfc"]["seconds"] is not None:
        ttfc = float(heur["ttfc"]["seconds"])
    elif ai and ai.conflict_time_s is not None:
        ttfc = float(ai.conflict_time_s)
    else:
        ttfc = None
    ttfc_sc = ttfc_score(ttfc)

    zones = []
    if heur:
        zones += [{**z, "source": "rules"} for z in heur["risk_zones"]]
    if ai:
        zones += [{**z.model_dump(), "source": "llm"} for z in ai.dropoff_zones]
    zones.sort(key=lambda z: (z["start_s"], z["end_s"]))

    if ai and ai.beats:
        beats = [b.model_dump() for b in sorted(ai.beats, key=lambda b: b.t_s)]
    elif heur:
        beats = [{"t_s": x["t"], "label": "", "intensity": x["intensity"]} for x in heur["windows"]]
    else:
        beats = []

    if ai and ai.top_fixes:
        fixes = ai.top_fixes[:5]
    else:
        fixes = list(dict.fromkeys(z["fix"] for z in zones if z.get("fix")))[:5]

    engine = "hybrid" if (heur and ai) else ("llm" if ai else "heuristic")
    duration = heur["duration_s"] if heur else (max((b["t_s"] for b in beats), default=0.0))
    return {
        "engine": engine,
        "duration_s": duration,
        "timeline_estimated": bool(heur and heur["timeline_estimated"] and not have_frames),
        "scores": {
            "hook_strength": hook,
            "time_to_first_conflict_s": ttfc,
            "ttfc_score": ttfc_sc,
            "pacing": pacing,
            "cliffhanger": cliff,
            "dropoff_risk": dropoff_from_scores(hook, ttfc_sc, pacing, cliff),
        },
        "hook_rationale": ai.hook_rationale if ai else "",
        "conflict_description": (ai.conflict_description if ai and ai.conflict_description
                                 else (heur["ttfc"]["excerpt"] if heur else "")),
        "cliffhanger_type": (ai.cliffhanger_type if ai and ai.cliffhanger_type
                             else (heur["cliffhanger"]["type"] if heur else "")),
        "cliffhanger_rationale": ai.cliffhanger_rationale if ai else "",
        "beats": beats,
        "risk_curve": heur["risk_curve"] if heur else [],
        "risk_zones": zones,
        "top_fixes": fixes,
        "flags": heur["flags"] if heur else [],
        "warnings": warnings,
        "heuristics": heur,
        "llm": ai.model_dump(exclude={"hook_rewrites"}) if ai else None,
        "script_lines": [l.to_dict() for l in parsed.lines[:400]] if parsed else [],
    }


def _variants(parsed: ParsedScript | None, ai: llm.LLMAnalysis | None, genre: str) -> list[dict]:
    out: list[dict] = []
    if ai:
        for rw in ai.hook_rewrites[:3]:
            out.append({**rw.model_dump(), "genre": genre})
    if len(out) < 3 and parsed:
        for v in offline_rewrites(parsed, genre)[len(out):3]:
            out.append(v)
    for label, v in zip("ABC", out):
        v["label"] = label
    return out[:3]


def run_analysis(db: Session, *, title: str, genre: str | None, episode_id: int | None,
                 script_text: str | None, video_path: str | None, video_hash: str | None,
                 force: bool = False) -> tuple[Version, bool]:
    genre = normalize_genre(genre)
    script_text = (script_text or "").strip() or None
    if not script_text and not video_path:
        raise AnalysisError(422, "Provide a script (text or file), a video, or both.")
    if script_text and len(script_text) > config.MAX_SCRIPT_CHARS:
        raise AnalysisError(413, f"Script exceeds {config.MAX_SCRIPT_CHARS} characters.")

    episode = None
    if episode_id is not None:
        episode = db.get(Episode, episode_id)
        if episode is None:
            raise AnalysisError(404, f"Episode {episode_id} not found.")
        chash = content_hash(script_text, video_hash, genre)
        if not force:
            hit = db.scalar(select(Version).where(Version.episode_id == episode.id,
                                                  Version.content_hash == chash))
            if hit:
                return hit, True
    else:
        chash = content_hash(script_text, video_hash, genre)

    warnings: list[str] = []
    parsed: ParsedScript | None = None
    heur: dict | None = None
    duration: float | None = None
    frames: list[tuple[float, bytes]] | None = None

    if video_path:
        try:
            duration = video.probe_duration(video_path)
        except video.VideoError as exc:
            raise AnalysisError(422, str(exc))

    if script_text:
        parsed = parse_script(script_text)
        if duration:
            parsed = rescale(parsed, duration)
        try:
            heur = analyze_script(parsed)
        except ValueError as exc:
            raise AnalysisError(422, str(exc))

    if video_path and llm.analyzer.enabled:
        try:
            frames = video.extract_frames(video_path, video.frame_times(duration or 0))
        except video.VideoError as exc:
            raise AnalysisError(422, str(exc))
    elif video_path:
        if not script_text:
            raise AnalysisError(503, "Video-only analysis needs a multimodal LLM. Set GEMINI_API_KEY, "
                                     "or also provide the script/transcript.")
        warnings.append("Video frames were not analyzed because no LLM is configured; scores use the script only.")

    ai: llm.LLMAnalysis | None = None
    if llm.analyzer.enabled:
        try:
            ai = llm.analyzer.analyze(title=title, genre=genre, script_text=script_text,
                                      heuristics=heur, frames=frames, duration=duration)
        except Exception as exc:  # network, API, or validation failure
            log.exception("LLM analysis failed")
            if not script_text:
                raise AnalysisError(502, f"LLM analysis failed: {type(exc).__name__}")
            warnings.append(f"LLM analysis failed ({type(exc).__name__}); showing rule-based results only.")
    else:
        warnings.append("No GEMINI_API_KEY configured: running the offline rule-based engine.")

    result = _build_result(parsed, heur, ai, frames is not None, warnings)
    variants = _variants(parsed, ai, genre)
    if len(variants) < 3:
        raise AnalysisError(502, "Could not generate hook rewrites.")

    if episode is None:
        episode = Episode(title=title.strip()[:200] or "Untitled episode", genre=genre)
        db.add(episode)
        db.flush()
    next_no = (db.scalar(select(func.max(Version.version_number)).where(Version.episode_id == episode.id)) or 0) + 1
    s = result["scores"]
    ver = Version(
        episode_id=episode.id, version_number=next_no, genre=genre, content_hash=chash,
        script_text=script_text, video_path=video_path, engine=result["engine"],
        hook_score=s["hook_strength"], ttfc_seconds=s["time_to_first_conflict_s"],
        pacing_score=s["pacing"], cliffhanger_score=s["cliffhanger"], dropoff_risk=s["dropoff_risk"],
        analysis=result,
    )
    db.add(ver)
    db.flush()
    for v in variants:
        db.add(HookVariant(version_id=ver.id, label=v["label"], genre=v.get("genre", genre),
                           angle=v["angle"][:120], text=v["text"], rationale=v.get("rationale", ""),
                           hypothesis=v.get("hypothesis", ""),
                           primary_metric=(v.get("primary_metric") or "")[:80]))
    db.commit()
    db.refresh(ver)
    return ver, False


def safe_remove(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass