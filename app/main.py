"""FastAPI app: upload -> analyze -> version history -> A/B experiments."""
from __future__ import annotations

import hashlib
import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from . import config, llm, video
from .db import Arm, Episode, Experiment, Version, get_db, init_db
from .schemas import (EventBatch, ExperimentCreate, experiment_out, version_out)
from .service import AnalysisError, run_analysis, safe_remove
from .stats import required_sample_size, summarize_experiment

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SCRIPT_EXTS = {".txt", ".md", ".fountain", ".pdf"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="HookLab: Hook & Cliffhanger Retention Analyzer", version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ helpers
def _read_script_upload(upload: UploadFile) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in SCRIPT_EXTS:
        raise HTTPException(415, f"Unsupported script type '{ext}'. Use {', '.join(sorted(SCRIPT_EXTS))}.")
    data = upload.file.read(config.MAX_SCRIPT_CHARS * 4 + 1)
    if len(data) > config.MAX_SCRIPT_CHARS * 4:
        raise HTTPException(413, "Script file is too large.")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            raise HTTPException(422, "Could not read text from that PDF (is it scanned/image-only?).")
    return data.decode("utf-8-sig", errors="replace")


def _save_video_upload(upload: UploadFile) -> tuple[str, str]:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(415, f"Unsupported video type '{ext}'. Use {', '.join(sorted(VIDEO_EXTS))}.")
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    sha, size = hashlib.sha256(), 0
    try:
        with open(dest, "wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Video exceeds {config.MAX_UPLOAD_MB} MB.")
                sha.update(chunk)
                out.write(chunk)
    except Exception:
        safe_remove(str(dest))
        raise
    return str(dest), sha.hexdigest()


def _get_or_404(db: Session, model, pk: int, what: str):
    obj = db.get(model, pk)
    if obj is None:
        raise HTTPException(404, f"{what} {pk} not found.")
    return obj


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok", "llm_enabled": llm.analyzer.enabled, "model": config.GEMINI_MODEL,
            "ffmpeg": video.ffmpeg_available(), "database": config.DATABASE_URL.split(":", 1)[0]}


@app.post("/api/analyze")
def analyze(
    title: str = Form(..., min_length=1, max_length=200),
    genre: str = Form("general"),
    episode_id: str | None = Form(None),
    script_text: str | None = Form(None),
    force: bool = Form(False),
    script: UploadFile | None = File(None),
    video_file: UploadFile | None = File(None, alias="video"),
    db: Session = Depends(get_db),
):
    ep_id: int | None = None
    if episode_id not in (None, ""):
        try:
            ep_id = int(episode_id)  # type: ignore[arg-type]
        except ValueError:
            raise HTTPException(422, "episode_id must be an integer.")

    text = script_text
    if script is not None and script.filename:
        text = _read_script_upload(script)

    video_path = video_hash = None
    if video_file is not None and video_file.filename:
        video_path, video_hash = _save_video_upload(video_file)

    try:
        ver, cached = run_analysis(db, title=title, genre=genre, episode_id=ep_id, script_text=text,
                                   video_path=video_path, video_hash=video_hash, force=force)
    except AnalysisError as exc:
        safe_remove(video_path)
        raise HTTPException(exc.status_code, exc.message)
    except Exception:
        safe_remove(video_path)
        raise
    if cached:
        safe_remove(video_path)   # identical content already stored; drop the duplicate upload
    return version_out(ver, cached=cached)


@app.get("/api/episodes")
def list_episodes(db: Session = Depends(get_db)):
    eps = db.scalars(select(Episode).order_by(Episode.id.desc())).all()
    return [{"id": e.id, "title": e.title, "genre": e.genre, "versions": len(e.versions),
             "latest_hook_score": e.versions[-1].hook_score if e.versions else None} for e in eps]


@app.get("/api/episodes/{episode_id}/versions")
def list_versions(episode_id: int, db: Session = Depends(get_db)):
    ep = _get_or_404(db, Episode, episode_id, "Episode")
    return [version_out(v, include_analysis=False) for v in ep.versions]


@app.get("/api/versions/{version_id}")
def get_version(version_id: int, db: Session = Depends(get_db)):
    return version_out(_get_or_404(db, Version, version_id, "Version"))


@app.get("/api/episodes/{episode_id}/compare")
def compare(episode_id: int, base: int = Query(...), head: int = Query(...), db: Session = Depends(get_db)):
    _get_or_404(db, Episode, episode_id, "Episode")
    a, b = _get_or_404(db, Version, base, "Version"), _get_or_404(db, Version, head, "Version")
    if a.episode_id != episode_id or b.episode_id != episode_id:
        raise HTTPException(422, "Both versions must belong to this episode.")
    sa, sb = a.analysis["scores"], b.analysis["scores"]
    keys = ["hook_strength", "time_to_first_conflict_s", "pacing", "cliffhanger", "dropoff_risk"]
    delta = {}
    for k in keys:
        x, y = sa.get(k), sb.get(k)
        delta[k] = None if x is None or y is None else round(y - x, 1)
    return {"base": {"id": a.id, "version_number": a.version_number, "scores": sa},
            "head": {"id": b.id, "version_number": b.version_number, "scores": sb}, "delta": delta}


@app.post("/api/versions/{version_id}/experiments", status_code=201)
def create_experiment(version_id: int, body: ExperimentCreate | None = None, db: Session = Depends(get_db)):
    ver = _get_or_404(db, Version, version_id, "Version")
    body = body or ExperimentCreate()
    exp = Experiment(version_id=ver.id,
                     name=body.name or f"{ver.episode.title} v{ver.version_number} hook test",
                     metric=body.metric or "3-second view rate")
    db.add(exp)
    db.flush()
    db.add(Arm(experiment_id=exp.id, label="Control (original opening)", is_control=True))
    for v in ver.variants:
        db.add(Arm(experiment_id=exp.id, variant_id=v.id, label=f"{v.label}: {v.angle}"))
    db.commit()
    db.refresh(exp)
    return experiment_out(exp)


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    return experiment_out(_get_or_404(db, Experiment, experiment_id, "Experiment"))


@app.post("/api/experiments/{experiment_id}/events")
def log_events(experiment_id: int, body: EventBatch, db: Session = Depends(get_db)):
    """Add impressions/conversions to arms. Atomic; the whole batch is rejected if any row is invalid."""
    exp = _get_or_404(db, Experiment, experiment_id, "Experiment")
    arm_ids = {a.id for a in exp.arms}
    for ev in body.events:
        if ev.arm_id not in arm_ids:
            db.rollback()
            raise HTTPException(404, f"Arm {ev.arm_id} is not part of experiment {experiment_id}.")
        res = db.execute(
            update(Arm)
            .where(Arm.id == ev.arm_id, Arm.conversions + ev.conversions <= Arm.impressions + ev.impressions)
            .values(impressions=Arm.impressions + ev.impressions,
                    conversions=Arm.conversions + ev.conversions)
        )
        if res.rowcount == 0:
            db.rollback()
            raise HTTPException(422, f"Arm {ev.arm_id}: conversions cannot exceed impressions.")
    db.commit()
    db.expire_all()
    return experiment_out(_get_or_404(db, Experiment, experiment_id, "Experiment"))


@app.get("/api/experiments/{experiment_id}/results")
def experiment_results(experiment_id: int, min_per_arm: int = Query(1000, ge=1),
                       db: Session = Depends(get_db)):
    exp = _get_or_404(db, Experiment, experiment_id, "Experiment")
    arms = [{"id": a.id, "label": a.label, "is_control": a.is_control, "variant_id": a.variant_id,
             "impressions": a.impressions, "conversions": a.conversions} for a in exp.arms]
    summary = summarize_experiment(arms, min_per_arm=min_per_arm)
    return {"experiment_id": exp.id, "name": exp.name, "metric": exp.metric, **summary}


@app.get("/api/stats/sample-size")
def sample_size(baseline: float = Query(..., gt=0, lt=1), relative_mde: float = Query(..., gt=0),
                alpha: float = Query(0.05, gt=0, lt=0.5), power: float = Query(0.8, gt=0.5, lt=0.999)):
    try:
        n = required_sample_size(baseline, relative_mde, alpha, power)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"per_arm": n, "baseline": baseline, "relative_mde": relative_mde, "alpha": alpha, "power": power}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")