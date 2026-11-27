# HookLab: Hook & Cliffhanger Retention Analyzer

Vertical micro-dramas live or die on the first 5 seconds and the last 5. HookLab lets a content or growth team audit an episode **before** spending on production or user-acquisition ads.

Upload a script (`.txt`, `.md`, `.fountain`, `.pdf`) and/or a rough cut (`.mp4`, `.mov`, `.webm`, ...). HookLab returns:

- **Hook strength** (0-100) for the first 5 seconds
- **Time-to-first-conflict** in seconds
- **Emotional-beat pacing** (0-100): density, escalation, reversals, dead air
- **Cliffhanger strength** (0-100) and type (question / reveal / interruption / ...)
- **Drop-off risk** (0-100), a per-5s risk curve, and flagged "where viewers leave" zones with fixes
- **Three A/B-testable hook rewrites**, each pulling a different lever and tailored to the genre (romance, thriller, mythology, revenge, fantasy, general), each with a falsifiable hypothesis and primary metric
- **Version history** for every draft, with deltas between drafts
- **A/B experiment tracking**: log impressions/conversions per arm and get Wilson CIs, Bonferroni-adjusted two-proportion z-tests, Bayesian P(best), a winner verdict, and a sample-size calculator

## Architecture

```
 script / video ──► parser ──► rule-based heuristics ─┐
                      │                               ├─► blend ─► scores + zones + 3 rewrites ─► SQL (versions, variants)
 video ──► ffmpeg ──► sampled frames ──► multimodal LLM ┘                                                │
 (dense in first 5s, sparse middle,         (forced tool-use → validated JSON)                           ▼
  last 2s)                                                                    experiments ─► arms ─► stats (z-test, Wilson, Bayes)
```

| Layer | Choice |
|---|---|
| API | Python 3.12, FastAPI, Pydantic v2 |
| Multimodal LLM | Anthropic Messages API (text + base64 image blocks), forced tool-use for schema-valid JSON |
| Video | ffmpeg / ffprobe (frame sampling, duration) |
| Storage | SQLAlchemy 2 → SQLite (default) or PostgreSQL via `DATABASE_URL` |
| UI | Dependency-free HTML/JS (served by FastAPI) |

### Design decisions

- **Hybrid scoring.** Deterministic, explainable rules run on every script; the LLM judges independently; scores are blended (`LLM_WEIGHT`, default 0.6). If the LLM is unavailable or errors, the system degrades to the rule-based engine and says so in `warnings` instead of failing.
- **Structured output.** The model must call a `submit_analysis` tool; the result is validated with Pydantic (scores clamped, exactly three rewrites enforced, missing ones padded from the offline engine).
- **Prompt-injection guard.** Script text and frames are treated as untrusted data in the system prompt.
- **Cost control.** Analyses are keyed by a content hash (script + video + genre); re-uploading an identical draft returns the cached result unless `force=true`.
- **Honest video handling.** Frames are sampled densely over the hook window (0, 0.5, 1, 2, 3, 4, 5s) and the ending. Audio is **not** transcribed: provide the script or a transcript for dialogue-level analysis.
- **Atomic experiment logging.** Event batches are all-or-nothing, with the invariant `conversions <= impressions` enforced inside a single SQL `UPDATE`.

### Scoring methodology (rule-based engine)

| Metric | How it's computed |
|---|---|
| Hook strength | 35 conflict in first 5s + 15 curiosity gap + 20 immediacy (first line ≤1s, ≤14 words, not a greeting) + 15 personal stakes + 15 pattern interrupt |
| Time-to-first-conflict | First line whose intensity ≥ threshold; scored on a piecewise curve (≤3s → 100, 5s → 85, 10s → 60, 20s → 30, 45s → 5) |
| Pacing | 30 beat density + 25 escalation + 20 reversals + 25 absence of flat stretches (≥10s with no beat) |
| Cliffhanger | open question, reveal, cut-off/interruption, final-line intensity, minus a penalty for resolved endings |
| Drop-off risk | `100 − (0.35·hook + 0.20·ttfc + 0.25·pacing + 0.20·cliffhanger)` |

> **Important:** these weights are hand-set rules of thumb, not fitted to retention data. Treat the scores as a *relative* audit tool (compare drafts and hooks), not as a forecast. To calibrate, regress the sub-scores against your own first-3s/5s/completion rates; the `/api/experiments` data model already stores what you need.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # needs ffmpeg on PATH for video
export ANTHROPIC_API_KEY=sk-ant-...          # optional; omit for the offline engine
uvicorn app.main:app --reload
# UI:   http://localhost:8000      API docs: http://localhost:8000/docs
```

Docker + PostgreSQL: `docker compose up --build`.

Try the two bundled drafts of the same episode (`samples/romance_weak_opening.txt` vs `romance_strong_opening.txt`): the weak opening scores hook 30 / first conflict 62.7s / risk 70; the reworked one scores hook 82 / first conflict 0s / risk 26.

## API

| Method & path | Purpose |
|---|---|
| `POST /api/analyze` | multipart: `title`, `genre`, `script_text` or `script` file, optional `video`, optional `episode_id`, `force` |
| `GET /api/episodes` · `GET /api/episodes/{id}/versions` | episodes and draft history |
| `GET /api/versions/{id}` | full analysis + 3 hook variants |
| `GET /api/episodes/{id}/compare?base=&head=` | score deltas between drafts |
| `POST /api/versions/{id}/experiments` | create an A/B test: control + variants A, B, C |
| `POST /api/experiments/{id}/events` | log impressions/conversions (atomic batch) |
| `GET /api/experiments/{id}/results` | rates, Wilson CIs, lift, adjusted p, P(best), verdict |
| `GET /api/stats/sample-size` | per-arm sample size for a baseline and relative lift |

```bash
curl -F title="Ep 12" -F genre=romance -F script=@samples/romance_weak_opening.txt localhost:8000/api/analyze
```

## Tests

```bash
pytest -q        # 32 tests
```

Covers the parser, every heuristic, the statistics (validated against reference values), the full API (uploads, PDF, real ffmpeg-generated video, caching, versioning, compare, experiments, validation), and the exact Anthropic request payload using a fake SDK client. The live LLM call itself needs an API key and is not exercised in CI.

## Script format

`NAME: line`, Fountain-style cues (`NAME` on its own line, dialogue below), `INT.`/`EXT.` headings, optional `[mm:ss]` timestamps, and anything else is action/on-screen text. Without timestamps, timing is estimated from word counts (≈2.6 spoken words/sec); when a video is attached, the estimated timeline is stretched to the real duration.

## Limitations / next steps

- Scores are uncalibrated (see above); next step is fitting them to real retention curves.
- No speech-to-text: add a transcription step (e.g. Whisper) for dialogue analysis from video alone.
- No auth or multi-tenancy; add before exposing publicly.
- PostgreSQL support is wired through SQLAlchemy and `docker-compose.yml` but the automated tests run on SQLite.
