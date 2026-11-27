"""Runtime configuration, read from environment variables."""
from dotenv import load_dotenv
load_dotenv()
"""Runtime configuration, read from environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# SQLite by default; set DATABASE_URL=postgresql+psycopg://user:pw@host/db for PostgreSQL.
_db_url = os.getenv("DATABASE_URL", "")
# Hosts such as Render hand out postgres:// or postgresql:// URLs; SQLAlchemy needs the psycopg driver named.
for _prefix in ("postgres://", "postgresql://"):
    if _db_url.startswith(_prefix):
        _db_url = "postgresql+psycopg://" + _db_url[len(_prefix):]
        break
DATABASE_URL = _db_url or f"sqlite:///{DATA_DIR / 'hooklab.db'}"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
# Check Google's docs for current model names; any multimodal Gemini model works.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Tried once if the main model times out / is overloaded (429, 500, 503, 504). Empty = no fallback.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
GEMINI_TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "180"))
# Optional reasoning limits (thinking models can take minutes otherwise). Only sent if set:
#   Gemini 3.x -> GEMINI_THINKING_LEVEL=low   (minimal|low|medium|high)
#   Gemini 2.5 -> GEMINI_THINKING_BUDGET=1024 (tokens; 0 disables on Flash)
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low") or None   # set empty to disable
GEMINI_THINKING_BUDGET = int(os.environ["GEMINI_THINKING_BUDGET"]) if os.getenv("GEMINI_THINKING_BUDGET") else None

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "600"))
MAX_SCRIPT_CHARS = int(os.getenv("MAX_SCRIPT_CHARS", "200000"))

# Share of the final score taken from the LLM when both engines ran (0..1).
LLM_WEIGHT = float(os.getenv("LLM_WEIGHT", "0.6"))