"""Parse a plain-text / Fountain-style micro-drama script into a timed line list.

Supported:
  * ``NAME: dialogue``                      (colon style)
  * ``NAME`` on its own line + dialogue     (Fountain cue style)
  * ``INT./EXT.`` scene headings
  * ``[mm:ss]`` explicit timestamps (optional, bracketed)
  * everything else is treated as action / on-screen text

Timing is *estimated* from word counts unless explicit timestamps are given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DIALOGUE_WPS = 2.6      # spoken words per second (fast, vertical-drama pace)
ACTION_WPS = 3.5        # visual action read-through rate
HEADING_SECONDS = 1.0
MIN_LINE_SECONDS = 0.8
DIALOGUE_PAUSE = 0.3

_TS = re.compile(r"^\[(\d{1,2}):(\d{2})\]\s*")
_HEADING = re.compile(r"^(?:INT\.?/EXT\.?|INT\.?|EXT\.?|I/E\.?)\s")
_COLON_DIALOGUE = re.compile(
    r"^([A-Z][A-Z0-9 .'\-]{0,29}?)(?:\s*\(([^)]*)\))?\s*:\s*(\S.*)$"
)
_CUE = re.compile(r"^[A-Z][A-Z0-9 .'\-]{0,29}(?:\s*\([^)]*\))?$")
_PAREN_ONLY = re.compile(r"^\((.*)\)$")
_NON_SPEAKERS = {"SUPER", "TITLE", "CHYRON", "TEXT", "SFX", "SOUND", "MUSIC", "CAPTION", "ON SCREEN"}


@dataclass
class Line:
    idx: int
    kind: str                 # "dialogue" | "action" | "heading"
    text: str
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0
    explicit_ts: float | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "idx": self.idx, "kind": self.kind, "speaker": self.speaker,
            "text": self.text, "start": round(self.start, 2), "end": round(self.end, 2),
        }


@dataclass
class ParsedScript:
    lines: list[Line] = field(default_factory=list)
    total_seconds: float = 0.0
    has_timestamps: bool = False

    def content_lines(self) -> list[Line]:
        return [l for l in self.lines if l.kind != "heading"]

    def speakers(self) -> list[str]:
        seen: list[str] = []
        for l in self.lines:
            if l.speaker and l.speaker not in seen:
                seen.append(l.speaker)
        return seen


def _clean_speaker(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def _duration(kind: str, text: str) -> float:
    words = len(text.split())
    if kind == "heading":
        return HEADING_SECONDS
    if kind == "dialogue":
        return max(MIN_LINE_SECONDS, words / DIALOGUE_WPS + DIALOGUE_PAUSE)
    return max(MIN_LINE_SECONDS, words / ACTION_WPS)


def parse_script(text: str) -> ParsedScript:
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(raw)
    items: list[tuple[str, str | None, str, float | None]] = []
    pending_ts: float | None = None
    i = 0
    while i < n:
        line = raw[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith(("#", "//", "[[")):
            continue  # markdown headers / comments / fountain notes

        ts = pending_ts
        pending_ts = None
        m = _TS.match(line)
        if m:
            ts = int(m.group(1)) * 60 + int(m.group(2))
            line = line[m.end():].strip()
            if not line:
                pending_ts = ts
                continue

        if _HEADING.match(line) or (line.startswith(".") and len(line) > 1 and line[1].isalpha()):
            items.append(("heading", None, line.lstrip("."), ts))
            continue

        pm = _PAREN_ONLY.match(line)
        if pm:
            items.append(("action", None, pm.group(1).strip(), ts))
            continue

        cm = _COLON_DIALOGUE.match(line)
        if cm:
            speaker = _clean_speaker(cm.group(1))
            body = cm.group(3).strip()
            if speaker in _NON_SPEAKERS:
                items.append(("action", None, body, ts))
            else:
                items.append(("dialogue", speaker, body, ts))
            continue

        if (_CUE.match(line) and any(c.isalpha() for c in line)
                and i < n and raw[i].strip()
                and not _HEADING.match(raw[i].strip())):
            speaker = _clean_speaker(re.sub(r"\([^)]*\)", "", line))
            body_parts: list[str] = []
            while i < n and raw[i].strip():
                seg = raw[i].strip()
                i += 1
                if _PAREN_ONLY.match(seg):
                    continue
                body_parts.append(seg)
            body = " ".join(body_parts).strip()
            if body:
                kind = "action" if speaker in _NON_SPEAKERS else "dialogue"
                items.append((kind, None if kind == "action" else speaker, body, ts))
                continue

        items.append(("action", None, line, ts))

    parsed = ParsedScript()
    t = 0.0
    for idx, (kind, speaker, body, ts) in enumerate(items):
        if ts is not None:
            parsed.has_timestamps = True
            t = float(ts)
        start = t
        end = start + _duration(kind, body)
        parsed.lines.append(Line(idx=idx, kind=kind, text=body, speaker=speaker,
                                 start=start, end=end, explicit_ts=ts))
        t = end
    parsed.total_seconds = max((l.end for l in parsed.lines), default=0.0)
    return parsed


def rescale(parsed: ParsedScript, target_seconds: float) -> ParsedScript:
    """Stretch an *estimated* timeline so it matches a real video duration."""
    if parsed.has_timestamps or parsed.total_seconds <= 0 or target_seconds <= 0:
        return parsed
    factor = target_seconds / parsed.total_seconds
    for l in parsed.lines:
        l.start *= factor
        l.end *= factor
    parsed.total_seconds = target_seconds
    return parsed
