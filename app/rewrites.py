"""Offline hook-rewrite engine: genre playbooks + the script's own strongest lines.

Used when no LLM is configured (or the LLM returns fewer than three variants).
Output is always exactly three A/B-testable openings, each pulling a *different* lever.
"""
from __future__ import annotations

from .heuristics import line_intensity
from .script_parser import Line, ParsedScript

GENRES = ("romance", "thriller", "mythology", "revenge", "fantasy", "general")

PLAYBOOKS: dict[str, dict[str, str]] = {
    "romance": {
        "super": "Three days before the wedding...",
        "question": "Tell me the truth. Who is she to you?",
        "flash": "The night everything changed",
        "guidance": "Stakes = betrayal, forbidden desire, public humiliation. Open on intimate tension in face and voice; leave the relationship status unclear.",
    },
    "thriller": {
        "super": "11 minutes until the call...",
        "question": "Don't turn around. They are already inside.",
        "flash": "Six hours earlier",
        "guidance": "Open mid-threat, withhold information, run a ticking clock. Sound design does half the work.",
    },
    "mythology": {
        "super": "The prophecy was never about the king.",
        "question": "Which one of us was the sacrifice meant for?",
        "flash": "Before the gods fell silent",
        "guidance": "Invoke prophecy, curse or god-vs-mortal conflict; show the cost of power; lead with spectacle.",
    },
    "revenge": {
        "super": "Seven years ago, they ruined me. Tonight, I return.",
        "question": "Do you remember me? You should.",
        "flash": "The night they thought I was gone",
        "guidance": "Establish the wrong and the debt fast; open on the moment the power balance flips.",
    },
    "fantasy": {
        "super": "The last gate falls at dawn.",
        "question": "If I tell you who you really are, you can never go back.",
        "flash": "Before the crown chose her",
        "guidance": "Lead with a rule-breaking image or a secret identity; make the world's stakes personal.",
    },
    "general": {
        "super": "One second later, everything changed.",
        "question": "What did you do?",
        "flash": "Earlier that day",
        "guidance": "Open on the conflict peak, a charged question, or a flash-forward; cut set-up.",
    },
}


def normalize_genre(genre: str | None) -> str:
    g = (genre or "general").strip().lower()
    return g[:40] or "general"


def playbook(genre: str) -> dict[str, str]:
    return PLAYBOOKS.get(normalize_genre(genre), PLAYBOOKS["general"])


def _speaker_lines(parsed: ParsedScript) -> list[Line]:
    return [l for l in parsed.lines if l.kind == "dialogue" and l.text.strip()]


def _short(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def offline_rewrites(parsed: ParsedScript, genre: str) -> list[dict]:
    pb = playbook(genre)
    g = normalize_genre(genre)
    dialogue = _speaker_lines(parsed)
    speakers = parsed.speakers() or ["HE", "SHE"]
    a = speakers[0]
    b = speakers[1] if len(speakers) > 1 else ("SHE" if a != "SHE" else "HE")

    # Peak line: highest intensity dialogue line, preferring ones *after* the current hook window.
    pool = [l for l in dialogue if l.start >= 5.0] or dialogue
    peak = max(pool, key=lambda l: (line_intensity(l.text, l.kind), -l.start), default=None)
    last = dialogue[-1] if dialogue else None

    peak_sp = (peak.speaker if peak else a) or a
    other = b if peak_sp == a else a
    peak_text = _short(peak.text) if peak else pb["question"]
    last_text = _short(last.text) if last else pb["question"]
    last_sp = (last.speaker if last else a) or a

    variants = [
        {
            "angle": "Conflict cold-open",
            "text": (f'[0:00] SMASH CUT IN, mid-confrontation, handheld close-up.\n'
                     f'{peak_sp}: "{peak_text}"\n'
                     f'[0:03] {other} freezes. One beat of silence.\n'
                     f'[0:04] SUPER: "{pb["super"]}"'),
            "rationale": "Moves the episode's highest-tension line to second 0 so conflict lands before the first swipe decision.",
            "hypothesis": "Opening on the peak conflict line lowers 0-3s swipe-away versus the original opening.",
            "primary_metric": "3-second view rate",
        },
        {
            "angle": "Open-loop question",
            "text": (f"[0:00] Extreme close-up on {a}'s eyes. No music.\n"
                     f'{a}: "{pb["question"]}"\n'
                     f"[0:03] {b} turns slowly; we never see their face.\n"
                     f"[0:04] SFX: sharp inhale, hard cut to the scene."),
            "rationale": "A direct, unanswered question creates a curiosity gap the viewer can only close by watching on.",
            "hypothesis": "A charged question in the first line raises 5-second hold rate versus the original opening.",
            "primary_metric": "5-second hold rate",
        },
        {
            "angle": "Flash-forward tease",
            "text": (f"[0:00] FLASH-FORWARD: desaturated, tilted frame.\n"
                     f'{last_sp}: "{last_text}"\n'
                     f'[0:03] SUPER: "{pb["flash"]}"\n'
                     f"[0:04] RECORD SCRATCH, hard cut to the real opening."),
            "rationale": "Shows the episode's payoff first, so the viewer stays to learn how the story gets there.",
            "hypothesis": "Teasing the cliffhanger moment raises completion rate and next-episode taps.",
            "primary_metric": "completion rate",
        },
    ]
    for label, v in zip("ABC", variants):
        v["label"] = label
        v["genre"] = g
    return variants
