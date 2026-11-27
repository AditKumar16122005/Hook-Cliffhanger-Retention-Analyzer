"""Deterministic, explainable retention heuristics for a parsed script.

These are *uncalibrated* rules of thumb drawn from short-form drama craft. They are
used (a) as a fast offline engine, (b) as grounding/sanity input for the LLM, and
(c) blended with the LLM's judgement. Calibrate the weights against your own
retention analytics before relying on absolute numbers.
"""
from __future__ import annotations

import math
import re

from .script_parser import Line, ParsedScript

WINDOW_S = 5.0
HOOK_WINDOW_S = 5.0
CONFLICT_RAW_THRESHOLD = 1.0
FLAT_INTENSITY = 0.2
FLAT_MIN_SECONDS = 10.0


def _rx(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


CONFLICT = _rx([
    r"\blie[ds]?\b", r"\bliar\b", r"\blying\b", r"\bbetray\w*", r"\bcheat\w*", r"\baffair\b",
    r"\bdivorc\w*", r"\bpregnan\w*", r"\bsecrets?\b", r"\bkill\w*", r"\bmurder\w*", r"\bdead\b",
    r"\bdie[sd]?\b", r"\bdeath\b", r"\bblood\w*", r"\bgun\b", r"\bknife\b", r"\bthreat\w*",
    r"\bget out\b", r"\bfired\b", r"\barrest\w*", r"\bpolice\b", r"\bhate\w*", r"\bhow dare\b",
    r"\bslap\w*", r"\bscream\w*", r"\btraitor\b", r"\bcurse[ds]?\b", r"\bdemon\w*",
    r"\bprophec\w*", r"\bsacrific\w*", r"\brevenge\b", r"\bblackmail\w*", r"\bshove[sd]?\b",
    r"\bslam\w*", r"\bsmash\w*", r"\bweapon\w*", r"\bpoison\w*", r"\bkidnap\w*",
    r"\bleave me\b", r"\bhow could you\b", r"\bi know what you\b", r"\bdon'?t touch\b",
    r"\bstay away\b", r"\bdestiny\b", r"\bfate\b", r"\bgods?\b", r"\bpay for\b", r"\bregret\b",
    r"\bnever forgive\b", r"\bruined?\b", r"\bhumiliat\w*", r"\bcaught\b",
])
EMOTION = _rx([
    r"\bi never (?:wanted|meant|told)\b", r"\bi can'?t\b", r"\bfind out\b", r"\btake everything\b",
    r"\b(?:it|this) isn'?t what you think\b", r"\bit'?s not what you think\b", r"\bdon'?t (?:go|leave|say)\b",
    r"\bplease\b", r"\bi'?m sorry\b", r"\bforgive\b", r"\bcan'?t (?:believe|do this)\b",
    r"\bwhat have\b", r"\bno one (?:knows|can)\b", r"\bit'?s too late\b",
])
ACCUSE = re.compile(r"\byou(?:'re| are| did| never| always| lied| knew| killed| told| promised| said)\b")
CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")
OPEN_LOOP = _rx([
    r"\bwhat did you\b", r"\bwho (?:is|are|was|were)\b", r"\bwhy did\b", r"\bdon'?t tell\b",
    r"\byou shouldn'?t\b", r"\bbefore it'?s too late\b", r"\bthe truth\b", r"\bwhat if\b",
    r"\bwhere is\b", r"\bwho sent\b", r"\bwhat happened\b", r"\bdo you (?:remember|know)\b",
    r"\bit was (?:you|him|her)\b", r"\bwhat'?s in\b", r"\bwhat have you done\b",
])
REVEAL = _rx([
    r"\bit was (?:you|him|her|me)\b", r"\bshe'?s alive\b", r"\bhe'?s alive\b", r"\bthe truth\b",
    r"\bi know who\b", r"\bpregnan\w*", r"\bmy (?:son|daughter|child|baby)\b",
    r"\bnever (?:died|left)\b", r"\bi (?:am|'m) the\b", r"\byou'?re not my\b", r"\bhe'?s my\b",
    r"\bshe'?s my\b", r"\bwas never\b", r"\bi'?m not (?:who|what)\b", r"\bit'?s (?:him|her|you)\b",
])
CUT = _rx([
    r"\bdoor (?:opens|slams|creaks|bursts)\b", r"\bphone (?:rings|buzzes|lights)\b",
    r"\bscreen (?:goes|cuts) (?:to )?black\b", r"\bfootsteps\b", r"\bfigure\b", r"\bsilhouette\b",
    r"\bcut to black\b", r"\bsmash cut\b", r"\bfreezes?\b", r"\bgasps?\b", r"\bnotification\b",
    r"\bmessage\b", r"\bappears?\b", r"\bsteps? (?:out|in|forward)\b",
])
RESOLVED = _rx([
    r"\bfinally\b", r"\bat last\b", r"\bhappily\b", r"\bit'?s over\b", r"\bpeace\b",
    r"\bgood ?night\b", r"\bthe end\b", r"\bi forgive you\b", r"\bi love you too\b",
    r"\ball is well\b", r"\bsafe now\b", r"\beverything'?s fine\b",
])
GREETING = re.compile(r"^(?:hello|hi|hey|good (?:morning|evening|afternoon)|so,|once upon|my name is|let me tell|welcome|today)\b")
STAKES = re.compile(
    r"\b(?:love|husband|wife|baby|child|son|daughter|mother|mom|father|dad|money|life|marry|"
    r"wedding|fianc\w+|boss|kingdom|throne|soul|heart|family)\b")


def _count(patterns: list[re.Pattern], text: str) -> int:
    low = text.lower()
    return sum(1 for p in patterns if p.search(low))


def line_intensity(text: str, kind: str = "dialogue") -> float:
    """Raw dramatic-intensity score for one line (0 .. ~4)."""
    if kind == "heading":
        return 0.0
    raw = float(min(_count(CONFLICT, text), 3))
    raw += 0.5 * min(_count(EMOTION, text), 2)
    if "?" in text:
        raw += 0.5
    if "!" in text:
        raw += 0.5
    if CAPS_WORD.search(text):
        raw += 0.7
    if ACCUSE.search(text.lower()):
        raw += 0.8
    if text.rstrip().endswith(("—", "-", "...", "…")):
        raw += 0.4
    return raw


def _norm(raw: float) -> float:
    return min(1.0, raw / 3.0)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _interp(x: float, pts: list[tuple[float, float]]) -> float:
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


TTFC_CURVE = [(0, 100), (3, 100), (5, 85), (10, 60), (20, 30), (45, 5)]


def ttfc_score(seconds: float | None) -> int:
    if seconds is None:
        return 5
    return round(_interp(seconds, TTFC_CURVE))


def _excerpt(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# --------------------------------------------------------------------------- hook
def score_hook(lines: list[Line]) -> dict:
    hook = [l for l in lines if l.start < HOOK_WINDOW_S and l.kind != "heading"]
    first = next((l for l in lines if l.kind != "heading"), None)
    if not hook or first is None:
        return {"score": 0, "components": {}, "excerpt": ""}
    hook_text = " ".join(l.text for l in hook)
    max_raw = max(line_intensity(l.text, l.kind) for l in hook)

    conflict = min(1.0, max_raw / 2.0)
    if _count(OPEN_LOOP, hook_text):
        curiosity = 1.0
    elif "?" in hook_text:
        curiosity = 0.6
    else:
        curiosity = 0.0
    immediacy = 0.0
    if first.start <= 1.0:
        immediacy += 0.4
    if first.words <= 14:
        immediacy += 0.3
    if not GREETING.match(first.text.lower().strip()):
        immediacy += 0.3
    stakes = min(1.0, len(STAKES.findall(hook_text.lower())) / 2.0)
    interrupt = _norm(line_intensity(first.text, first.kind))

    comp = {
        "conflict_in_first_5s": round(conflict * 35, 1),
        "curiosity_gap": round(curiosity * 15, 1),
        "immediacy": round(immediacy * 20, 1),
        "personal_stakes": round(stakes * 15, 1),
        "pattern_interrupt": round(interrupt * 15, 1),
    }
    return {"score": round(_clamp(sum(comp.values()))), "components": comp,
            "excerpt": _excerpt(hook_text, 140)}


# --------------------------------------------------------------------------- ttfc
def find_ttfc(lines: list[Line], total: float) -> dict:
    for l in lines:
        if l.kind == "heading":
            continue
        if line_intensity(l.text, l.kind) >= CONFLICT_RAW_THRESHOLD:
            return {"seconds": round(l.start, 1), "score": ttfc_score(l.start),
                    "line_idx": l.idx, "excerpt": _excerpt(l.text), "found": True}
    return {"seconds": None, "score": ttfc_score(None), "line_idx": None,
            "excerpt": "", "found": False}


# --------------------------------------------------------------------------- pacing
def window_intensities(lines: list[Line], total: float) -> list[dict]:
    n = max(1, math.ceil(total / WINDOW_S))
    out = []
    for w in range(n):
        ws, we = w * WINDOW_S, (w + 1) * WINDOW_S
        best = 0.0
        for l in lines:
            if l.kind == "heading":
                continue
            end = max(l.end, l.start + 0.01)
            if l.start < we and end > ws:
                v = _norm(line_intensity(l.text, l.kind))
                if not (ws <= l.start < we):
                    v *= 0.6          # a line carried over from the previous window
                best = max(best, v)
        out.append({"t": ws, "intensity": round(best, 3)})
    return out


def score_pacing(windows: list[dict], total: float) -> dict:
    vals = [w["intensity"] for w in windows]
    n = len(vals)
    density = sum(1 for v in vals if v >= 1 / 3) / n
    a = min(1.0, density / 0.6)

    half = max(1, n // 2)
    first_mean = sum(vals[:half]) / half
    second = vals[half:] or vals
    second_mean = sum(second) / len(second)
    b = _clamp((0.6 + (second_mean - first_mean) * 1.5) * 100, 0, 100) / 100

    turns = 0
    for i in range(1, n - 1):
        d1, d2 = vals[i] - vals[i - 1], vals[i + 1] - vals[i]
        if d1 * d2 < 0 and min(abs(d1), abs(d2)) >= 0.15:
            turns += 1
    c = min(1.0, turns / max(1.0, total / 15.0))

    flats, run_start = [], None
    for i, v in enumerate(vals + [1.0]):          # sentinel closes a trailing run
        if v < FLAT_INTENSITY and run_start is None:
            run_start = i
        elif v >= FLAT_INTENSITY and run_start is not None:
            s, e = run_start * WINDOW_S, min(total, i * WINDOW_S)
            if e - s >= FLAT_MIN_SECONDS:
                flats.append({"start": round(s, 1), "end": round(e, 1)})
            run_start = None
    flat_seconds = sum(f["end"] - f["start"] for f in flats)
    d = max(0.0, 1.0 - flat_seconds / max(1.0, 0.3 * total))

    comp = {"beat_density": round(a * 30, 1), "escalation": round(b * 25, 1),
            "reversals": round(c * 20, 1), "no_dead_air": round(d * 25, 1)}
    return {"score": round(_clamp(sum(comp.values()))), "components": comp,
            "flat_stretches": flats, "reversals": turns}


# --------------------------------------------------------------------------- cliffhanger
def score_cliffhanger(lines: list[Line], total: float) -> dict:
    body = [l for l in lines if l.kind != "heading"]
    if not body:
        return {"score": 0, "type": "none", "components": {}, "excerpt": ""}
    tail = [l for l in body if l.end > total - 8.0] or body[-2:]
    tail = tail[-3:]
    text = " ".join(l.text for l in tail)
    last = tail[-1]

    question = 1.0 if (last.text.rstrip().endswith("?") or _count(OPEN_LOOP, last.text)) else 0.0
    reveal = 1.0 if _count(REVEAL, text) else 0.0
    cut = 1.0 if (last.text.rstrip().endswith(("—", "-", "...", "…")) or _count(CUT, text)) else 0.0
    final_int = _norm(line_intensity(last.text, last.kind))
    resolved = 1.0 if _count(RESOLVED, text) else 0.0

    comp = {"open_question": 25 * question, "reveal": 25 * reveal, "cut_off_or_interruption": 20 * cut,
            "final_line_intensity": round(20 * final_int, 1), "baseline": 10.0,
            "resolution_penalty": -30 * resolved}
    score = round(_clamp(sum(comp.values())))

    if resolved and score < 45:
        kind = "resolved (weak)"
    elif reveal:
        kind = "reveal"
    elif question:
        kind = "question"
    elif cut:
        kind = "interruption"
    elif final_int >= 0.5:
        kind = "emotional peak"
    else:
        kind = "flat ending"
    return {"score": score, "type": kind, "components": comp, "excerpt": _excerpt(text, 140)}


# --------------------------------------------------------------------------- orchestrator
def dropoff_from_scores(hook: float, ttfc: float, pacing: float, cliff: float) -> int:
    return round(100 - (0.35 * hook + 0.20 * ttfc + 0.25 * pacing + 0.20 * cliff))


def analyze_script(parsed: ParsedScript) -> dict:
    lines = parsed.lines
    total = parsed.total_seconds
    if not parsed.content_lines() or total <= 0:
        raise ValueError("Script contains no parsable content.")

    hook = score_hook(lines)
    ttfc = find_ttfc(lines, total)
    windows = window_intensities(lines, total)
    pacing = score_pacing(windows, total)
    cliff = score_cliffhanger(lines, total)

    risk_curve = []
    for i, w in enumerate(windows):
        risk = 0.15 + 0.6 * (1 - w["intensity"])
        if i == 0:
            risk += 0.25 * (1 - hook["score"] / 100)
        risk_curve.append({"t": w["t"], "risk": round(max(0.0, min(1.0, risk)), 3)})

    zones, flags = [], []
    if hook["score"] < 50:
        zones.append({"start_s": 0.0, "end_s": min(total, HOOK_WINDOW_S), "severity": "high",
                      "reason": "Weak opening: little conflict, curiosity or stakes in the first 5 seconds.",
                      "fix": "Open on the conflict peak or a charged question; cut set-up and greetings."})
    if ttfc["seconds"] is None:
        flags.append("No conflict beat detected anywhere in the script.")
        zones.append({"start_s": 0.0, "end_s": round(total, 1), "severity": "high",
                      "reason": "No conflict detected in the whole episode.",
                      "fix": "Introduce a confrontation, accusation or reveal early."})
    elif ttfc["seconds"] > 10:
        zones.append({"start_s": 0.0, "end_s": ttfc["seconds"], "severity": "high" if ttfc["seconds"] > 20 else "medium",
                      "reason": f"First conflict only arrives at {ttfc['seconds']}s.",
                      "fix": "Move the first confrontation earlier or tease it in the opening seconds."})
    for f in pacing["flat_stretches"]:
        zones.append({"start_s": f["start"], "end_s": f["end"], "severity": "medium",
                      "reason": "Flat stretch: no escalation or emotional beat.",
                      "fix": "Add a reversal, new information, or raise the stakes here."})
    if cliff["score"] < 50:
        flags.append("Ending is unlikely to drive a next-episode tap.")
    if total > 150:
        flags.append(f"Estimated runtime {total:.0f}s is long for a vertical micro-drama episode.")

    return {
        "duration_s": round(total, 1),
        "timeline_estimated": not parsed.has_timestamps,
        "hook": hook, "ttfc": ttfc, "pacing": pacing, "cliffhanger": cliff,
        "windows": windows, "risk_curve": risk_curve, "risk_zones": zones, "flags": flags,
        "dropoff_risk": dropoff_from_scores(hook["score"], ttfc["score"], pacing["score"], cliff["score"]),
    }
