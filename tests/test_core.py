import pytest

from app.heuristics import analyze_script, line_intensity, ttfc_score
from app.rewrites import offline_rewrites
from app.script_parser import parse_script, rescale
from app.stats import (prob_best, required_sample_size, summarize_experiment,
                       two_prop_ztest, wilson_interval)


# ---------------------------------------------------------------- parser
def test_parser_styles_and_timestamps():
    p = parse_script("""INT. HOUSE: DAY
SARAH (V.O.)
I never told anyone.
(whispering)
Not even you.

SUPER: Three years later
[00:10] TOM: NO!
""")
    kinds = [(l.kind, l.speaker) for l in p.lines]
    assert kinds == [("heading", None), ("dialogue", "SARAH"), ("action", None), ("dialogue", "TOM")]
    assert p.lines[1].text == "I never told anyone. Not even you."
    assert p.has_timestamps and p.lines[3].start == 10.0
    assert p.lines[1].start == 1.0           # heading takes 1s


def test_parser_ignores_comments_and_blank():
    p = parse_script("# title\n\n// note\n[[note]]\nA: hi there\n")
    assert len(p.lines) == 1 and p.lines[0].speaker == "A"


def test_rescale_only_when_estimated():
    p = parse_script("A: hello there friend\nB: what is going on here")
    total = p.total_seconds
    rescale(p, total * 2)
    assert p.total_seconds == pytest.approx(total * 2)
    q = parse_script("[00:05] A: hello")
    before = q.total_seconds
    rescale(q, 100)
    assert q.total_seconds == before


# ---------------------------------------------------------------- heuristics
def test_intensity_ordering():
    assert line_intensity("Good morning, how are you?") < line_intensity("You lied to me! Who is she?")
    assert line_intensity("INT. HOUSE", "heading") == 0


def test_ttfc_curve_monotonic():
    vals = [ttfc_score(s) for s in (0, 3, 5, 10, 20, 45, 100)]
    assert vals == sorted(vals, reverse=True) and vals[0] == 100 and ttfc_score(None) == 5


def test_strong_beats_weak(weak_script, strong_script):
    w = analyze_script(parse_script(weak_script))
    s = analyze_script(parse_script(strong_script))
    assert s["hook"]["score"] > w["hook"]["score"] + 30
    assert s["ttfc"]["seconds"] < w["ttfc"]["seconds"]
    assert s["dropoff_risk"] < w["dropoff_risk"]
    assert any("First conflict only arrives" in z["reason"] for z in w["risk_zones"])


def test_empty_script_raises():
    with pytest.raises(ValueError):
        analyze_script(parse_script("# only a heading\n"))


def test_single_line_script_does_not_crash():
    r = analyze_script(parse_script("MAYA: You lied to me!"))
    assert 0 <= r["hook"]["score"] <= 100 and r["windows"]


def test_resolved_ending_scores_lower():
    base = "A: You lied to me!\nB: I know. I'm sorry.\n"
    resolved = analyze_script(parse_script(base + "A: I forgive you. Finally, peace."))
    open_ = analyze_script(parse_script(base + "A: Who is she?"))
    assert open_["cliffhanger"]["score"] > resolved["cliffhanger"]["score"]


def test_offline_rewrites_three_distinct(weak_script):
    rw = offline_rewrites(parse_script(weak_script), "thriller")
    assert [r["label"] for r in rw] == ["A", "B", "C"]
    assert len({r["angle"] for r in rw}) == 3 and len({r["text"] for r in rw}) == 3
    assert all(r["genre"] == "thriller" for r in rw)
    assert "Don't turn around" in rw[1]["text"]


def test_unknown_genre_falls_back():
    rw = offline_rewrites(parse_script("A: hi\nB: hello"), "cyberpunk-noir")
    assert len(rw) == 3 and rw[0]["genre"] == "cyberpunk-noir"


# ---------------------------------------------------------------- stats
def test_sample_size_matches_reference():
    assert 3700 <= required_sample_size(0.30, 0.10) <= 3800
    with pytest.raises(ValueError):
        required_sample_size(0.9, 0.5)


def test_wilson_and_ztest():
    lo, hi = wilson_interval(30, 100)
    assert lo < 0.30 < hi and 0 <= lo and hi <= 1
    assert wilson_interval(0, 0) is None
    z, p = two_prop_ztest(100, 1000, 130, 1000)
    assert z > 0 and 0.03 < p < 0.04
    assert two_prop_ztest(0, 100, 0, 100) == (0.0, 1.0)


def test_prob_best_sums_to_one_and_ranks():
    pb = prob_best([(100, 1000), (130, 1000), (100, 1000)])
    assert sum(pb) == pytest.approx(1.0) and pb[1] == max(pb)


def _arms(rows):
    return [{"id": i, "label": f"a{i}", "is_control": i == 0, "variant_id": None,
             "impressions": n, "conversions": k} for i, (n, k) in enumerate(rows)]


def test_experiment_verdicts():
    assert summarize_experiment(_arms([(100, 10), (100, 20)]))["status"] == "insufficient_data"
    win = summarize_experiment(_arms([(5000, 1000), (5000, 1300), (5000, 1000)]))
    assert win["status"] == "winner" and win["winner_arm_id"] == 1
    assert win["arms"][1]["lift_vs_control"] == pytest.approx(0.3)
    flat = summarize_experiment(_arms([(5000, 1000), (5000, 1010)]))
    assert flat["status"] == "inconclusive"
