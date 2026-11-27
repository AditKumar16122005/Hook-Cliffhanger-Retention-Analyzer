import io
import subprocess

import pytest

from app import llm as llm_mod


def _analyze(client, **kw):
    data = {"title": "Ep 12", "genre": "romance", **kw.pop("data", {})}
    return client.post("/api/analyze", data=data, **kw)


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["llm_enabled"] is False


def test_analyze_text_offline_end_to_end(client, weak_script):
    r = _analyze(client, data={"script_text": weak_script})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "heuristic" and body["version_number"] == 1 and body["cached"] is False
    s = body["scores"]
    assert set(s) >= {"hook_strength", "time_to_first_conflict_s", "pacing", "cliffhanger", "dropoff_risk"}
    assert [v["label"] for v in body["variants"]] == ["A", "B", "C"]
    assert any("offline" in w for w in body["analysis"]["warnings"])
    assert body["analysis"]["script_lines"]


def test_script_file_upload_and_versioning_cache_and_compare(client, weak_script, strong_script):
    ep = _analyze(client, data={"script_text": weak_script}).json()["episode_id"]
    # same content -> cached, no new version
    again = _analyze(client, data={"script_text": weak_script, "episode_id": str(ep)}).json()
    assert again["cached"] is True and again["version_number"] == 1
    # force re-analysis -> new version
    forced = _analyze(client, data={"script_text": weak_script, "episode_id": str(ep), "force": "true"}).json()
    assert forced["version_number"] == 2
    # revised script as a file upload -> v3
    files = {"script": ("v2.txt", io.BytesIO(strong_script.encode()), "text/plain")}
    v3 = _analyze(client, data={"episode_id": str(ep)}, files=files).json()
    assert v3["version_number"] == 3
    vs = client.get(f"/api/episodes/{ep}/versions").json()
    assert [v["version_number"] for v in vs] == [1, 2, 3]
    cmp = client.get(f"/api/episodes/{ep}/compare", params={"base": vs[0]["id"], "head": vs[2]["id"]}).json()
    assert cmp["delta"]["hook_strength"] > 30 and cmp["delta"]["dropoff_risk"] < 0
    assert client.get(f"/api/versions/{v3['id']}").json()["analysis"]["scores"] == v3["scores"]
    assert any(e["id"] == ep for e in client.get("/api/episodes").json())


def test_validation_errors(client):
    assert _analyze(client).status_code == 422                                    # nothing provided
    assert _analyze(client, data={"script_text": "# only a heading"}).status_code == 422
    assert _analyze(client, data={"script_text": "A: hi", "episode_id": "9999"}).status_code == 404
    assert _analyze(client, data={"script_text": "A: hi", "episode_id": "abc"}).status_code == 422
    bad = {"script": ("x.exe", io.BytesIO(b"hi"), "application/octet-stream")}
    assert _analyze(client, files=bad).status_code == 415
    badv = {"video": ("x.avi", io.BytesIO(b"hi"), "video/x-msvideo")}
    assert _analyze(client, data={"script_text": "A: hi there"}, files=badv).status_code == 415
    assert client.get("/api/versions/424242").status_code == 404


def test_pdf_script_upload(client, weak_script):
    pytest.importorskip("pypdf")
    # build a minimal PDF containing script text using ffmpeg-free pure python
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    w = PdfWriter()
    page = w.add_blank_page(width=612, height=792)
    lines = ["MAYA: You lied to me, Daniel. Who is Clara?", "DANIEL: Where did you hear that name?"]
    stream = "BT /F1 12 Tf 50 700 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    s = DecodedStreamObject(); s.set_data(stream.encode())
    page[NameObject("/Contents")] = w._add_object(s)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    buf = io.BytesIO(); w.write(buf)
    r = _analyze(client, files={"script": ("s.pdf", io.BytesIO(buf.getvalue()), "application/pdf")})
    assert r.status_code == 200, r.text
    assert "MAYA" in [l["speaker"] for l in r.json()["analysis"]["script_lines"]]


def test_experiment_flow(client, weak_script):
    ver = _analyze(client, data={"script_text": weak_script, "force": "true"}).json()
    exp = client.post(f"/api/versions/{ver['id']}/experiments", json={"name": "Hook test"})
    assert exp.status_code == 201
    exp = exp.json()
    assert len(exp["arms"]) == 4 and exp["arms"][0]["is_control"]
    ctl, a, b, c = [x["id"] for x in exp["arms"]]
    # insufficient data first
    r = client.get(f"/api/experiments/{exp['id']}/results").json()
    assert r["status"] == "insufficient_data"
    ev = {"events": [{"arm_id": ctl, "impressions": 5000, "conversions": 1000},
                     {"arm_id": a, "impressions": 5000, "conversions": 1300},
                     {"arm_id": b, "impressions": 5000, "conversions": 1010},
                     {"arm_id": c, "impressions": 5000, "conversions": 990}]}
    assert client.post(f"/api/experiments/{exp['id']}/events", json=ev).status_code == 200
    res = client.get(f"/api/experiments/{exp['id']}/results").json()
    assert res["status"] == "winner" and res["winner_arm_id"] == a
    assert res["arms"][1]["lift_vs_control"] == pytest.approx(0.3)
    # invalid: conversions > impressions (and nothing is partially applied)
    bad = {"events": [{"arm_id": b, "impressions": 10, "conversions": 5},
                      {"arm_id": c, "impressions": 1, "conversions": 10000}]}
    assert client.post(f"/api/experiments/{exp['id']}/events", json=bad).status_code == 422
    after = {x["id"]: x["impressions"] for x in client.get(f"/api/experiments/{exp['id']}").json()["arms"]}
    assert after[b] == 5000
    # arm from another experiment
    other = client.post(f"/api/versions/{ver['id']}/experiments").json()
    foreign = other["arms"][0]["id"]
    assert client.post(f"/api/experiments/{exp['id']}/events",
                       json={"events": [{"arm_id": foreign, "impressions": 1, "conversions": 0}]}).status_code == 404
    assert client.post(f"/api/experiments/{exp['id']}/events",
                       json={"events": [{"arm_id": a, "impressions": -1, "conversions": 0}]}).status_code == 422


def test_sample_size_endpoint(client):
    r = client.get("/api/stats/sample-size", params={"baseline": 0.3, "relative_mde": 0.1}).json()
    assert 3700 <= r["per_arm"] <= 3800
    assert client.get("/api/stats/sample-size", params={"baseline": 0.9, "relative_mde": 0.5}).status_code == 422


# ---------------------------------------------------------------- video + LLM
@pytest.fixture()
def video_file(tmp_path):
    p = tmp_path / "ep.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=20:size=360x640:rate=15",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_video_only_without_llm_is_503(client, video_file):
    with open(video_file, "rb") as f:
        r = _analyze(client, files={"video": ("ep.mp4", f, "video/mp4")})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_script_plus_video_without_llm_rescales_timeline(client, video_file, strong_script):
    with open(video_file, "rb") as f:
        r = _analyze(client, data={"script_text": strong_script},
                     files={"video": ("ep.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    a = r.json()["analysis"]
    assert a["duration_s"] == pytest.approx(20.0, abs=0.5)
    assert any("not analyzed" in w for w in a["warnings"])


def test_corrupt_video_is_422_and_cleaned_up(client):
    r = _analyze(client, data={"script_text": "A: hi there"},
                 files={"video": ("bad.mp4", io.BytesIO(b"not a video"), "video/mp4")})
    assert r.status_code == 422


def _fake_llm_result():
    return llm_mod.LLMAnalysis(
        hook_score=90, hook_rationale="Opens on accusation.", conflict_time_s=1.5,
        conflict_description="Maya accuses Daniel.", pacing_score=70,
        beats=[{"t_s": 0, "label": "accusation", "intensity": 0.9}, {"t_s": 10, "label": "denial", "intensity": 0.5}],
        cliffhanger_score=80, cliffhanger_type="reveal", cliffhanger_rationale="Baby reveal.",
        dropoff_zones=[{"start_s": 5, "end_s": 12, "severity": "medium", "reason": "Repetition", "fix": "Cut line"}],
        hook_rewrites=[{"angle": f"Lever {i}", "text": f"[0:00] opening {i}", "rationale": "r",
                        "hypothesis": "h", "primary_metric": "3-second view rate"} for i in range(3)],
        top_fixes=["Cut the greeting"])


def test_hybrid_engine_with_mocked_multimodal_llm(client, video_file, strong_script, monkeypatch):
    captured = {}

    def fake_analyze(self, **kw):
        captured.update(kw)
        content = self.build_content(**kw)
        captured["n_images"] = sum(1 for b in content if b["type"] == "image")
        captured["has_script_tag"] = any("<script>" in b.get("text", "") for b in content)
        return _fake_llm_result()

    monkeypatch.setattr(llm_mod.LLMAnalyzer, "enabled", property(lambda s: True))
    monkeypatch.setattr(llm_mod.LLMAnalyzer, "analyze", fake_analyze)
    with open(video_file, "rb") as f:
        r = _analyze(client, data={"script_text": strong_script, "force": "true"},
                     files={"video": ("ep.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "hybrid"
    assert captured["n_images"] >= 7 and captured["has_script_tag"]
    heur_hook = body["analysis"]["heuristics"]["hook"]["score"]
    assert body["scores"]["hook_strength"] == round(0.6 * 90 + 0.4 * heur_hook)
    assert body["scores"]["time_to_first_conflict_s"] == 1.5          # video present -> trust LLM timing
    assert [v["angle"] for v in body["variants"]] == ["Lever 0", "Lever 1", "Lever 2"]
    assert body["analysis"]["top_fixes"] == ["Cut the greeting"]
    assert {z["source"] for z in body["analysis"]["risk_zones"]} >= {"llm"}


def test_llm_failure_falls_back_to_heuristics(client, weak_script, monkeypatch):
    def boom(self, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm_mod.LLMAnalyzer, "enabled", property(lambda s: True))
    monkeypatch.setattr(llm_mod.LLMAnalyzer, "analyze", boom)
    r = _analyze(client, data={"script_text": weak_script, "force": "true"})
    assert r.status_code == 200
    assert r.json()["engine"] == "heuristic"
    assert any("LLM analysis failed" in w for w in r.json()["analysis"]["warnings"])


def test_llm_short_rewrites_are_padded(client, weak_script, monkeypatch):
    res = _fake_llm_result()
    res.hook_rewrites = res.hook_rewrites[:1]
    monkeypatch.setattr(llm_mod.LLMAnalyzer, "enabled", property(lambda s: True))
    monkeypatch.setattr(llm_mod.LLMAnalyzer, "analyze", lambda self, **kw: res)
    r = _analyze(client, data={"script_text": weak_script, "force": "true"}).json()
    assert [v["label"] for v in r["variants"]] == ["A", "B", "C"] and r["variants"][0]["angle"] == "Lever 0"


def test_llm_score_out_of_range_is_clamped():
    m = _fake_llm_result().model_dump()
    m["hook_score"] = 140
    assert llm_mod.LLMAnalysis.model_validate(m).hook_score == 100
