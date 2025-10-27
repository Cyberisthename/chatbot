import json
from pathlib import Path

from quantacap.experiments.quion_vizrun import run_quion_viz


def test_quion_reverse_smoke(tmp_path):
    prefix = tmp_path / "reverse"
    summary = run_quion_viz(
        scenario="reverse",
        steps=20,
        stride=5,
        out_prefix=str(prefix),
        video_format="gif",
    )
    series = Path(summary["series"])
    assert series.is_file()
    with open(series, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["scenario"] == "reverse"
    assert payload["metrics"]["F_final"] >= 0.999999 - 1e-6
    frames = sorted((tmp_path / "reverse_frames").glob("frame_*.png"))
    assert len(frames) >= 5


def test_quion_freeze_smoke(tmp_path):
    prefix = tmp_path / "freeze"
    summary = run_quion_viz(
        scenario="freeze",
        steps=30,
        stride=5,
        tau=1e-3,
        out_prefix=str(prefix),
        video_format="gif",
    )
    series = Path(summary["series"])
    assert series.is_file()
    with open(series, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload["metrics"]
    assert metrics["frames_committed"] < metrics["total_steps"]
    assert metrics["frames_rendered"] >= metrics["frames_committed"]
    frames = sorted((tmp_path / "freeze_frames").glob("frame_*.png"))
    assert len(frames) >= metrics["frames_committed"]


def test_quion_animation_fallback(tmp_path):
    prefix = tmp_path / "anim"
    summary = run_quion_viz(
        scenario="noise",
        steps=20,
        stride=5,
        jitter=5e-4,
        out_prefix=str(prefix),
        video_format="gif",
    )
    video_path = Path(summary["video"])
    assert video_path.suffix == ".gif"
    assert video_path.is_file()
    series = Path(summary["series"])
    with open(series, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload["metrics"]
    assert metrics["coherence_guard_final"] >= metrics["coherence_open_final"] - 1e-6
