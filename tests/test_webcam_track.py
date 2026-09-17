"""Unit coverage for the coast layer in scripts/webcam_live_detect.py.

Context: 5% of moving frames in the 2026-08-23 measurement produced no box at
all, even at conf 0.05. No threshold recovers those. What recovers them is a
lost track staying Kalman-predicted for a few frames, which is what
coasted_tracks() surfaces. A coasted box is drawn thin and labelled COAST -- it
must never be mistaken for a frame the model actually saw something on.

These tests use a stub tracker so they stay fast and need no model weights; the
end-to-end behaviour (ID held across a 3-frame dropout, same ID on recovery) was
verified separately against a real ByteTrack run.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wld():
    spec = importlib.util.spec_from_file_location("wld", ROOT / "scripts" / "webcam_live_detect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(tid, end_frame, box=(10.0, 20.0, 60.0, 120.0)):
    return SimpleNamespace(track_id=tid, end_frame=end_frame, xyxy=list(box))


def _model(frame_id, lost):
    tracker = SimpleNamespace(frame_id=frame_id, lost_stracks=lost)
    return SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]))


def test_a_recently_lost_track_is_coasted(wld):
    got = wld.coasted_tracks(_model(frame_id=12, lost=[_track(7, end_frame=9)]), max_age=30)
    assert got == [(7, [10.0, 20.0, 60.0, 120.0], 3)]


def test_a_track_lost_longer_than_the_budget_is_dropped(wld):
    """Otherwise a survivor who has genuinely left keeps haunting the frame."""
    assert wld.coasted_tracks(_model(frame_id=50, lost=[_track(7, end_frame=9)]), max_age=30) == []


def test_a_track_updated_this_frame_is_not_coasted(wld):
    """Age 0 means the detector saw it -- it belongs in the confirmed list,
    and drawing it twice would show a solid and a COAST box on one subject."""
    assert wld.coasted_tracks(_model(frame_id=9, lost=[_track(7, end_frame=9)]), max_age=30) == []


def test_multiple_lost_tracks_each_carry_their_own_age(wld):
    got = wld.coasted_tracks(_model(frame_id=20, lost=[_track(1, 18), _track(2, 12)]), max_age=30)
    assert sorted((tid, age) for tid, _, age in got) == [(1, 2), (2, 8)]


def test_no_tracker_yet_is_not_an_error(wld):
    """First frame, and every frame when --no-track is set."""
    assert wld.coasted_tracks(SimpleNamespace(predictor=None), max_age=30) == []
    assert wld.coasted_tracks(SimpleNamespace(predictor=SimpleNamespace(trackers=[])), max_age=30) == []
    assert wld.coasted_tracks(SimpleNamespace(), max_age=30) == []


def test_cli_threshold_overrides_reach_the_tracker_config(wld):
    """A changed --conf-thres must move track_high_thresh with it, or the
    displayed bar and the tracker's first-stage bar silently diverge."""
    import yaml

    path = wld.tracker_cfg_with({"track_high_thresh": 0.42, "track_low_thresh": 0.11})
    try:
        assert Path(path) != wld.DEFAULT_TRACKER_CFG, "an override must not be written into the repo config"
        cfg = yaml.safe_load(Path(path).read_text())
        assert cfg["track_high_thresh"] == 0.42
        assert cfg["track_low_thresh"] == 0.11
        assert cfg["tracker_type"] == "bytetrack"
        assert cfg["new_track_thresh"] == 0.30, "creating a track must stay stricter than continuing one"
    finally:
        wld.cleanup_tracker_cfg(path)


def test_unchanged_defaults_generate_no_file_at_all(wld):
    """The common case writes nothing, so nothing can leak."""
    import yaml

    repo = yaml.safe_load(wld.DEFAULT_TRACKER_CFG.read_text())
    path = wld.tracker_cfg_with(
        {"track_high_thresh": repo["track_high_thresh"], "track_low_thresh": repo["track_low_thresh"]}
    )
    assert Path(path) == wld.DEFAULT_TRACKER_CFG


def test_cleanup_refuses_to_delete_the_tracked_repo_config(wld):
    """tracker_cfg_with returns the repo path on the default run, so an
    unguarded unlink in the shutdown path would delete a versioned file."""
    wld.cleanup_tracker_cfg(str(wld.DEFAULT_TRACKER_CFG))
    assert wld.DEFAULT_TRACKER_CFG.exists()


def test_new_track_threshold_stays_above_the_second_pass_floor(wld):
    """Feeding low-confidence boxes to the tracker is only safe while starting
    a NEW track needs stronger evidence than continuing an existing one."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "bytetrack_sar.yaml").read_text())
    assert cfg["new_track_thresh"] > cfg["track_low_thresh"]
    assert cfg["track_high_thresh"] > cfg["track_low_thresh"]
