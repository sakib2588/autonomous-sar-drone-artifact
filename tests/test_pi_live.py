"""Regression tests for the on-Pi live pipeline's recording behaviour.

Both bugs covered here made a MOVING person look uncapturable in the recorded
footage while a stationary one looked fine, which is the symptom that started
the 2026-08-23 investigation:

  1. Boxes were drawn only on the 1-in-N frames the model actually ran on, so
     the recording strobed them on and off. webcam_smoke_test.py's docstring
     already specified the correct behaviour; 15_pi_live.py contradicted it.
  2. The mp4 header was stamped with --fps (30) while the real loop runs far
     slower, so footage played back several times too fast.

15_pi_live.py is written to run ON the Pi, where src/sar/bench/{decode,sysmon}.py
sit beside it as flat modules. Stub those two imports so the module can be
loaded on a dev box; nothing under test touches either of them.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pi_live():
    decode = types.ModuleType("decode")
    decode.decode_yolo_output = lambda *a, **k: []
    sysmon = types.ModuleType("sysmon")
    sysmon.describe = lambda flags: ""
    sysmon.is_clean = lambda flags: True
    sysmon.parse_throttled = lambda text: 0
    decode.CLASS_NAMES = ("person", "vehicle", "two_wheeler")
    sys.modules.setdefault("decode", decode)
    sys.modules.setdefault("sysmon", sysmon)

    # The real tiling module, not a stub: merge_tile_detections owns the
    # coordinate shift the frame_coords test below is checking.
    tspec = importlib.util.spec_from_file_location("tiling", ROOT / "src" / "sar" / "data" / "tiling.py")
    tiling = importlib.util.module_from_spec(tspec)
    tspec.loader.exec_module(tiling)
    sys.modules.setdefault("tiling", tiling)

    spec = importlib.util.spec_from_file_location("pi_live", ROOT / "scripts" / "15_pi_live.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _det(conf=0.8):
    return {"box_xyxy": [10.0, 10.0, 40.0, 60.0], "class_name": "person", "conf": conf}


# --- bug 1: the strobe ---------------------------------------------------

def test_boxes_are_held_across_the_frames_between_inferences(pi_live):
    """With infer_every_n=5, a detection on frame 0 must still be drawn on 1-4."""
    dets = [_det()]
    drawn = [pi_live.held_boxes(i, 0, dets, hold_frames=5) for i in range(1, 5)]
    assert all(d == dets for d in drawn), "recording strobes: boxes vanish between inferences"


def test_held_boxes_expire_instead_of_freezing_forever(pi_live):
    dets = [_det()]
    assert pi_live.held_boxes(5, 0, dets, hold_frames=5) == []


def test_an_inference_that_finds_nothing_clears_the_held_box(pi_live):
    """A real loss must not leave a stale survivor painted on the footage."""
    assert pi_live.held_boxes(1, 0, [], hold_frames=5) == []


def test_held_boxes_are_visually_distinct_from_real_ones(pi_live):
    """Footage must never imply an inference that did not happen."""
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    fresh = pi_live.draw_detections(base.copy(), [_det()], 160, fresh=True)
    held = pi_live.draw_detections(base.copy(), [_det()], 160, fresh=False)
    assert not np.array_equal(fresh, held)
    assert fresh.sum() > held.sum(), "held box should be drawn thinner than a real one"


# --- bug 2: the frame-rate lie -------------------------------------------

def test_segment_sidecar_records_true_fps_not_the_header(pi_live, tmp_path):
    class FakeWriter:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    seg = tmp_path / "flight_20260823T120000.mp4"
    writer = FakeWriter()
    # 90 frames over 10 s = 9 FPS, written under a 30 FPS header: the exact
    # 3.3x playback speed-up that made walking look like teleporting.
    pi_live.close_segment(writer, seg, frames=90, started=1000.0, ended=1010.0, nominal_fps=30.0)

    assert writer.released
    timing = json.loads((tmp_path / "flight_20260823T120000.timing.json").read_text())
    assert timing["header_fps"] == 30.0
    assert timing["true_fps"] == 9.0
    assert timing["frames_written"] == 90


def test_sidecar_survives_a_zero_length_segment(pi_live, tmp_path):
    """SIGTERM can land immediately after a rotation; that must not divide by zero."""
    class FakeWriter:
        def release(self):
            pass

    seg = tmp_path / "flight_empty.mp4"
    pi_live.close_segment(FakeWriter(), seg, frames=0, started=5.0, ended=5.0, nominal_fps=30.0)
    timing = json.loads((tmp_path / "flight_empty.timing.json").read_text())
    assert timing["frames_written"] == 0


# --- tiled inference coordinate space -------------------------------------

def test_tiled_boxes_are_drawn_without_a_second_rescale(pi_live):
    """merge_tile_detections already returns FULL-FRAME pixels. Applying the
    model-space scaling to them again shrinks every box by w/model_size --
    boxes would render in the top-left corner at half size."""
    base = np.zeros((720, 1280, 3), dtype=np.uint8)
    det = [{"box_xyxy": [600.0, 300.0, 700.0, 500.0], "class_name": "person", "conf": 0.9}]

    tiled = pi_live.draw_detections(base.copy(), det, 640, frame_coords=True)
    untiled = pi_live.draw_detections(base.copy(), det, 640, frame_coords=False)
    assert not np.array_equal(tiled, untiled), "frame_coords had no effect"

    ys, xs = np.nonzero(tiled[:, :, 1])
    assert 595 <= xs.min() <= 605 and 695 <= xs.max() <= 705, "tiled box moved; it must be drawn as-is"


def test_tile_grid_covers_the_frame_with_the_trained_overlap(pi_live):
    """A train/deploy tiling mismatch is a silent accuracy leak, so the grid
    must match results/wisard_tile_manifest.json (tile 640, overlap 0.2)."""
    import tiling

    tiles = tiling.generate_tiles(1280, 720, tile_size=640, overlap=0.2)
    assert len(tiles) == 6, f"expected 6 tiles for 1280x720, got {len(tiles)}"
    assert all((x1 - x0, y1 - y0) == (640, 640) for x0, y0, x1, y1 in tiles)
    assert max(x1 for _, _, x1, _ in tiles) == 1280, "right edge not covered"
    assert max(y1 for _, _, _, y1 in tiles) == 720, "bottom edge not covered"
