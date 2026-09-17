"""Tests for tiled-training dataset construction (R4).

Resizing a 1500-2000px aerial frame to 640 shrinks a 30px person to ~10px at
training time. Tiling is the fix and it is primary, not an inference-only
trick -- these tests exist to prove tile-local box coordinates are correct
before any training run trusts them, and that inference-side merging
recovers full-frame coordinates without double-counting across tile overlap.
"""

import pytest

from src.sar.data.tiling import (
    generate_tiles,
    clip_box_to_tile,
    tile_dataset,
    select_tiles,
    merge_tile_detections,
)


# --- generate_tiles -------------------------------------------------------


def test_tiles_cover_the_full_image_with_no_gap():
    tiles = generate_tiles(img_w=1920, img_h=1080, tile_size=640, overlap=0.2)
    max_x = max(t[2] for t in tiles)
    max_y = max(t[3] for t in tiles)
    min_x = min(t[0] for t in tiles)
    min_y = min(t[1] for t in tiles)
    assert (min_x, min_y) == (0, 0)
    assert (max_x, max_y) == (1920, 1080)


def test_every_tile_is_exactly_tile_size():
    tiles = generate_tiles(img_w=1920, img_h=1080, tile_size=640, overlap=0.2)
    for x0, y0, x1, y1 in tiles:
        assert (x1 - x0, y1 - y0) == (640, 640)


def test_last_tile_is_flush_to_the_far_edge_not_hanging_off():
    tiles = generate_tiles(img_w=1000, img_h=640, tile_size=640, overlap=0.2)
    xs = sorted({t[2] for t in tiles})
    assert xs[-1] == 1000  # flush, never > img_w


def test_image_smaller_than_tile_size_yields_a_single_tile():
    tiles = generate_tiles(img_w=400, img_h=300, tile_size=640, overlap=0.2)
    assert tiles == [(0, 0, 400, 300)]


def test_overlap_produces_more_tiles_than_no_overlap():
    no_overlap = generate_tiles(img_w=1920, img_h=1080, tile_size=640, overlap=0.0)
    with_overlap = generate_tiles(img_w=1920, img_h=1080, tile_size=640, overlap=0.2)
    assert len(with_overlap) >= len(no_overlap)


# --- clip_box_to_tile -------------------------------------------------------


def test_box_fully_inside_tile_is_returned_in_tile_local_coords_unchanged_size():
    tile = (640, 0, 1280, 640)
    box = (700, 50, 750, 150)  # fully inside
    clipped = clip_box_to_tile(box, tile, min_visible_frac=0.3)
    assert clipped == (60, 50, 110, 150)  # shifted by tile origin


def test_box_entirely_outside_tile_is_dropped():
    tile = (0, 0, 640, 640)
    box = (1000, 1000, 1050, 1050)
    assert clip_box_to_tile(box, tile, min_visible_frac=0.3) is None


def test_box_below_visibility_threshold_is_dropped():
    tile = (0, 0, 640, 640)
    # 100x100 box, only a 10x10 sliver (1%) inside the tile
    box = (630, 630, 730, 730)
    assert clip_box_to_tile(box, tile, min_visible_frac=0.3) is None


def test_box_above_visibility_threshold_is_clipped_not_dropped():
    tile = (0, 0, 640, 640)
    # 100x100 box, 60x100 (60%) inside the tile
    box = (580, 300, 680, 400)
    clipped = clip_box_to_tile(box, tile, min_visible_frac=0.3)
    assert clipped == (580, 300, 640, 400)


# --- tile_dataset: the plan's exact requirement -----------------------------


def test_synthetic_box_reproduced_in_exactly_the_tiles_that_contain_it():
    # 1280x640 image -> two side-by-side 640x640 tiles at overlap=0 for a clean check
    img_w, img_h = 1280, 640
    boxes = [(0, 600, 300, 680, 340)]  # (cls, x0,y0,x1,y1) -- straddles the tile boundary at x=640
    tiled = tile_dataset(img_w, img_h, boxes, tile_size=640, overlap=0.0, min_visible_frac=0.3)

    tiles_with_this_box = [(tile, labels) for tile, labels in tiled if labels]
    # 80x40 box split evenly by the boundary: 40x40 (50%) visible in each tile -- both pass min_visible_frac=0.3
    assert len(tiles_with_this_box) == 2

    for tile, labels in tiles_with_this_box:
        assert len(labels) == 1
        cls_id, lx0, ly0, lx1, ly1 = labels[0]
        assert cls_id == 0
        tx0, ty0, tx1, ty1 = tile
        # reconstruct full-frame coords from tile-local + tile origin, must match original box
        assert (lx0 + tx0, ly0 + ty0) in [(600, 300), (640, 300)]
        assert (lx1 + tx0, ly1 + ty0) in [(640, 340), (680, 340)]


def test_tile_dataset_produces_negative_tiles_for_object_free_regions():
    img_w, img_h = 1280, 640
    boxes = [(0, 0, 0, 40, 40)]  # only in the top-left corner
    tiled = tile_dataset(img_w, img_h, boxes, tile_size=640, overlap=0.0, min_visible_frac=0.3)
    negative_tiles = [t for t, labels in tiled if not labels]
    assert len(negative_tiles) >= 1


# --- select_tiles: negative retention ---------------------------------------


def test_select_tiles_keeps_all_positive_tiles():
    tiled = [((0, 0, 640, 640), [(0, 10, 10, 20, 20)])] * 5 + [((640, 0, 1280, 640), [])] * 20
    selected = select_tiles(tiled, negative_keep_frac=0.2, seed=42)
    positives_kept = [t for t, l in selected if l]
    assert len(positives_kept) == 5


def test_select_tiles_samples_roughly_the_requested_negative_fraction():
    tiled = [((0, 0, 640, 640), [(0, 10, 10, 20, 20)])] + [
        ((i, 0, i + 640, 640), []) for i in range(1000)
    ]
    selected = select_tiles(tiled, negative_keep_frac=0.2, seed=42)
    negatives_kept = [t for t, l in selected if not l]
    assert 150 <= len(negatives_kept) <= 250  # ~20% of 1000, sampling tolerance


def test_select_tiles_is_deterministic_under_a_fixed_seed():
    tiled = [((i, 0, i + 640, 640), []) for i in range(200)]
    a = select_tiles(tiled, negative_keep_frac=0.2, seed=42)
    b = select_tiles(tiled, negative_keep_frac=0.2, seed=42)
    assert a == b


# --- merge_tile_detections: tiled-inference side ----------------------------


def test_merge_recovers_full_frame_coordinates_from_tile_local_detections():
    dets_per_tile = [
        ((640, 0, 1280, 640), [(0, 0.9, 10, 10, 60, 60)]),  # tile-local box
    ]
    merged = merge_tile_detections(dets_per_tile, iou_thresh=0.5)
    assert len(merged) == 1
    cls_id, conf, x0, y0, x1, y1 = merged[0]
    assert (x0, y0, x1, y1) == (650, 10, 700, 60)  # shifted into full-frame coords


def test_merge_deduplicates_the_same_object_detected_in_two_overlapping_tiles():
    # same full-frame object (660..710, 10..60) detected independently in both
    # overlapping tiles that contain it -- must collapse to one via NMS
    dets_per_tile = [
        ((600, 0, 1240, 640), [(0, 0.9, 60, 10, 110, 60)]),   # -> full-frame 660,10,710,60
        ((640, 0, 1280, 640), [(0, 0.85, 20, 10, 70, 60)]),   # -> full-frame 660,10,710,60
    ]
    merged = merge_tile_detections(dets_per_tile, iou_thresh=0.5)
    assert len(merged) == 1
    assert merged[0][1] == 0.9  # keeps the higher-confidence detection


def test_merge_keeps_distinct_detections_in_different_classes_even_if_boxes_overlap():
    dets_per_tile = [
        ((0, 0, 640, 640), [(0, 0.9, 10, 10, 60, 60), (1, 0.8, 10, 10, 60, 60)]),
    ]
    merged = merge_tile_detections(dets_per_tile, iou_thresh=0.5)
    assert len(merged) == 2
