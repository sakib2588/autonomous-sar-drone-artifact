import pytest

from src.sar.data.sar_eval import (
    WISARD_PERSON_CLASS_ID,
    YoloBox,
    is_ir_sequence,
    is_vis_sequence,
    map_wisard_line,
    parse_wisard_label,
    sequence_id_of,
)


def test_wisard_class_zero_maps_to_person_which_is_also_zero():
    # WiSARD is person-only and already emits class 0; SAR_RGB_CLASSES puts
    # person at index 0 too, so the remap is identity -- but it is asserted
    # rather than assumed, because a silent shift would mislabel every box.
    assert map_wisard_line("0 0.5 0.5 0.1 0.2") == YoloBox(0, 0.5, 0.5, 0.1, 0.2)


def test_non_person_class_is_rejected_not_silently_remapped():
    # WiSARD ships person-only. Any other id means the assumption broke and the
    # mapping must fail loudly rather than coerce it into person.
    with pytest.raises(ValueError, match="expected person-only"):
        map_wisard_line("1 0.5 0.5 0.1 0.2")


def test_malformed_line_raises():
    with pytest.raises(ValueError, match="5 fields"):
        map_wisard_line("0 0.5 0.5 0.1")


def test_out_of_range_coordinate_raises():
    # Normalised YOLO coords must be in [0,1]; a pixel-space value slipping
    # through would place boxes off-image and silently deflate recall.
    with pytest.raises(ValueError, match="normalised"):
        map_wisard_line("0 640.0 360.0 40.0 80.0")


def test_parse_label_file_skips_blank_lines(tmp_path):
    p = tmp_path / "frame_00000001.txt"
    p.write_text("0 0.4 0.3 0.02 0.04\n\n0 0.6 0.7 0.03 0.05\n")
    boxes = parse_wisard_label(p)
    assert len(boxes) == 2
    assert boxes[0].cx == pytest.approx(0.4)


def test_count_txt_is_not_a_label_file(tmp_path):
    # Every WiSARD sequence directory carries a count.txt summary alongside the
    # per-frame labels. Treating it as a label file would raise on its prose.
    p = tmp_path / "count.txt"
    p.write_text("number of humans: 1022\n number of images: 264")
    with pytest.raises(ValueError, match="not a label file"):
        parse_wisard_label(p)


def test_empty_label_file_is_a_valid_negative_frame(tmp_path):
    # 3 of 264 sample frames had zero boxes. Those are true negatives and must
    # survive as empty, not be dropped -- dropping them inflates precision.
    p = tmp_path / "frame_00000002.txt"
    p.write_text("")
    assert parse_wisard_label(p) == []


def test_sequence_id_groups_frames_from_one_flight():
    a = sequence_id_of("210417_MtErie_Enterprise_VIS_0003_00000001.jpeg")
    b = sequence_id_of("210417_MtErie_Enterprise_VIS_0003_00000002.jpeg")
    c = sequence_id_of("210417_MtErie_Enterprise_VIS_0004_00000001.jpeg")
    assert a == b == "210417_MtErie_Enterprise_VIS_0003"
    assert a != c


def test_vis_and_ir_sequences_are_distinguishable():
    # The RGB branch must never ingest thermal frames; the taxonomies are
    # deliberately separate.
    assert is_vis_sequence("210417_MtErie_Enterprise_VIS_0003") is True
    assert is_vis_sequence("210417_MtErie_Enterprise_IR_0004") is False
    assert is_ir_sequence("210417_MtErie_Enterprise_IR_0004") is True


def test_modality_suffix_without_trailing_index_is_recognised():
    # Real naming in the extracted set: several sequences end in the modality
    # with no trailing number, which a `_VIS_` substring test misses.
    assert is_vis_sequence("200402_Carnation_Inspire_VIS") is True
    assert is_vis_sequence("200402_Karen_Inspire_VIS") is True


def test_flir_in_the_platform_name_is_not_read_as_thermal():
    # `200717_Mission_FLIR_VIS` is a VISUAL sequence shot on a FLIR platform.
    # A substring test for "IR" would misclassify it and silently pull thermal
    # frames into the RGB label space.
    assert is_vis_sequence("200717_Mission_FLIR_VIS") is True
    assert is_ir_sequence("200717_Mission_FLIR_VIS") is False
    assert is_vis_sequence("210327_Airfield_FLIR_VIS_1") is True


def test_person_class_id_constant_matches_taxonomy():
    from src.sar.data.taxonomy_rgb import rgb_class_id

    assert WISARD_PERSON_CLASS_ID == rgb_class_id("person")


def test_out_of_range_box_raises_by_default():
    # Default policy is strict, so a new data drop cannot silently introduce
    # boundary overruns without someone deciding what to do about them.
    with pytest.raises(ValueError, match="outside"):
        map_wisard_line("0 0.5 1.117148 0.02 0.04")


def test_clip_policy_keeps_the_visible_part_of_an_edge_box():
    # A person half out of frame is exactly the SAR case; keep what is visible.
    # cy=1.1 with h=0.4 spans y 0.9..1.3; only 0.9..1.0 is visible.
    b = map_wisard_line("0 0.5 1.1 0.2 0.4", on_out_of_range="clip")
    assert b is not None
    assert b.cy == pytest.approx(0.95)
    assert b.h == pytest.approx(0.1)
    # The in-frame axis is untouched.
    assert b.cx == pytest.approx(0.5)
    assert b.w == pytest.approx(0.2)


def test_clip_policy_drops_a_box_entirely_outside_the_frame():
    # 164 boxes sit fully off-image. Counting them as missed detections would
    # penalise the model for something not present.
    assert map_wisard_line("0 1.8 0.5 0.1 0.1", on_out_of_range="clip") is None


def test_pixel_space_labels_raise_even_under_clip():
    # Clipping a pixel-space label would turn a broken annotation into a
    # plausible-looking box. That must stay an error under every policy.
    with pytest.raises(ValueError, match="normalised"):
        map_wisard_line("0 640.0 360.0 40.0 80.0", on_out_of_range="clip")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown on_out_of_range"):
        map_wisard_line("0 0.5 1.5 0.1 0.1", on_out_of_range="squish")
