"""
Unit tests for coordinate regex parsing and sequence formatting.
"""

from gaze_estimation.models.sequence import (
    parse_gaze_point,
    parse_gaussian_gaze_point,
    parse_gaze_sequence,
    format_gaze_point,
    build_incremental_prompt,
)


def test_parse_gaze_point():
    assert parse_gaze_point("<x=50.2 y=48.9>") == (50.2, 48.9)
    assert parse_gaze_point("<x=0.0 y=100.0>") == (0.0, 100.0)
    assert parse_gaze_point("<x=50 y=50>") == (50.0, 50.0)
    assert parse_gaze_point("<x= y=>") is None
    assert parse_gaze_point("no point here") is None


def test_parse_gaussian_gaze_point():
    assert parse_gaussian_gaze_point("<x=50.2 y=48.9 v=10.5>") == (50.2, 48.9, 10.5)
    assert parse_gaussian_gaze_point("<x=50.2 y=48.9>") is None


def test_parse_gaze_sequence():
    seq = "BOS <x=50.2 y=48.9> The <x=48.1 y=45.2> cat <x= y=> EOS <x=35.4 y=65.2>"
    points = parse_gaze_sequence(seq)
    assert len(points) == 4
    assert points[0] == (50.2, 48.9)
    assert points[1] == (48.1, 45.2)
    assert points[2] is None
    assert points[3] == (35.4, 65.2)


def test_format_and_prompt_builder():
    assert format_gaze_point(50.23, 48.91) == "<x=50.2 y=48.9>"
    assert format_gaze_point(None, None) == "<x= y=>"

    prompt = build_incremental_prompt(["The", "cat"], [(50.0, 50.0), (45.0, 45.0)])
    assert prompt == "BOS <x=50.0 y=50.0> The <x=45.0 y=45.0> cat"
