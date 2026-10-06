"""
Unit tests for image letterboxing and coordinate transforms.
"""

from PIL import Image
import pytest

from gaze_estimation.data.transforms import (
    letterbox_image,
    scale_coords_to_canvas,
    canvas_to_normalized,
    normalized_to_canvas,
    normalize_coords,
    normalize_bbox,
    is_point_in_bbox,
)


def test_letterbox_image_dimensions():
    # Test wide image
    img = Image.new("RGB", (800, 400), color="white")
    padded, scale, pad_x, pad_y = letterbox_image(img, 512, 320)
    assert padded.size == (512, 320)
    assert pad_x >= 0
    assert pad_y >= 0

    # Test tall image
    img_tall = Image.new("RGB", (400, 800), color="white")
    padded_tall, scale_t, pad_xt, pad_yt = letterbox_image(img_tall, 512, 320)
    assert padded_tall.size == (512, 320)
    assert pad_xt >= 0
    assert pad_yt >= 0


def test_coordinate_transforms_roundtrip():
    canvas_w, canvas_h = 512, 320
    orig_x, orig_y = 256.0, 160.0
    x_norm, y_norm = canvas_to_normalized(orig_x, orig_y, canvas_w, canvas_h)
    assert pytest.approx(x_norm) == 50.0
    assert pytest.approx(y_norm) == 50.0

    x_back, y_back = normalized_to_canvas(x_norm, y_norm, canvas_w, canvas_h)
    assert pytest.approx(x_back) == orig_x
    assert pytest.approx(y_back) == orig_y


def test_bbox_normalization_and_hit():
    raw_bbox = [840, 525, 420, 262.5]  # Center box in 1680x1050
    norm_bbox = normalize_bbox(raw_bbox, orig_w=1680, orig_h=1050)
    assert pytest.approx(norm_bbox[0]) == 50.0
    assert pytest.approx(norm_bbox[1]) == 50.0
    assert pytest.approx(norm_bbox[2]) == 25.0
    assert pytest.approx(norm_bbox[3]) == 25.0

    # Inside point
    assert is_point_in_bbox((60.0, 60.0), norm_bbox) is True
    # Outside point
    assert is_point_in_bbox((10.0, 10.0), norm_bbox) is False
