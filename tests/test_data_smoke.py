"""
Milestone 1 Smoke Test: Data & Preprocessing Pipeline
Verifies split integrity, bbox normalization, image preprocessing, coordinate parsing, and batch collation.
"""
import pytest
import numpy as np
from PIL import Image

from data.processing import (
    parse_coordinate,
    parse_gaussian_coordinate,
    convert_bbox_to_minmax,
    normalize_bbox_for_gaze_predictor,
    create_speaker_bbox_overlay,
    create_listener_padded_image,
    calculate_distance,
    is_point_in_bbox,
    calculate_gaussian_bbox_probability,
    reconstruct_text_from_tokens,
)
from data.datasets import (
    DatasetSplitManager,
    SyntheticReferringExpressionDataset,
    collate_referring_expression_batch,
)
from configs.constants import LISTENER_IMAGE_WIDTH, LISTENER_IMAGE_HEIGHT, SPEAKER_MAX_DIM


def test_splits_integrity():
    """Verify that verified split JSONs are present and load properly."""
    manager = DatasetSplitManager(splits_dir="data/splits")
    
    # Test COCO 2014
    train_ids, val_ids = manager.load_splits("coco_2014")
    assert len(train_ids) > 0, "COCO-2014 train split is empty"
    assert len(val_ids) > 0, "COCO-2014 val split is empty"

    # Test RefCOCO
    ref_train, ref_val = manager.load_splits("refcoco")
    assert len(ref_train) > 0, "RefCOCO train split is empty"
    assert len(ref_val) > 0, "RefCOCO val split is empty"


def test_bbox_conversions():
    """Test COCO [x, y, w, h] to minmax conversion."""
    image_size = (640, 480)
    coco_box = [100, 150, 50, 60]  # x, y, w, h
    minmax = convert_bbox_to_minmax(coco_box, image_size)
    assert minmax == (100.0, 150.0, 150.0, 210.0)

    # Clamping test
    out_of_bounds = [-20, -10, 800, 600]
    clamped = convert_bbox_to_minmax(out_of_bounds, image_size)
    assert clamped[0] >= 0 and clamped[1] >= 0
    assert clamped[2] <= image_size[0] and clamped[3] <= image_size[1]


def test_normalize_bbox_for_gaze_predictor():
    """Verify bbox normalization maps properly into 0-100 listener space."""
    orig_size = (1000, 800)
    bbox_minmax = (100.0, 100.0, 500.0, 500.0)
    norm_bbox = normalize_bbox_for_gaze_predictor(bbox_minmax, orig_size)

    for val in norm_bbox:
        assert 0.0 <= val <= 100.0, f"Coordinate {val} outside [0, 100] normalized range"
    assert norm_bbox[0] < norm_bbox[2]
    assert norm_bbox[1] < norm_bbox[3]


def test_image_preprocessing():
    """Verify speaker and listener image dimensions."""
    test_img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    bbox = (100.0, 100.0, 300.0, 300.0)

    # Speaker image: resized with max dim <= SPEAKER_MAX_DIM
    speaker_img = create_speaker_bbox_overlay(test_img, bbox, max_dim=SPEAKER_MAX_DIM)
    w, h = speaker_img.size
    assert max(w, h) == SPEAKER_MAX_DIM
    assert isinstance(speaker_img, Image.Image)

    # Listener image: exact letterbox dimensions
    listener_img = create_listener_padded_image(test_img)
    assert listener_img.size == (LISTENER_IMAGE_WIDTH, LISTENER_IMAGE_HEIGHT)


def test_coordinate_parsing():
    """Test point and Gaussian coordinate parsing."""
    point = parse_coordinate("<x=45.2 y=67.8>")
    assert point == (45.2, 67.8)

    point_with_text = parse_coordinate("The object is at <x=12.0 y=99.5> in the corner")
    assert point_with_text == (12.0, 99.5)

    assert parse_coordinate("<x= y=>") is None
    assert parse_coordinate("No coordinates here") is None

    gaussian = parse_gaussian_coordinate("<x=50.0 y=40.0 v=8.5>")
    assert gaussian == (50.0, 40.0, 8.5)


def test_geometry_and_gaussian_probability():
    """Test distance, point-in-box, and Gaussian mass calculation."""
    p1 = (10.0, 20.0)
    p2 = (13.0, 24.0)
    assert calculate_distance(p1, p2) == pytest.approx(5.0)

    bbox = (20.0, 20.0, 40.0, 40.0)
    assert is_point_in_bbox((30.0, 30.0), bbox) is True
    assert is_point_in_bbox((10.0, 10.0), bbox) is False

    # Center of Gaussian inside bbox should have high probability mass
    prob_inside = calculate_gaussian_bbox_probability(30.0, 30.0, 5.0, bbox)
    assert prob_inside > 0.8, f"Expected high mass inside bbox, got {prob_inside}"

    # Far away Gaussian should have negligible probability mass
    prob_outside = calculate_gaussian_bbox_probability(90.0, 90.0, 5.0, bbox)
    assert prob_outside < 0.01, f"Expected near zero mass outside bbox, got {prob_outside}"


def test_dataset_and_batch_collation():
    """Test synthetic dataset generation and batch collation."""
    dataset = SyntheticReferringExpressionDataset(num_samples=4, seed=123)
    assert len(dataset) == 4

    item0 = dataset[0]
    assert item0.sample_id == "synthetic_0000"
    assert item0.speaker_image.size[0] <= SPEAKER_MAX_DIM
    assert item0.listener_image.size == (LISTENER_IMAGE_WIDTH, LISTENER_IMAGE_HEIGHT)
    assert len(item0.prompt) > 0

    # Test batch collate
    batch = collate_referring_expression_batch([dataset[0], dataset[1]])
    assert len(batch["sample_ids"]) == 2
    assert batch["bbox_normalized"].shape == (2, 4)
    assert len(batch["speaker_images"]) == 2
    assert len(batch["listener_images"]) == 2


def test_text_reconstruction():
    """Test token to text reconstruction."""
    tokens = ["The", "Ġred", "Ġapple"]
    text = reconstruct_text_from_tokens(tokens)
    assert "The" in text and "apple" in text
