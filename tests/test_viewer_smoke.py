"""
Smoke test for qualitative visualizer.
"""

from PIL import Image
from evaluation.qualitative import draw_gaze_trajectory


def test_draw_gaze_trajectory_smoke():
    """Verify trajectory visualization on synthetic image."""
    base_img = Image.new("RGB", (512, 320), color="black")
    bbox_100 = (30.0, 30.0, 70.0, 70.0)
    gaze_points = [(10.0, 10.0), (30.0, 30.0), (50.0, 50.0)]

    # Draw hit trajectory
    vis_hit = draw_gaze_trajectory(
        image=base_img,
        target_bbox_100=bbox_100,
        gaze_points_100=gaze_points,
        caption="the red circle in the center",
        is_success=True,
    )

    assert isinstance(vis_hit, Image.Image)
    w, h = vis_hit.size
    assert w == 512
    assert h >= 320

    # Draw miss trajectory
    vis_miss = draw_gaze_trajectory(
        image=base_img,
        target_bbox_100=bbox_100,
        gaze_points_100=[(10.0, 10.0), (20.0, 20.0)],
        caption="small square",
        is_success=False,
    )
    assert isinstance(vis_miss, Image.Image)
