"""
Baseline listener implementations and ART predictor helpers.
Implements the baseline comparisons discussed in Section 4 and Appendix C.3.
"""

from typing import List, Dict, Any, Optional, Tuple


class ARTBaseline:
    """
    Reader and coordinate adapter for ART model predictions (Mondal et al., 2024).
    ART outputs fixations in 512x320 letterbox coordinates.
    """

    @staticmethod
    def entry_to_normalized_gaze(entry: Dict[str, Any]) -> List[Optional[Tuple[float, float]]]:
        """
        Convert ART entry with X and Y lists (in 512x320) to normalized 0-100 coordinates.
        Takes the final fixation per word window.
        """
        if not entry or "X" not in entry or "Y" not in entry:
            return []

        x_seq = [x[-1] if isinstance(x, list) and len(x) > 0 else None for x in entry["X"]]
        y_seq = [y[-1] if isinstance(y, list) and len(y) > 0 else None for y in entry["Y"]]

        norm_points = []
        for x, y in zip(x_seq, y_seq):
            if x is not None and y is not None:
                norm_x = (float(x) / 512.0) * 100.0
                norm_y = (float(y) / 320.0) * 100.0
                norm_points.append((norm_x, norm_y))
            else:
                norm_points.append(None)
        return norm_points


class BboxCenterBaseline:
    """
    Baseline listener (Appendix C.3.2) that always fixates on the center
    of the target referent bounding box: (x + w/2, y + h/2).
    """

    @staticmethod
    def get_center_fixation(bbox_100: List[float]) -> Optional[Tuple[float, float]]:
        if not bbox_100 or len(bbox_100) != 4:
            return None
        x, y, w, h = bbox_100
        return (x + w / 2.0, y + h / 2.0)


class LastPointBaseline:
    """
    Baseline listener (Appendix C.3.1) that only predicts a final fixation point at EOS.
    """

    @staticmethod
    def get_final_fixation(points: List[Optional[Tuple[float, float]]]) -> Optional[Tuple[float, float]]:
        for p in reversed(points or []):
            if p is not None:
                return p
        return None
