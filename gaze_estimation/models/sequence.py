"""
Sequence and coordinate grammar utilities for Molmo-REC-Gaze.
Defines regex parsers and formatters matching the GazeRL shared contract.
"""

import re
from typing import Optional, Tuple, List

# Regexes matching the GazeRL contract specified in AGENT_GAZE_PREDICTION.md
POINT_REGEX = re.compile(r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)>', re.IGNORECASE)
GAUSSIAN_REGEX = re.compile(r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)\s+v=(\d+(?:\.\d+)?)>', re.IGNORECASE)
NULL_REGEX = re.compile(r'<x=\s*y=\s*>', re.IGNORECASE)
TAG_ANY_REGEX = re.compile(r'<x=(\d+(?:\.\d+)?|\s*)\s+y=(\d+(?:\.\d+)?|\s*)(?:\s+v=(\d+(?:\.\d+)?|\s*))?>', re.IGNORECASE)


def parse_gaze_point(text: str) -> Optional[Tuple[float, float]]:
    """
    Parse a single fixation point <x=... y=...> from text.
    Returns (x, y) on continuous 0-100 scale, or None if null/missing.
    """
    if not text:
        return None
    match = POINT_REGEX.search(text)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        return (x, y)
    return None


def parse_gaussian_gaze_point(text: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse a Gaussian fixation point <x=... y=... v=...> from text.
    Returns (x, y, sigma) where sigma in [0, 100], or None if null/missing.
    """
    if not text:
        return None
    match = GAUSSIAN_REGEX.search(text)
    if match:
        x, y, v = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return (x, y, v)
    return None


def parse_gaze_sequence(text: str) -> List[Optional[Tuple[float, float]]]:
    """
    Extract all fixation points from a full interleaved sequence.
    Returns a list of (x, y) coordinates or None for null tags (<x= y=>).
    """
    points = []
    for match in TAG_ANY_REGEX.finditer(text):
        x_str = match.group(1).strip()
        y_str = match.group(2).strip()
        if x_str and y_str:
            try:
                points.append((float(x_str), float(y_str)))
            except ValueError:
                points.append(None)
        else:
            points.append(None)
    return points


def format_gaze_point(x: Optional[float], y: Optional[float], precision: int = 1) -> str:
    """Format coordinate pair to <x=.. y=..> string."""
    if x is None or y is None:
        return "<x= y=>"
    return f"<x={x:.{precision}f} y={y:.{precision}f}>"


def build_incremental_prompt(
    words: List[str],
    predicted_gazes: List[Optional[Tuple[float, float]]],
) -> str:
    """
    Build incremental prompt sequence:
        BOS <x=.. y=..> w_1 <x=.. y=..> w_2 ...
    """
    parts = ["BOS"]
    if predicted_gazes:
        parts.append(format_gaze_point(*predicted_gazes[0]) if predicted_gazes[0] else "<x= y=>")

    for i, word in enumerate(words):
        parts.append(word)
        if i + 1 < len(predicted_gazes):
            g = predicted_gazes[i + 1]
            parts.append(format_gaze_point(*g) if g else "<x= y=>")

    return " ".join(parts)
