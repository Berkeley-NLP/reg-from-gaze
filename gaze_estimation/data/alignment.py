"""
Temporal alignment and auditory latency processing for RefCOCO-Gaze.
Implements the +200ms audio latency shift and multi-fixation resolution
specified in Wright et al. (2026), Section 4.
"""

import json
from typing import List, Dict, Any, Optional, Tuple


def apply_auditory_latency_delay(
    data: List[Dict[str, Any]],
    word_timing: List[Dict[str, Any]],
    latency_ms: int = 200,
    default_word_duration_ms: int = 500,
) -> List[Dict[str, Any]]:
    """
    Apply +200ms auditory processing delay to word onsets and reassign fixation word indices.
    """
    timing_by_ref_id = {item["ref_id"]: item for item in word_timing if "ref_id" in item}

    for record in data:
        ref_id = record.get("REF_ID")
        timing_entry = timing_by_ref_id.get(ref_id)
        if not timing_entry or "word_onset" not in timing_entry:
            continue

        sound_on = record.get("SOUND_ON", 0)
        adjusted_onsets = [onset + sound_on + latency_ms for onset in timing_entry["word_onset"]]
        fix_start = record.get("FIX_START", [])

        word_durations = []
        for i in range(len(adjusted_onsets) - 1):
            word_durations.append(adjusted_onsets[i + 1] - adjusted_onsets[i])
        avg_duration = sum(word_durations) / len(word_durations) if word_durations else default_word_duration_ms
        last_word_end = adjusted_onsets[-1] + avg_duration

        adjusted_indices = []
        for fixation in fix_start:
            if fixation < adjusted_onsets[0]:
                adjusted_indices.append(-99)  # BOS
            elif fixation >= last_word_end:
                adjusted_indices.append(99)   # EOS
            else:
                word_index = next((i for i, onset in enumerate(adjusted_onsets) if fixation < onset), len(adjusted_onsets)) - 1
                adjusted_indices.append(word_index)

        record["FIX_WORDINDEX"] = adjusted_indices

    return data


def extract_word_aligned_scanpath(
    entry: Dict[str, Any],
    include_bos_eos: bool = True,
) -> List[Optional[Tuple[float, float]]]:
    """
    Extract gold gaze points from human data entry.
    Matches the exact indexing used in paper evaluation:
      - index 0: BOS (-99)
      - index 1..num_words: words (0..num_words-1)
      - index num_words+1: EOS (99)
    """
    ref_words = entry.get("REF_WORDS") or entry.get("ref_words", [])
    fix_x = entry.get("FIX_X") or entry.get("fix_x", [])
    fix_y = entry.get("FIX_Y") or entry.get("fix_y", [])
    fix_wordindex = entry.get("FIX_WORDINDEX") or entry.get("fix_wordindex", [])
    num_words = len(ref_words)

    if not fix_x or not fix_y:
        total_len = num_words + 2 if include_bos_eos else num_words
        return [None] * total_len

    last_fix = {}
    if include_bos_eos:
        for i, widx in enumerate(fix_wordindex):
            if widx == -99:
                last_fix[0] = i
            elif 0 <= widx < num_words:
                last_fix[widx + 1] = i
            elif widx == 99:
                last_fix[num_words + 1] = i

        total_positions = num_words + 2
        gold_gaze = []
        for idx in range(total_positions):
            if idx in last_fix:
                fi = last_fix[idx]
                gold_gaze.append([float(fix_x[fi]), float(fix_y[fi])])
            else:
                gold_gaze.append(None)
    else:
        for i, widx in enumerate(fix_wordindex):
            if 0 <= widx < num_words:
                last_fix[widx] = i
        gold_gaze = []
        for idx in range(num_words):
            if idx in last_fix:
                fi = last_fix[idx]
                gold_gaze.append([float(fix_x[fi]), float(fix_y[fi])])
            else:
                gold_gaze.append(None)

    return gold_gaze
