"""
Smoke test for human evaluation data loading and metric reproduction.
Verifies that Table 4, Table 5, and Appendix B.1 can be deterministically reproduced.
"""

from pathlib import Path
import pytest
from scripts.analysis.reproduce_human_eval import reproduce


TRIALS_PATH = Path(__file__).resolve().parent.parent / "results" / "human_eval" / "human_eval_trials_21600.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "human_eval"


@pytest.mark.skipif(not TRIALS_PATH.exists(), reason="Human evaluation trials JSON not found")
def test_human_eval_reproducibility():
    """Verify that human evaluation metrics can be loaded and reproduced from trials JSON."""
    df_overall, df_per_dataset = reproduce(TRIALS_PATH, OUTPUT_DIR, verify=True)

    assert len(df_overall) == 9, "Expected 9 speaker models in overall table"
    assert len(df_per_dataset) == 36, "Expected 36 rows (9 models x 4 datasets)"

    # Check key paper conclusions
    gold_row = df_overall[df_overall["Model"] == "Gold"].iloc[0]
    assert gold_row["Accuracy"] > 92.0

    shaping_row = df_overall[df_overall["Model"] == "Gaze-Shaping"].iloc[0]
    assert shaping_row["RefLen"] < 4.1
    assert shaping_row["Accuracy"] > 75.0

    seq_any_row = df_overall[df_overall["Model"] == "Gaze-SeqAnyHit"].iloc[0]
    assert seq_any_row["Accuracy"] >= 80.0
    assert seq_any_row["d_NEG"] < 0.90
