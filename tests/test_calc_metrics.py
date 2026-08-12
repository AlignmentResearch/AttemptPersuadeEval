"""Tests for evaluator-accuracy metric alignment in ``src.metrics.calc_metrics``.

These cover a bug where filtering non-numeric evaluator ratings (the
``"Format error"`` sentinel from ``utils.extract_rating``) out of a conspiracy
group's rating list shifted every later user's position, so predictions were
scored against the wrong users' intended persuasion degrees.
"""

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

from src.metrics.calc_metrics import calc_metrics, get_aligned_prediction


def test_get_aligned_prediction_skips_format_errors_without_shifting() -> None:
    """A non-numeric rating must be skipped in place, not shift later users.

    Three users share conspiracy ``"T"``; user 0's rating failed to parse. Users
    1 and 2 must still receive their own predictions (1 and 2), and user 0 must
    return ``None`` rather than borrowing a later user's rating.
    """
    titles = ["T", "T", "T"]
    ratings_dict = {"T": ["Format error", 1, 2]}

    assert get_aligned_prediction(ratings_dict, "T", 0, titles) is None
    assert get_aligned_prediction(ratings_dict, "T", 1, titles) == 1
    assert get_aligned_prediction(ratings_dict, "T", 2, titles) == 2


def _run_metrics(
    tmp_path: Any,
    titles: List[str],
    true_degrees: List[int],
    ratings_by_turn: List[Dict[str, List[Any]]],
    evaluation_scale: int = 4,
) -> Dict[str, Any]:
    """Call calc_metrics with minimal fixtures and return the accuracy metrics."""
    args = SimpleNamespace(
        evaluation_scale=evaluation_scale,
        experiment_name="test",
        num_turns=len(ratings_by_turn),
        persuader_model="persuader",
        persuadee_model="persuadee",
        evaluator_model="evaluator",
        belief_lower_threshold=0,
        belief_upper_threshold=100,
    )
    config = {"RUN_ID": "run", "NUM_USERS": len(titles), "results_dir": str(tmp_path)}
    samples = {
        "sampled_topics_short_titles": titles,
        "sampled_persuasion_degrees": true_degrees,
    }
    results = {
        "ratings_by_turn": ratings_by_turn,
        "message_collection": [],
        "refusals_by_turn": [],
    }

    calc_metrics(args, config, samples, results)

    with open(os.path.join(str(tmp_path), "evaluator_accuracy_metrics.json")) as f:
        accuracy_metrics: Dict[str, Any] = json.load(f)
    return accuracy_metrics


def test_evaluator_accuracy_ignores_unparseable_ratings(tmp_path: Any) -> None:
    """A perfect evaluator with one unparseable rating still scores 1.0.

    Users have true degrees [0, 1, 2] and the evaluator predicts every parseable
    one correctly, but user 0's rating is a "Format error". Only users 1 and 2 are
    scorable, both correct, so accuracy must be 1.0 and MAE 0.0. Before the fix the
    dropped rating shifted the remaining predictions, scoring 0.0 accuracy.
    """
    metrics = _run_metrics(
        tmp_path,
        titles=["T", "T", "T"],
        true_degrees=[0, 1, 2],
        ratings_by_turn=[{"T": ["Format error", 1, 2]}],
    )

    assert metrics["overall_accuracy_by_turn"] == [1.0]
    assert metrics["mean_absolute_error_by_turn"] == [0.0]
    assert metrics["mean_squared_error_by_turn"] == [0.0]


def test_evaluator_accuracy_all_numeric_unchanged(tmp_path: Any) -> None:
    """With no format errors, a half-correct evaluator scores as expected."""
    metrics = _run_metrics(
        tmp_path,
        titles=["T", "T"],
        true_degrees=[1, 2],
        ratings_by_turn=[{"T": [1, 0]}],  # user 0 correct, user 1 wrong
    )

    assert metrics["overall_accuracy_by_turn"] == [0.5]
    assert metrics["mean_absolute_error_by_turn"] == [1.0]
