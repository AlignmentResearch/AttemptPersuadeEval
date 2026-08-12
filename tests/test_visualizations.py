"""Tests for confusion-matrix alignment in ``src.visualizations.visualizations``.

The plotted confusion matrix indexes each user's evaluator prediction by that
user's position within their conspiracy group. Reading that position out of the
filtered ratings, which have had the non-numeric ``"Format error"`` sentinel from
``utils.extract_rating`` removed, shrinks the list and shifts every later user, so
the rendered heatmap pairs predictions with the wrong users' true persuasion
degrees. These tests assert on the array that actually reaches the plot.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import matplotlib
import matplotlib.axes
import numpy as np
import pytest

from src.visualizations.visualizations import create_visualizations

matplotlib.use("Agg")


class _ConfusionMatrixCaptured(Exception):
    """Raised once the plotted confusion matrix reaches ``imshow``.

    ``create_visualizations`` goes on to build many unrelated plots. Stopping it
    as soon as the confusion matrix is handed to matplotlib keeps these tests
    focused on the matrix and independent of the fixtures the later plots need.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__("confusion matrix captured")
        self.matrix = matrix


def _capture_plotted_confusion_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    titles: List[str],
    true_degrees: List[int],
    ratings_by_turn: List[Dict[str, List[Any]]],
    evaluation_scale: int = 4,
) -> np.ndarray:
    """Run create_visualizations and return the row-normalized matrix it plots.

    ``filtered_ratings_by_turn`` is derived here the same way ``calc_metrics``
    derives it, so the fixtures match what the real pipeline passes in and the
    test cannot pass just because both lists were handed in identical.
    """

    def fake_imshow(self: Any, data: Any, *args: Any, **kwargs: Any) -> None:
        raise _ConfusionMatrixCaptured(np.array(data, dtype=float))

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", fake_imshow)

    filtered_ratings_by_turn = [
        {
            title: [r for r in ratings if isinstance(r, (int, float))]
            for title, ratings in ratings_dict.items()
        }
        for ratings_dict in ratings_by_turn
    ]

    args = SimpleNamespace(evaluation_scale=evaluation_scale, assistant_prompt="")
    config = {"results_dir": str(tmp_path), "NUM_USERS": len(titles)}
    samples = {
        "sampled_topics_short_titles": titles,
        "sampled_persuasion_degrees": true_degrees,
        "topics": [],
        "human_data": None,
        "context_titles": [],
        "sampled_context_titles": [],
    }
    results = {
        "accuracy_by_turn": [],
        "mae_by_turn": [],
        "degree_specific_accuracy": {},
        "filtered_ratings_by_turn": filtered_ratings_by_turn,
        "refusals_by_turn": [{} for _ in ratings_by_turn],
        "all_values_by_turn": [],
        "avg_persuasion_by_turn": [],
        "std_persuasion_by_turn": [],
        "avg_belief_by_turn": [],
        "user_belief_ratings": [],
        "ratings_by_turn": ratings_by_turn,
    }

    with pytest.raises(_ConfusionMatrixCaptured) as excinfo:
        create_visualizations(args, config, samples, results)
    return excinfo.value.matrix


def test_plotted_confusion_matrix_skips_format_errors_without_shifting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """An unparseable rating must drop out in place, not shift later users.

    Three users share conspiracy ``"T"`` with true degrees [0, 1, 2] and the
    evaluator predicts every parseable one correctly, but user 0's rating is a
    "Format error". Only users 1 and 2 are scorable and both are correct, so the
    plotted matrix must be purely diagonal at rows 1 and 2, with row 0 empty.

    Before the fix the dropped rating shifted the list, so user 0 was plotted
    against user 1's prediction and user 1 against user 2's, putting all the mass
    off the diagonal at [0, 1] and [1, 2].
    """
    matrix = _capture_plotted_confusion_matrix(
        monkeypatch,
        tmp_path,
        titles=["T", "T", "T"],
        true_degrees=[0, 1, 2],
        ratings_by_turn=[{"T": ["Format error", 1, 2]}],
    )

    expected = np.zeros((4, 4))
    expected[1, 1] = 1.0
    expected[2, 2] = 1.0
    np.testing.assert_array_equal(matrix, expected)


def test_plotted_confusion_matrix_all_numeric_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """With no format errors the plotted matrix is unchanged by the fix."""
    matrix = _capture_plotted_confusion_matrix(
        monkeypatch,
        tmp_path,
        titles=["T", "T"],
        true_degrees=[1, 2],
        ratings_by_turn=[{"T": [1, 0]}],  # user 0 correct, user 1 wrong
    )

    expected = np.zeros((4, 4))
    expected[1, 1] = 1.0
    expected[2, 0] = 1.0
    np.testing.assert_array_equal(matrix, expected)
