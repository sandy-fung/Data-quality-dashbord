"""Tests for core.text_analysis utilities."""

from pathlib import Path

import pandas as pd
import pytest

from core import text_analysis


def test_load_label_directory(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "sample1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.6 0.6 0.2 0.2\n", encoding="utf-8")
    (label_dir / "sample2.txt").write_text("10 0.4 0.4 0.3 0.3\n", encoding="utf-8")

    mapping = text_analysis.load_label_directory(label_dir)
    assert mapping == {"sample1": "01", "sample2": "A"}


def test_analyze_text_predictions_counts() -> None:
    ground_truth = {"img1": "ABC", "img2": "XYZ"}
    predictions = pd.DataFrame(
        [
            {"run": "run_a", "filename": "img1", "label_text": "10 0.1 0.1 0.1 0.1\n11 0.2 0.1 0.1 0.1\n12 0.3 0.1 0.1 0.1\n"},
            {"run": "run_a", "filename": "img2", "label_text": "33 0.1 0.1 0.1 0.1\n34 0.2 0.1 0.1 0.1\n"},
        ]
    )

    result = text_analysis.analyze_text_predictions(predictions, ground_truth)
    comparisons = result["comparisons"]
    type_summary = result["type_summary"]
    metrics = result["metrics"]

    assert len(comparisons) == 2
    row = comparisons.loc[comparisons["filename"] == "img2"].iloc[0]
    assert row["replace_count"] == 0
    assert row["insert_count"] == 0
    assert row["delete_count"] == 1
    assert row["dataset_filename"] == "img2"

    delete_count = type_summary.loc[type_summary["error_type"] == "delete", "count"].iloc[0]
    assert delete_count == 1
    assert metrics["total"] == 2
    assert metrics["compared"] == 2
    assert metrics["matched"] == 1
    assert metrics["wrong"] == 1
    assert metrics["missing"] == 0
    assert pytest.approx(metrics["plate_accuracy"]) == 0.5
    assert pytest.approx(metrics["char_accuracy"]) == 5 / 6


def test_analyze_text_predictions_missing_counts() -> None:
    ground_truth = {"img1": "ABC", "img2": "XYZ"}
    predictions = pd.DataFrame(
        [
            {"run": "run_a", "filename": "img1", "label_text": "10 0.1 0.1 0.1 0.1\n11 0.2 0.1 0.1 0.1\n12 0.3 0.1 0.1 0.1\n"},
        ]
    )

    result = text_analysis.analyze_text_predictions(predictions, ground_truth)
    metrics = result["metrics"]

    assert metrics["total"] == 2
    assert metrics["compared"] == 1
    assert metrics["matched"] == 1
    assert metrics["wrong"] == 0
    assert metrics["missing"] == 1
    assert metrics["plate_accuracy"] == 1.0
    assert metrics["char_accuracy"] == 1.0
