"""Tests for core.metrics module."""

import pandas as pd
import pytest

from core import metrics


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "value": [10, 12, 9, 14, 13, 15],
            "category": ["A", "B", "A", "A", "B", "C"],
            "intensity": [0.1, 0.2, 0.5, 0.4, 0.2, 0.1],
        }
    )


def test_detect_columns(sample_df: pd.DataFrame) -> None:
    numeric_cols = metrics.detect_numeric_columns(sample_df)
    assert "value" in numeric_cols
    assert "intensity" in numeric_cols

    datetime_cols = metrics.detect_datetime_columns(sample_df)
    assert datetime_cols == ["timestamp"]

    cat_cols = metrics.detect_categorical_columns(sample_df)
    assert "category" in cat_cols


def test_numeric_distribution(sample_df: pd.DataFrame) -> None:
    distribution = metrics.compute_numeric_distribution(sample_df, "value", bins=3)
    assert sum(distribution["count"]) == len(sample_df)


def test_categorical_distribution(sample_df: pd.DataFrame) -> None:
    distribution = metrics.compute_categorical_distribution(sample_df, "category")
    assert distribution["label"][0] == "A"
    assert distribution["count"][0] == 3


def test_time_series(sample_df: pd.DataFrame) -> None:
    result = metrics.compute_time_series(sample_df, "timestamp", "value", freq="D")
    assert not result.empty
    assert "mean" in result.columns
    assert "count" in result.columns


def test_correlation_heatmap(sample_df: pd.DataFrame) -> None:
    matrix = metrics.compute_correlation_heatmap(sample_df, columns=["value", "intensity"])
    assert matrix.shape == (2, 2)
    assert matrix.loc["value", "value"] == pytest.approx(1.0)


def test_compute_error_breakdown_orders_by_errors() -> None:
    summary = {
        "alpha": {"count": 8, "missing": 2},
        "beta": {"count": 10, "missing": 0, "non_hashable_values": 3},
    }
    result = metrics.compute_error_breakdown(summary)
    assert list(result["column"]) == ["beta", "alpha"]
    assert result.loc[0, "error_count"] == 3
    assert result.loc[1, "error_count"] == 2
    assert pytest.approx(result.loc[0, "error_rate"]) == 3 / (10 + 0)
    assert pytest.approx(result.loc[1, "error_rate"]) == 2 / (8 + 2)


def test_compute_error_breakdown_handles_empty_summary() -> None:
    result = metrics.compute_error_breakdown({})
    assert result.empty
    assert list(result.columns) == [
        "column",
        "non_null_count",
        "missing_count",
        "non_hashable_values",
        "total_rows",
        "error_count",
        "error_rate",
    ]


def test_compute_stratified_accuracy_grid_counts(sample_df: pd.DataFrame) -> None:
    detail_df = pd.DataFrame(
        {
            "matched": [True, True],
            "data_value": [10, 14],
            "data_intensity": [0.1, 0.4],
        }
    )
    grid = metrics.compute_stratified_accuracy_grid(
        sample_df,
        x_column="value",
        strata_column="intensity",
        x_bins=2,
        strata_bins=2,
        detail_df=detail_df,
    )
    assert {"dataset_count", "wrong_count", "accuracy_pct"}.issubset(grid.columns)
    assert grid["dataset_count"].sum() == len(sample_df)
    assert grid["wrong_count"].sum() == 2
    populated = grid[grid["dataset_count"] > 0].iloc[0]
    expected_accuracy = 100.0 - (populated["wrong_count"] / populated["dataset_count"]) * 100.0
    assert populated["accuracy_pct"] == pytest.approx(expected_accuracy)


def test_prepare_grouped_box_data_structure(sample_df: pd.DataFrame) -> None:
    grouped = metrics.prepare_grouped_box_data(
        sample_df,
        group_column="value",
        value_column="intensity",
        bins=3,
    )
    assert set(grouped.columns) == {"group_label", "value", "group_order"}
    assert not grouped.empty
    assert grouped["group_label"].dtype.name == "category"
    assert grouped["group_order"].min() == 0
