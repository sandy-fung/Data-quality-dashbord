"""Tests for core.plots overlay helpers."""

import pandas as pd
import pytest

pytest.importorskip("plotly")

from core import plots


def test_error_overlay_bar_overlay_mode() -> None:
    fig = plots.error_overlay_bar(
        labels=["col_a", "col_b"],
        total_counts=[10, 5],
        error_counts=[3, 1],
    )
    assert fig.layout.barmode == "overlay"
    assert len(fig.data) == 2
    assert fig.data[0].name == "Total Records"
    assert fig.data[1].name == "Error Records"
    assert fig.data[0].marker.color == "rgba(99, 110, 250, 0.45)"
    assert fig.data[1].marker.color == "rgba(255, 127, 14, 0.85)"


def test_error_overlay_bar_validates_lengths() -> None:
    with pytest.raises(ValueError):
        plots.error_overlay_bar(
            labels=["col_a", "col_b"],
            total_counts=[10],
            error_counts=[2, 1],
        )


def test_run_accuracy_chart_with_accuracy_line() -> None:
    fig = plots.run_accuracy_chart(
        labels=["run_a", "run_b"],
        run_counts=[10, 20],
        dataset_total=100,
    )
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[0].name == "New Additions"
    assert fig.data[1].type == "scatter"
    assert fig.data[1].name == "Accuracy"
    # Accuracy for run_a = 1 - 10/100 = 0.9 -> 90%
    assert fig.data[1].y[0] == pytest.approx(90.0)


def test_run_accuracy_chart_without_dataset_total() -> None:
    fig = plots.run_accuracy_chart(
        labels=["run_a"],
        run_counts=[5],
        dataset_total=0,
    )
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_numeric_accuracy_overlay_shapes() -> None:
    fig = plots.numeric_accuracy_overlay(
        column="value",
        bin_start=[0.0, 1.0],
        bin_end=[1.0, 2.0],
        dataset_counts=[20, 10],
        new_counts=[2, 5],
        wrong_label="Run A",
    )
    assert len(fig.data) == 3
    assert fig.data[0].name == "Total"
    assert fig.data[1].name == "Run A"
    assert fig.data[2].name == "Accuracy"
    assert fig.data[2].y[0] == pytest.approx(90.0)


def test_numeric_accuracy_overlay_handles_zero_dataset() -> None:
    fig = plots.numeric_accuracy_overlay(
        column="value",
        bin_start=[0.0],
        bin_end=[1.0],
        dataset_counts=[0],
        new_counts=[3],
    )
    assert fig.data[2].y[0] is None


def test_stratified_heatmap_renders_heatmap() -> None:
    stats_df = pd.DataFrame(
        {
            "x_label": ["0.0 to 1.0", "1.0 to 2.0"],
            "strata_label": ["0.0 to 0.5", "0.5 to 1.0"],
            "dataset_count": [5, 2],
            "wrong_count": [1, 0],
            "accuracy_pct": [80.0, 100.0],
            "x_order": [0, 1],
            "strata_order": [0, 1],
        }
    )
    fig = plots.stratified_heatmap(
        stats_df=stats_df,
        value_column="dataset_count",
        title="Sample",
        colorbar_title="Count",
        value_label="Count",
        value_format="%{z}",
    )
    assert fig.data
    assert fig.data[0].type == "heatmap"
    assert "Count" in fig.data[0].hovertemplate


def test_stratified_heatmap_custom_hover_shows_counts() -> None:
    stats_df = pd.DataFrame(
        {
            "x_label": ["0.0 to 1.0"],
            "strata_label": ["0.0 to 0.5"],
            "dataset_count": [8],
            "wrong_count": [2],
            "accuracy_pct": [75.0],
            "x_order": [0],
            "strata_order": [0],
        }
    )
    fig = plots.stratified_heatmap(
        stats_df=stats_df,
        value_column="accuracy_pct",
        title="Accuracy",
        colorbar_title="Accuracy (%)",
        value_label=None,
        hover_column="dataset_count",
        hover_label="Count",
    )
    heatmap = fig.data[0]
    assert heatmap.customdata[0][0][0] == 8
    assert "Accuracy (%)" not in heatmap.hovertemplate
    assert "Count" in heatmap.hovertemplate


def test_grouped_box_plot_renders_box() -> None:
    data = pd.DataFrame(
        {
            "group_label": pd.Categorical(["0 to 1", "0 to 1", "1 to 2"], ordered=True),
            "value": [0.1, 0.2, 0.3],
            "group_order": [0, 0, 1],
        }
    )
    fig = plots.grouped_box_plot(
        data=data,
        group_column="group_label",
        value_column="value",
        title="Grouped Box",
    )
    assert fig.data
    assert fig.data[0].type == "box"
