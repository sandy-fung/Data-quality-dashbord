"""Plotly figure builders for the JSON quality dashboard."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def numeric_histogram(data: Dict[str, List], column: str) -> go.Figure:
    """Build histogram figure from pre-computed bins."""
    fig = go.Figure()
    bin_centers = [
        (start + end) / 2 for start, end in zip(data["bin_start"], data["bin_end"])
    ]

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=data["count"],
            hovertemplate="Range: %{customdata}<br>Count: %{y}<extra></extra>",
            customdata=[
                f"{start:.3f} – {end:.3f}"
                for start, end in zip(data["bin_start"], data["bin_end"])
            ],
        )
    )
    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Count",
        bargap=0.05,
    )
    return fig


def categorical_bar(data: Dict[str, List], column: str) -> go.Figure:
    """Build bar chart for categorical distribution."""
    fig = px.bar(
        x=data["label"],
        y=data["count"],
        labels={"x": column, "y": "Count"},
        text_auto=True,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title=f"Top categories for {column}", xaxis_tickangle=-30)
    return fig


def time_series_line(grouped: pd.DataFrame, value_label: str) -> go.Figure:
    """Build time series line plot with range slider."""
    fig = px.line(grouped, x="timestamp", y="mean", markers=True)
    fig.update_layout(
        title=f"Time Series ({value_label})",
        xaxis_title="Timestamp",
        yaxis_title=value_label,
    )
    fig.add_bar(
        x=grouped["timestamp"],
        y=grouped["count"],
        name="Count",
        yaxis="y2",
        marker_color="rgba(99, 110, 250, 0.3)",
    )
    fig.update_layout(
        yaxis2=dict(
            title="Sample Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    return fig


def correlation_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Build correlation matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
    )
    return fig


def error_overlay_bar(
    *,
    labels: List[str],
    total_counts: List[int],
    error_counts: List[int],
) -> go.Figure:
    """Overlay error counts on top of total counts using shared axes."""
    if not (len(labels) == len(total_counts) == len(error_counts)):
        raise ValueError("labels, total_counts, and error_counts must have the same length")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=total_counts,
            name="Total Records",
            marker_color="rgba(99, 110, 250, 0.45)",
            hovertemplate="%{x}<br>Total: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=error_counts,
            name="Error Records",
            marker_color="rgba(255, 127, 14, 0.85)",
            hovertemplate="%{x}<br>Errors: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Error Counts by Column",
        xaxis_title="Column",
        yaxis_title="Records",
        barmode="overlay",
        bargap=0.2,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def run_accuracy_chart(
    *,
    labels: List[str],
    run_counts: List[int],
    dataset_total: int,
) -> go.Figure:
    """Display run-level error totals with optional accuracy line."""
    if len(labels) != len(run_counts):
        raise ValueError("labels and run_counts must have the same length")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=run_counts,
            name="New Additions",
            marker_color="rgba(155, 89, 182, 0.85)",
            hovertemplate="%{x}<br>New additions: %{y}<extra></extra>",
        )
    )

    if dataset_total > 0:
        accuracy_pct = [
            max(0.0, min(1.0, 1 - (count / dataset_total))) * 100 for count in run_counts
        ]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=accuracy_pct,
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="rgba(52, 152, 219, 0.9)", shape="spline"),
                marker=dict(size=8),
                yaxis="y2",
                hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Run Count"),
            yaxis2=dict(
                title="Accuracy (%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                tickformat=".0f",
            ),
        )
    else:
        fig.update_layout(yaxis=dict(title="Run Count"))

    fig.update_layout(
        title="Run Error Volume",
        xaxis_title="Run",
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def numeric_accuracy_overlay(
    *,
    column: str,
    bin_start: List[float],
    bin_end: List[float],
    dataset_counts: List[int],
    new_counts: List[int],
    wrong_label: str = "Wrong",
) -> go.Figure:
    """Display numeric distribution with total vs wrong counts and accuracy line."""
    if not (
        len(bin_start) == len(bin_end) == len(dataset_counts) == len(new_counts)
    ):
        raise ValueError("bin arrays and count arrays must have the same length")

    bin_centers = [(start + end) / 2 for start, end in zip(bin_start, bin_end)]
    hover_ranges = [f"{start:.3f} – {end:.3f}" for start, end in zip(bin_start, bin_end)]

    fig = go.Figure()
    dataset_customdata = [
        (hover_range, wrong_count)
        for hover_range, wrong_count in zip(hover_ranges, new_counts)
    ]
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=dataset_counts,
            name="Total",
            marker_color="rgba(52, 73, 94, 0.4)",
            hovertemplate=(
                "Range: %{customdata[0]}<br>Total: %{y}"
                f"<br>{wrong_label}: %{{customdata[1]}}<extra></extra>"
            ),
            customdata=dataset_customdata,
        )
    )
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=new_counts,
            name=wrong_label,
            marker_color="rgba(155, 89, 182, 0.85)",
            hovertemplate="Range: %{customdata}<br>%{fullData.name}: %{y}<extra></extra>",
            customdata=hover_ranges,
        )
    )

    accuracy_points: List[Optional[float]] = []
    for dataset_count, new_count in zip(dataset_counts, new_counts):
        if dataset_count > 0:
            accuracy = max(0.0, min(1.0, 1 - (new_count / dataset_count)))
            accuracy_points.append(accuracy * 100)
        else:
            accuracy_points.append(None)

    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=accuracy_points,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="rgba(52, 152, 219, 0.9)", shape="spline"),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            connectgaps=False,
        )
    )

    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Accuracy (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
            tickformat=".0f",
        ),
        barmode="overlay",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def stratified_heatmap(
    *,
    stats_df: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_title: str,
    colorscale: str = "Blues",
    zmin: float | None = None,
    zmax: float | None = None,
    fill_value: float | None = 0.0,
    text_formatter: Callable[[float], str] | None = None,
    x_title: str = "",
    y_title: str = "",
    value_label: str | None = "Value",
    value_format: str = "%{z}",
    hover_column: str | None = None,
    hover_label: str | None = "Count",
    hover_format: str = "%{customdata}",
) -> go.Figure:
    """Render a heatmap from stratified bin statistics."""
    fig = go.Figure()
    if stats_df.empty:
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
        return fig

    x_labels = (
        stats_df[["x_label", "x_order"]]
        .drop_duplicates()
        .sort_values("x_order")
        ["x_label"]
        .tolist()
    )
    strata_labels = (
        stats_df[["strata_label", "strata_order"]]
        .drop_duplicates()
        .sort_values("strata_order")
        ["strata_label"]
        .tolist()
    )

    pivot = (
        stats_df.pivot_table(
            index="strata_label",
            columns="x_label",
            values=value_column,
            aggfunc="first",
        )
        .reindex(index=strata_labels, columns=x_labels)
    )

    matrix = pivot.copy()
    if fill_value is not None:
        matrix = matrix.fillna(fill_value)

    customdata = None
    if hover_column is not None and hover_column in stats_df.columns:
        hover_pivot = (
            stats_df.pivot_table(
                index="strata_label",
                columns="x_label",
                values=hover_column,
                aggfunc="first",
            )
            .reindex(index=strata_labels, columns=x_labels)
        )
        hover_matrix = hover_pivot.fillna(0)
        customdata = hover_matrix.to_numpy()[..., None]

    text = None
    if text_formatter is not None:
        text = [
            [
                text_formatter(value) if not pd.isna(value) else ""
                for value in row
            ]
            for row in matrix.values
        ]

    hover_lines: List[str] = []
    if value_label:
        hover_lines.append(f"{value_label}: {value_format}")
    if customdata is not None and hover_label:
        hover_lines.append(f"{hover_label}: {hover_format}")

    if hover_lines:
        hover_body = "<br>".join(hover_lines)
        hover_template = f"Strata: %{{y}}<br>X bin: %{{x}}<br>{hover_body}<extra></extra>"
    else:
        hover_template = "Strata: %{y}<br>X bin: %{x}<extra></extra>"

    heatmap = go.Heatmap(
        z=matrix.values,
        x=x_labels,
        y=strata_labels,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
        zmin=zmin,
        zmax=zmax,
        text=text,
        texttemplate="%{text}" if text is not None else None,
        hovertemplate=hover_template,
        customdata=customdata,
    )
    fig.add_trace(heatmap)
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def grouped_box_plot(
    *,
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
    title: str,
) -> go.Figure:
    """Render a grouped box plot for binned numeric distributions."""
    fig = go.Figure()
    if data.empty:
        fig.update_layout(title=title, xaxis_title=group_column, yaxis_title=value_column)
        return fig

    # Ensure plotting order follows categorical codes.
    ordered_categories = (
        data[[group_column, "group_order"]]
        .drop_duplicates()
        .sort_values("group_order")[group_column]
        .tolist()
    )

    plot_df = data.copy()
    plot_df[group_column] = pd.Categorical(
        plot_df[group_column],
        categories=ordered_categories,
        ordered=True,
    )
    fig = px.box(
        plot_df,
        x=group_column,
        y=value_column,
        points="outliers",
    )
    fig.update_layout(
        title=title,
        xaxis_title=group_column,
        yaxis_title=value_column,
        xaxis_tickangle=-30,
    )
    return fig
