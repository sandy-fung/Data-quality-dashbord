"""Analytical metrics for the JSON quality dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from pandas import CategoricalDtype


@dataclass
class ChartConfig:
    """Chart rendering instructions."""

    chart_type: str
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    aggregation: Optional[str] = None


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric column names sorted by variance descending."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return []
    variance = df[numeric_cols].var().sort_values(ascending=False)
    return variance.index.tolist()


def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return datetime-like columns in the dataset."""
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()


def detect_categorical_columns(df: pd.DataFrame, max_unique: int = 32) -> List[str]:
    """
    Detect categorical columns by dtype and cardinality.

    Strings are considered categorical when unique values do not exceed max_unique.
    """
    candidates: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            candidates.append(column)
            continue
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, CategoricalDtype):
            unique_count = series.nunique(dropna=True)
            if 0 < unique_count <= max_unique:
                candidates.append(column)
    return candidates


def compute_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
) -> Dict[str, List]:
    """Return histogram-ready data for numeric column."""
    series = df[column].dropna()
    hist, bin_edges = pd.cut(series, bins=bins, retbins=True, include_lowest=True)
    counts = hist.value_counts(sort=False)
    return {
        "bin_start": [round(edge, 6) for edge in bin_edges[:-1]],
        "bin_end": [round(edge, 6) for edge in bin_edges[1:]],
        "count": counts.tolist(),
    }


def compute_categorical_distribution(df: pd.DataFrame, column: str, top: int = 20) -> Dict[str, List]:
    """Return counts for categorical column."""
    series = df[column].astype("string")
    counts = series.value_counts(dropna=False).head(top)
    return {
        "label": counts.index.fillna("<NA>").tolist(),
        "count": counts.tolist(),
    }


def compute_time_series(
    df: pd.DataFrame,
    datetime_column: str,
    value_column: str,
    freq: str = "D",
) -> pd.DataFrame:
    """Aggregate value column over time using chosen frequency."""
    if datetime_column not in df.columns:
        raise ValueError(f"Column '{datetime_column}' not found")
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found")

    time_series = df[[datetime_column, value_column]].dropna()
    if time_series.empty:
        return pd.DataFrame(columns=["timestamp", "value"])

    grouped = (
        time_series.set_index(datetime_column)
        .groupby(pd.Grouper(freq=freq))[value_column]
        .agg(["mean", "count", "min", "max"])
        .reset_index()
    )
    grouped.rename(columns={datetime_column: "timestamp"}, inplace=True)
    return grouped


def compute_numeric_summary(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Quick descriptive statistics for numeric columns."""
    series = df[column].dropna()
    return {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def compute_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Return correlation matrix for the selected numeric columns."""
    numeric_df = df if columns is None else df[columns]
    numeric_df = numeric_df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr()


def compute_error_breakdown(summary: Dict[str, Dict]) -> pd.DataFrame:
    """Compute per-column error statistics based on summary metadata."""
    records: List[Dict[str, float]] = []
    for column, meta in summary.items():
        non_null = int(meta.get("count") or 0)
        missing = int(meta.get("missing") or 0)
        non_hashable = int(meta.get("non_hashable_values") or 0)
        total_rows = non_null + missing
        error_count = missing + non_hashable
        error_rate = (error_count / total_rows) if total_rows else 0.0
        records.append(
            {
                "column": column,
                "non_null_count": non_null,
                "missing_count": missing,
                "non_hashable_values": non_hashable,
                "total_rows": total_rows,
                "error_count": error_count,
                "error_rate": error_rate,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "column",
                "non_null_count",
                "missing_count",
                "non_hashable_values",
                "total_rows",
                "error_count",
                "error_rate",
            ]
        )

    breakdown = pd.DataFrame(records)
    breakdown.sort_values(by="error_count", ascending=False, inplace=True)
    breakdown.reset_index(drop=True, inplace=True)
    return breakdown


def compute_stratified_accuracy_grid(
    df: pd.DataFrame,
    *,
    x_column: str,
    strata_column: str,
    x_bins: int,
    strata_bins: int,
    detail_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute dataset counts and accuracy for two-dimensional numeric bins.

    detail_df is expected to contain matched label rows with columns prefixed by ``data_``.
    """
    if x_column not in df.columns or strata_column not in df.columns:
        raise ValueError("Selected columns must exist in the dataset.")

    numeric_x = pd.to_numeric(df[x_column], errors="coerce")
    numeric_strata = pd.to_numeric(df[strata_column], errors="coerce")
    valid_mask = numeric_x.notna() & numeric_strata.notna()
    if not valid_mask.any():
        return pd.DataFrame(
            columns=[
                "x_label",
                "strata_label",
                "dataset_count",
                "wrong_count",
                "accuracy_pct",
                "x_order",
                "strata_order",
            ]
        )

    x_cats, x_edges = pd.cut(
        numeric_x[valid_mask],
        bins=x_bins,
        include_lowest=True,
        retbins=True,
        right=True,
    )
    strata_cats, strata_edges = pd.cut(
        numeric_strata[valid_mask],
        bins=strata_bins,
        include_lowest=True,
        retbins=True,
        right=True,
    )

    x_categories = x_cats.cat.categories
    strata_categories = strata_cats.cat.categories
    index = pd.MultiIndex.from_product([x_categories, strata_categories], names=["x_bin", "strata_bin"])

    grouped = (
        pd.DataFrame({"x_bin": x_cats, "strata_bin": strata_cats})
        .groupby(["x_bin", "strata_bin"], observed=False)
        .size()
    )
    dataset_counts = grouped.reindex(index, fill_value=0).reset_index(name="dataset_count")

    wrong_counts = pd.Series(0, index=index, dtype="int64")
    if detail_df is not None and not detail_df.empty:
        matched = detail_df[detail_df.get("matched") == True]  # noqa: E712
        x_detail_col = f"data_{x_column}"
        strata_detail_col = f"data_{strata_column}"
        if x_detail_col in matched.columns and strata_detail_col in matched.columns:
            x_detail = pd.to_numeric(matched[x_detail_col], errors="coerce")
            strata_detail = pd.to_numeric(matched[strata_detail_col], errors="coerce")
            detail_mask = x_detail.notna() & strata_detail.notna()
            if detail_mask.any():
                x_detail_bins = pd.cut(
                    x_detail[detail_mask],
                    bins=x_edges,
                    include_lowest=True,
                    right=True,
                )
                strata_detail_bins = pd.cut(
                    strata_detail[detail_mask],
                    bins=strata_edges,
                    include_lowest=True,
                    right=True,
                )
                detail_grouped = (
                    pd.DataFrame({"x_bin": x_detail_bins, "strata_bin": strata_detail_bins})
                    .groupby(["x_bin", "strata_bin"], observed=False)
                    .size()
                )
                wrong_counts = detail_grouped.reindex(index, fill_value=0).astype("int64")

    dataset_counts["x_bin"] = pd.Categorical(dataset_counts["x_bin"], categories=x_categories, ordered=True)
    dataset_counts["strata_bin"] = pd.Categorical(dataset_counts["strata_bin"], categories=strata_categories, ordered=True)
    dataset_counts["wrong_count"] = wrong_counts.values
    dataset_counts["accuracy_pct"] = dataset_counts.apply(
        lambda row: max(0.0, min(100.0, 100.0 - (row["wrong_count"] / row["dataset_count"]) * 100.0))
        if row["dataset_count"] > 0
        else None,
        axis=1,
    )
    dataset_counts["x_order"] = dataset_counts["x_bin"].cat.codes
    dataset_counts["strata_order"] = dataset_counts["strata_bin"].cat.codes
    dataset_counts["x_label"] = dataset_counts["x_bin"].apply(_interval_to_label)
    dataset_counts["strata_label"] = dataset_counts["strata_bin"].apply(_interval_to_label)

    return dataset_counts[
        [
            "x_label",
            "strata_label",
            "dataset_count",
            "wrong_count",
            "accuracy_pct",
            "x_order",
            "strata_order",
        ]
    ].reset_index(drop=True)


def prepare_grouped_box_data(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    bins: int,
) -> pd.DataFrame:
    """
    Prepare data for grouped box plots by binning one numeric column.
    """
    if group_column not in df.columns or value_column not in df.columns:
        raise ValueError("Selected columns must exist in the dataset.")

    group_numeric = pd.to_numeric(df[group_column], errors="coerce")
    value_numeric = pd.to_numeric(df[value_column], errors="coerce")
    valid_mask = group_numeric.notna() & value_numeric.notna()
    if not valid_mask.any():
        return pd.DataFrame(columns=["group_label", "value", "group_order"])

    group_bins = pd.cut(
        group_numeric[valid_mask],
        bins=bins,
        include_lowest=True,
        right=True,
    )
    categories = group_bins.cat.categories
    label_map = {interval: _interval_to_label(interval) for interval in categories}

    grouped_df = pd.DataFrame(
        {
            "group_label": pd.Categorical(
                [label_map[interval] for interval in group_bins],
                categories=[label_map[interval] for interval in categories],
                ordered=True,
            ),
            "value": value_numeric[valid_mask].astype(float).values,
        }
    )
    grouped_df["group_order"] = grouped_df["group_label"].cat.codes
    return grouped_df


def _interval_to_label(interval: pd.Interval) -> str:
    """Convert an interval to a readable ASCII range label."""
    left = interval.left
    right = interval.right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return f"{left:.3f} to {right:.3f}"
    return f"{left} to {right}"





