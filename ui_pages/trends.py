"""Trends and interactive visualization page."""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image, ImageOps

import pandas as pd
import streamlit as st
try:
    from streamlit_plotly_events import plotly_events
except ImportError:  # pragma: no cover
    plotly_events = None  # type: ignore[misc]

from core import datasets as ds_state
from core import metrics, plots, text_analysis
from core.labels import build_label_run_frames

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    RESAMPLE_FILTER = Image.LANCZOS  # type: ignore[attr-defined]

THUMBNAILS_PER_ROW = 5
THUMBNAIL_MAX_SIZE = 160




def render_trends_page() -> None:
    """Render trends page with interactive explorer and saved runs."""
    st.title("📈 Trends")
    ds_state.ensure_dataset_state()
    ds_state.deduplicate_datasets()
    entries = list(ds_state.list_datasets().values())
    if not entries:
        st.info("Add a dataset on the Data Source page to explore trends.")
        return

    labels = ds_state.build_display_labels(entries)
    tabs = st.tabs(labels)

    original_entry = ds_state.get_active_dataset()
    original_id = original_entry.id if original_entry else None

    for tab, entry, label in zip(tabs, entries, labels):
        with tab:
            ds_state.set_active_dataset(entry.id)
            st.caption(
                f"Rows: {len(entry.dataset)} | Ground truth: {len(entry.ground_truth)} entries"
            )
            _render_interactive_explorer(entry, entry.id)

    if original_id and ds_state.get_dataset(original_id):
        ds_state.set_active_dataset(original_id)

def _render_interactive_explorer(entry, key_suffix: str) -> None:
    st.subheader("Interactive explorer")

    dataset = entry.dataset
    label_runs = entry.label_runs or []
    entry.label_runs = label_runs
    detail_all_df: pd.DataFrame = pd.DataFrame()
    run_names: List[str] = []
    if label_runs:
        detail_all_df, _ = build_label_run_frames(label_runs, dataset)
        run_names = [run["name"] for run in label_runs]

    run_options: List[str] = ["All runs", *run_names] if run_names else ["All runs"]
    selected_run = _select_run_option(
        run_options,
        label="Label run",
        key=f"trend_active_run_{key_suffix}",
    )
    detail_selected = _filter_detail_by_run(detail_all_df, selected_run)
    active_run_label: Optional[str] = None if selected_run == "All runs" else selected_run
    dataset_for_run = _filter_dataset_by_run(dataset, detail_selected, selected_run)

    numeric_cols = metrics.detect_numeric_columns(dataset)

    st.markdown("#### Numeric Distribution")
    if not numeric_cols:
        st.warning("No numeric columns available.")
    else:
        # Restore previous selection or use first column
        prev_col_key = f"_prev_numeric_col_{key_suffix}"
        prev_col = st.session_state.get(prev_col_key)
        default_col_idx = 0
        if prev_col and prev_col in numeric_cols:
            default_col_idx = numeric_cols.index(prev_col)

        column = st.selectbox(
            "Numeric column",
            options=numeric_cols,
            index=default_col_idx,
            key=f"trend_numeric_col_{key_suffix}",
        )
        st.session_state[prev_col_key] = column

        # Restore previous bins value
        prev_bins_key = f"_prev_numeric_bins_{key_suffix}"
        default_bins = st.session_state.get(prev_bins_key, 40)

        bins = st.slider(
            "Bins",
            min_value=10,
            max_value=120,
            value=default_bins,
            key=f"trend_numeric_bins_{key_suffix}",
        )
        st.session_state[prev_bins_key] = bins
        hist_data = metrics.compute_numeric_distribution(dataset, column, bins=bins)
        dataset_counts = hist_data["count"]
        new_counts = [0] * len(dataset_counts)

        if not detail_selected.empty:
            matched = detail_selected[detail_selected["matched"] == True]  # noqa: E712
            value_column = f"data_{column}"
            filename_column = "filename" if "filename" in matched.columns else None
            if value_column in matched.columns:
                deduped = matched
                if filename_column:
                    deduped = matched.drop_duplicates(subset=[filename_column])
                values = pd.to_numeric(deduped[value_column], errors="coerce").dropna()
                if not values.empty:
                    bin_edges = [hist_data["bin_start"][0]] + hist_data["bin_end"]
                    categories = pd.cut(values, bins=bin_edges, include_lowest=True, right=True)
                    counts_series = categories.value_counts(sort=False)
                    new_counts = [int(value) for value in counts_series.tolist()]

        if active_run_label:
            wrong_label = active_run_label
        elif run_names:
            wrong_label = "All runs"
        else:
            wrong_label = "Flagged"

        overlay_fig = plots.numeric_accuracy_overlay(
            column=column,
            bin_start=hist_data["bin_start"],
            bin_end=hist_data["bin_end"],
            dataset_counts=dataset_counts,
            new_counts=new_counts,
            wrong_label=wrong_label,
        )
        bin_edges = _compute_bin_edges(hist_data)

        if plotly_events is not None:
            events = _capture_plotly_events(
                overlay_fig,
                chart_key=f"trend_numeric_plot_{column}_{key_suffix}",
            )
            st.caption("Click a bar to preview records within the selected numeric range.")
            selected_index = _resolve_selected_bin(
                column=column,
                bins=bins,
                events=events,
                bin_count=len(dataset_counts),
                dataset_id=key_suffix,
            )
        else:
            st.plotly_chart(overlay_fig, use_container_width=True, key=f"numeric_overlay_{key_suffix}")
            st.caption(
                "Select a range below to preview records. Install `streamlit-plotly-events` to enable bar clicks."
            )
            selected_index = _manual_bin_selector(
                column=column,
                bin_edges=bin_edges,
                dataset_id=key_suffix,
            )

        if selected_index is None:
            st.info("Choose a range to inspect matching records and thumbnails.")
        else:
            _render_numeric_bin_preview(
                dataset=dataset,
                column=column,
                bin_edges=bin_edges,
                bin_index=selected_index,
                detail_df=detail_selected if not detail_selected.empty else None,
                run_label=active_run_label,
                dataset_id=key_suffix,
                entry=entry,
            )

    st.markdown("#### Stratified Accuracy Grid")
    shared_controls_ready = len(numeric_cols) >= 2
    if not shared_controls_ready:
        st.warning("Need at least two numeric columns for stratified analysis.")
    else:
        # Column settings for Stratified Accuracy Grid
        # Restore X column selection
        prev_x_key = f"_prev_shared_x_col_{key_suffix}"
        prev_x = st.session_state.get(prev_x_key)
        default_x_idx = 0
        if prev_x and prev_x in numeric_cols:
            default_x_idx = numeric_cols.index(prev_x)

        x_column_shared = st.selectbox(
            "X column",
            options=numeric_cols,
            index=default_x_idx,
            key=f"trend_shared_x_col_{key_suffix}",
        )
        st.session_state[prev_x_key] = x_column_shared

        # Restore X bins value
        prev_x_bins_key = f"_prev_shared_x_bins_{key_suffix}"
        default_x_bins = st.session_state.get(prev_x_bins_key, 12)

        x_bins_shared = st.slider(
            "X bins",
            min_value=2,
            max_value=60,
            value=default_x_bins,
            key=f"trend_shared_x_bins_{key_suffix}",
        )
        st.session_state[prev_x_bins_key] = x_bins_shared

        strata_candidates = [col for col in numeric_cols if col != x_column_shared]
        if not strata_candidates:
            strata_candidates = numeric_cols

        # Restore strata column selection
        prev_strata_key = f"_prev_shared_strata_col_{key_suffix}"
        prev_strata = st.session_state.get(prev_strata_key)
        default_strata_idx = 0
        if prev_strata and prev_strata in strata_candidates:
            default_strata_idx = strata_candidates.index(prev_strata)

        strata_column_shared = st.selectbox(
            "Strata column",
            options=strata_candidates,
            index=default_strata_idx,
            key=f"trend_shared_strata_col_{key_suffix}",
        )
        st.session_state[prev_strata_key] = strata_column_shared

        # Restore strata bins value
        prev_strata_bins_key = f"_prev_shared_strata_bins_{key_suffix}"
        default_strata_bins = st.session_state.get(prev_strata_bins_key, 8)

        strata_bins_shared = st.slider(
            "Strata bins",
            min_value=2,
            max_value=40,
            value=default_strata_bins,
            key=f"trend_shared_strata_bins_{key_suffix}",
        )
        st.session_state[prev_strata_bins_key] = strata_bins_shared
        x_column = x_column_shared
        strata_column = strata_column_shared
        x_bins = x_bins_shared or 12
        strata_bins = strata_bins_shared or 8
        stats_df = metrics.compute_stratified_accuracy_grid(
            dataset,
            x_column=x_column,
            strata_column=strata_column,
            x_bins=int(x_bins),
            strata_bins=int(strata_bins),
            detail_df=detail_selected if not detail_selected.empty else None,
        )
        if stats_df.empty:
            st.info("No rows available after binning. Adjust your selections.")
        else:
            def format_count(value: float | None) -> str:
                if pd.isna(value):
                    return ""
                return f"{int(round(float(value)))}"

            def format_accuracy(value: float | None) -> str:
                if pd.isna(value):
                    return ""
                return f"{float(value):.1f}%"
            count_fig = plots.stratified_heatmap(
                stats_df=stats_df,
                value_column="dataset_count",
                title=f"Sample count by {x_column} and {strata_column}",
                colorbar_title="Count",
                colorscale="Blues",
                zmin=0,
                fill_value=0.0,
                text_formatter=format_count,
                x_title=f"{x_column} bins",
                y_title=f"{strata_column} bins",
                value_label=None,
                hover_columns=["accuracy_pct", "wrong_count"],
                hover_labels=["Accuracy (%)", "Matched"],
                hover_formats=["%{customdata:.1f}", "%{customdata:.0f}"],
            )
            accuracy_fig = plots.stratified_heatmap(
                stats_df=stats_df,
                value_column="accuracy_pct",
                title=f"Accuracy (%) by {x_column} and {strata_column}",
                colorbar_title="Accuracy (%)",
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                fill_value=None,
                text_formatter=format_accuracy,
                x_title=f"{x_column} bins",
                y_title=f"{strata_column} bins",
                value_label=None,
                hover_columns=["dataset_count", "wrong_count"],
                hover_labels=["Count", "Matched"],
                hover_formats=["%{customdata:.0f}", "%{customdata:.0f}"],
            )
            col_count, col_accuracy = st.columns(2)
            with col_count:
                st.plotly_chart(count_fig, use_container_width=True, key=f"stratified_count_{key_suffix}")
            with col_accuracy:
                st.plotly_chart(accuracy_fig, use_container_width=True, key=f"stratified_accuracy_{key_suffix}")

            stats_display = stats_df[
                ["x_label", "strata_label", "dataset_count", "wrong_count"]
            ].copy()
            stats_display.rename(
                columns={
                    "x_label": f"{x_column} bin",
                    "strata_label": f"{strata_column} bin",
                    "dataset_count": "Count",
                    "wrong_count": "Matched",
                },
                inplace=True,
            )
            stats_details = stats_df.reset_index(drop=True)
            stats_display = stats_display.reset_index(drop=True)
            table_key = f"stratified_accuracy_table_{key_suffix}_{x_column}_{strata_column}"
            st.dataframe(
                stats_display,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                key=table_key,
            )
            if active_run_label:
                st.caption(f"Matched counts reflect labels from run '{active_run_label}'.")
            selection_rows = (
                st.session_state.get(table_key, {})
                .get("selection", {})
                .get("rows", [])
            )
            if selection_rows:
                _render_stratified_selection_thumbnails(
                    dataset=dataset,
                    stats_details=stats_details,
                    selected_indices=selection_rows,
                    x_column=x_column,
                    strata_column=strata_column,
                    x_bins=int(x_bins),
                    strata_bins=int(strata_bins),
                    detail_df=detail_selected if not detail_selected.empty else None,
                    run_label=active_run_label,
                    dataset_id=key_suffix,
                    entry=entry,
                )
            else:
                st.caption("Select rows above to preview thumbnails for the chosen bin combinations.")

    # Correlation Heatmap moved to bottom of page
    st.markdown("#### Correlation Heatmap")
    if len(numeric_cols) < 2:
        message = (
            "Need at least two numeric columns for correlation analysis."
            if numeric_cols
            else "No numeric columns available for correlation analysis."
        )
        st.warning(message)
    else:
        # Restore previous selection
        prev_corr_key = f"_prev_corr_cols_{key_suffix}"
        prev_corr = st.session_state.get(prev_corr_key)
        if prev_corr:
            # Filter to only valid columns still in the dataset
            default_corr = [col for col in prev_corr if col in numeric_cols]
        else:
            default_corr = numeric_cols[: min(len(numeric_cols), 6)]

        selected = st.multiselect(
            "Numeric columns",
            options=numeric_cols,
            default=default_corr,
            key=f"trend_corr_cols_{key_suffix}",
        )
        st.session_state[prev_corr_key] = selected
        if len(selected) < 2:
            st.info("Select at least two columns to compute correlation.")
        else:
            matrix = metrics.compute_correlation_heatmap(dataset, columns=selected)
            if matrix.empty:
                st.info("Correlation matrix is empty for the selected columns.")
            else:
                fig = plots.correlation_heatmap(matrix)
                st.plotly_chart(fig, use_container_width=True, key=f"correlation_heatmap_{key_suffix}")


def _capture_plotly_events(fig, *, chart_key: str) -> List[Dict]:
    """Render a Plotly figure and capture click events via the optional dependency."""
    if plotly_events is None:
        return []

    return plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        use_container_width=True,
        key=chart_key,
    )


def _select_run_option(
    run_options: List[str],
    *,
    label: str,
    key: str,
) -> str:
    """Render a run selector that preserves the user's previous choice."""
    default_idx = 0
    stored_value = st.session_state.get(key)
    if isinstance(stored_value, str) and stored_value in run_options:
        default_idx = run_options.index(stored_value)
    return st.selectbox(label, options=run_options, index=default_idx, key=key)


def _filter_detail_by_run(detail_df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Return detail dataframe filtered by selected run."""
    if detail_df is None or detail_df.empty or run_name == "All runs":
        return detail_df.copy() if detail_df is not None and not detail_df.empty else pd.DataFrame()
    return detail_df[detail_df["run"] == run_name].copy()


def _filter_dataset_by_run(
    dataset: pd.DataFrame,
    detail_df: pd.DataFrame,
    run_name: str,
) -> pd.DataFrame:
    """Filter dataset rows to those referenced by the selected run when possible."""
    if (
        dataset is None
        or dataset.empty
        or detail_df is None
        or detail_df.empty
        or run_name == "All runs"
        or "data__row_index" not in detail_df.columns
    ):
        return dataset

    indices = (
        pd.to_numeric(detail_df["data__row_index"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if len(indices) == 0:
        return dataset.iloc[0:0]
    return dataset.loc[dataset.index.isin(indices)].copy()


def _compute_bin_edges(hist_data: Dict[str, List]) -> List[float]:
    """Return full list of bin edges from histogram data."""
    bin_start = hist_data.get("bin_start") or []
    bin_end = hist_data.get("bin_end") or []
    if not bin_start or not bin_end:
        return []
    first_edge = float(bin_start[0])
    trailing_edges = [float(edge) for edge in bin_end]
    return [first_edge, *trailing_edges]


def _resolve_selected_bin(
    *,
    column: str,
    bins: int,
    events: Optional[List[Dict]],
    bin_count: int,
    dataset_id: str,
) -> Optional[int]:
    """Keep track of the last clicked bin index for the numeric distribution."""
    state = st.session_state.setdefault("numeric_bin_selection", {})
    state_key = f"{dataset_id}|{column}|{bins}"
    selected_index: Optional[int] = state.get(state_key)

    if events:
        for point in events:
            curve_number = point.get("curveNumber")
            if curve_number not in (0, 1):
                continue
            point_index = point.get("pointIndex")
            if isinstance(point_index, int) and 0 <= point_index < bin_count:
                state[state_key] = point_index
                selected_index = point_index
                break

    return selected_index


def _manual_bin_selector(*, column: str, bin_edges: List[float], dataset_id: str) -> Optional[int]:
    """Fallback selector when interactive Plotly events are unavailable."""
    if not bin_edges or len(bin_edges) < 2:
        return None

    labels = [f"{bin_edges[idx]:.3f} to {bin_edges[idx + 1]:.3f}" for idx in range(len(bin_edges) - 1)]
    state = st.session_state.setdefault("numeric_bin_selection", {})
    state_key = f"{dataset_id}|{column}|manual"
    default_index = state.get(state_key, 0 if labels else None)

    selected_label = st.selectbox(
        "Select range",
        options=labels,
        index=default_index if default_index is not None and default_index < len(labels) else 0,
        key=f"manual_bin_selector_{dataset_id}_{column}",
    )
    selected_index = labels.index(selected_label)
    state[state_key] = selected_index
    return selected_index


def _render_numeric_bin_preview(
    *,
    dataset: pd.DataFrame,
    column: str,
    bin_edges: List[float],
    bin_index: int,
    detail_df: Optional[pd.DataFrame],
    run_label: Optional[str],
    dataset_id: str,
    entry,
) -> None:
    """Display dataset rows and thumbnails for the selected numeric bin."""
    if not bin_edges or bin_index >= len(bin_edges) - 1:
        st.info("Selected bin is out of range.")
        return

    start = bin_edges[bin_index]
    end = bin_edges[bin_index + 1]
    left_inclusive = bin_index == 0
    values = pd.to_numeric(dataset[column], errors="coerce")
    mask = values.notna()
    if left_inclusive:
        mask &= values >= start
    else:
        mask &= values > start
    mask &= values <= end

    bin_rows = dataset[mask].copy()
    range_label = f"{start:.3f} to {end:.3f}"
    st.markdown(f"##### Records for range {range_label}")

    if bin_rows.empty:
        st.info("No rows fall within the selected range.")
        return

    row_count = len(bin_rows)
    st.caption(f"{row_count} rows in this range.")
    range_token = range_label.replace(" ", "_")
    table_key = f"numeric_bin_records_{dataset_id}_{column}_{range_token}"
    display_df = bin_rows.reset_index(drop=True)
    with st.expander(f"Dataset records ({row_count})", expanded=False):
        clean_df = display_df.drop(columns=["image_path", "label_path"], errors="ignore")
        st.dataframe(clean_df, use_container_width=True, hide_index=True, on_select="rerun", key=table_key)

    flagged_count = 0
    if detail_df is not None and not detail_df.empty:
        matched = detail_df[detail_df.get("matched") == True]  # noqa: E712
        value_column = f"data_{column}"
        if value_column in matched.columns:
            matched_values = pd.to_numeric(matched[value_column], errors="coerce")
            detail_mask = matched_values.notna()
            if left_inclusive:
                detail_mask &= matched_values >= start
            else:
                detail_mask &= matched_values > start
            detail_mask &= matched_values <= end
            flagged_rows = matched[detail_mask].copy()

            # Add ground_truth and prediction columns
            ground_truth_map = entry.ground_truth if entry and entry.ground_truth else {}

            def get_ground_truth(filename):
                basename = Path(str(filename)).stem if filename else ""
                return ground_truth_map.get(basename, "")

            def get_prediction(label_text):
                if not label_text or pd.isna(label_text):
                    return ""
                try:
                    return text_analysis.parse_prediction_text(str(label_text))
                except Exception:
                    return ""

            flagged_rows["ground_truth"] = flagged_rows["filename"].apply(get_ground_truth)
            flagged_rows["prediction"] = flagged_rows["label_text"].apply(get_prediction)

            # Only keep run, filename, ground_truth, prediction
            label_columns = [
                col
                for col in ["run", "filename", "ground_truth", "prediction"]
                if col in flagged_rows.columns
            ]
            dedupe_columns = label_columns.copy()
            include_value = value_column in flagged_rows.columns
            if include_value:
                dedupe_columns.append(value_column)
            dataset_index_col = "data__row_index" if "data__row_index" in flagged_rows.columns else None
            if dataset_index_col:
                dedupe_columns.append(dataset_index_col)

            if dedupe_columns:
                flagged_preview = (
                    flagged_rows[dedupe_columns]
                    .drop_duplicates()
                )
            else:
                flagged_preview = pd.DataFrame(columns=["run", "filename"])

            if include_value:
                flagged_preview.rename(columns={value_column: column}, inplace=True)

            if dataset_index_col and dataset_index_col in flagged_preview.columns:
                flagged_preview = flagged_preview.sort_values(dataset_index_col)
                flagged_preview.index = flagged_preview[dataset_index_col].astype(int)
                flagged_preview.index.name = "dataset_index"
                flagged_preview.drop(columns=[dataset_index_col], inplace=True)
            else:
                flagged_preview = flagged_preview.reset_index(drop=True)

            flagged_count = len(flagged_preview)
            if flagged_count:
                caption_run = run_label or "All runs"
                st.caption(
                    f"{flagged_count} matched records from run '{caption_run}' fall within this range."
                )
                with st.expander(f"Matched records ({flagged_count})", expanded=False):
                    clean_flagged = flagged_preview.drop(columns=["image_path", "label_path"], errors="ignore")
                    st.dataframe(clean_flagged, use_container_width=True)
            else:
                if run_label:
                    st.caption(f"No matched records from run '{run_label}' fall within this range.")
                else:
                    st.caption("No matched label records fall within this range.")

    filename_column = _resolve_filename_column(display_df)
    if filename_column is None:
        return

    selection_rows = (
        st.session_state.get(table_key, {})
        .get("selection", {})
        .get("rows", [])
    )
    if not selection_rows:
        st.caption("Select rows in the table above to preview matching thumbnails.")
        return

    selected_filenames: List[str] = []
    for row_index in selection_rows:
        if 0 <= row_index < len(display_df):
            value = display_df.iloc[row_index][filename_column]
            if pd.notna(value):
                selected_filenames.append(str(value))

    if not selected_filenames:
        st.info("Selected rows do not include filename values.")
        return

    _render_image_gallery(
        filenames=selected_filenames,
        filename_column=filename_column,
        range_label=range_label,
    )


def _render_stratified_selection_thumbnails(
    *,
    dataset: pd.DataFrame,
    stats_details: pd.DataFrame,
    selected_indices: List[int],
    x_column: str,
    strata_column: str,
    x_bins: int,
    strata_bins: int,
    detail_df: Optional[pd.DataFrame],
    run_label: Optional[str],
    dataset_id: str,
    entry,
) -> None:
    """Display matched records table and thumbnails for selected stratified accuracy bins."""
    if not selected_indices:
        return

    filename_column = _resolve_filename_column(dataset)
    if filename_column is None:
        st.info("Dataset does not include a filename column for thumbnail previews.")
        return

    x_values = pd.to_numeric(dataset[x_column], errors="coerce")
    strata_values = pd.to_numeric(dataset[strata_column], errors="coerce")
    valid_mask = x_values.notna() & strata_values.notna()
    if not valid_mask.any():
        st.info("No numeric rows available to render stratified thumbnails.")
        return

    try:
        _, x_edges = pd.cut(
            x_values[valid_mask],
            bins=x_bins,
            include_lowest=True,
            retbins=True,
            right=True,
        )
        _, strata_edges = pd.cut(
            strata_values[valid_mask],
            bins=strata_bins,
            include_lowest=True,
            retbins=True,
            right=True,
        )
    except ValueError:
        st.info("Unable to compute stratified bin edges for thumbnail previews.")
        return

    normalized_indices = sorted({idx for idx in selected_indices if 0 <= idx < len(stats_details)})
    if not normalized_indices:
        st.info("Selected rows are outside the stratified table range.")
        return

    previews: Dict[str, List[str]] = {}
    messages: List[str] = []

    for index in normalized_indices:
        row = stats_details.iloc[index]
        x_order = int(row.get("x_order", 0))
        strata_order = int(row.get("strata_order", 0))
        if x_order >= len(x_edges) - 1 or strata_order >= len(strata_edges) - 1:
            messages.append("Selected bin edges could not be resolved.")
            continue

        x_start = float(x_edges[x_order])
        x_end = float(x_edges[x_order + 1])
        strata_start = float(strata_edges[strata_order])
        strata_end = float(strata_edges[strata_order + 1])

        mask = valid_mask.copy()
        if x_order == 0:
            mask &= x_values >= x_start
        else:
            mask &= x_values > x_start
        mask &= x_values <= x_end

        if strata_order == 0:
            mask &= strata_values >= strata_start
        else:
            mask &= strata_values > strata_start
        mask &= strata_values <= strata_end

        matched_rows = dataset[mask].copy()
        bin_label = f"{row['x_label']} | {row['strata_label']}"
        if matched_rows.empty:
            messages.append(f"No dataset rows fall within {bin_label}.")
            continue

        filenames = (
            matched_rows[filename_column]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not filenames:
            messages.append(f"No filename values available for {bin_label}.")
            continue

        previews[bin_label] = filenames

    # Display matched records table for the selected bins
    if detail_df is not None and not detail_df.empty and previews:
        matched = detail_df[detail_df.get("matched") == True]  # noqa: E712
        x_detail_col = f"data_{x_column}"
        strata_detail_col = f"data_{strata_column}"

        if x_detail_col in matched.columns and strata_detail_col in matched.columns:
            # Collect all matched records from selected bins
            all_flagged_rows = []

            for index in normalized_indices:
                row = stats_details.iloc[index]
                x_order = int(row.get("x_order", 0))
                strata_order = int(row.get("strata_order", 0))

                if x_order >= len(x_edges) - 1 or strata_order >= len(strata_edges) - 1:
                    continue

                x_start = float(x_edges[x_order])
                x_end = float(x_edges[x_order + 1])
                strata_start = float(strata_edges[strata_order])
                strata_end = float(strata_edges[strata_order + 1])

                x_detail = pd.to_numeric(matched[x_detail_col], errors="coerce")
                strata_detail = pd.to_numeric(matched[strata_detail_col], errors="coerce")

                detail_mask = x_detail.notna() & strata_detail.notna()

                if x_order == 0:
                    detail_mask &= x_detail >= x_start
                else:
                    detail_mask &= x_detail > x_start
                detail_mask &= x_detail <= x_end

                if strata_order == 0:
                    detail_mask &= strata_detail >= strata_start
                else:
                    detail_mask &= strata_detail > strata_start
                detail_mask &= strata_detail <= strata_end

                flagged_rows = matched[detail_mask].copy()
                all_flagged_rows.append(flagged_rows)

            if all_flagged_rows:
                combined_flagged = pd.concat(all_flagged_rows, ignore_index=True)

                # Add ground_truth and prediction columns
                ground_truth_map = entry.ground_truth if entry and entry.ground_truth else {}

                def get_ground_truth(filename):
                    basename = Path(str(filename)).stem if filename else ""
                    return ground_truth_map.get(basename, "")

                def get_prediction(label_text):
                    if not label_text or pd.isna(label_text):
                        return ""
                    try:
                        return text_analysis.parse_prediction_text(str(label_text))
                    except Exception:
                        return ""

                combined_flagged["ground_truth"] = combined_flagged["filename"].apply(get_ground_truth)
                combined_flagged["prediction"] = combined_flagged["label_text"].apply(get_prediction)

                # Only keep run, filename, ground_truth, prediction
                label_columns = [
                    col
                    for col in ["run", "filename", "ground_truth", "prediction"]
                    if col in combined_flagged.columns
                ]

                if label_columns:
                    flagged_preview = combined_flagged[label_columns].drop_duplicates().reset_index(drop=True)

                    flagged_count = len(flagged_preview)
                    if flagged_count:
                        caption_run = run_label or "All runs"
                        st.caption(
                            f"{flagged_count} matched records from run '{caption_run}' in selected bins."
                        )
                        matched_table_key = f"stratified_matched_records_{dataset_id}_{x_column}_{strata_column}"
                        with st.expander(f"Matched records ({flagged_count})", expanded=False):
                            st.dataframe(
                                flagged_preview,
                                use_container_width=True,
                                hide_index=True,
                                on_select="rerun",
                                key=matched_table_key,
                            )

                        # Check if user selected rows from matched records table
                        matched_selection = (
                            st.session_state.get(matched_table_key, {})
                            .get("selection", {})
                            .get("rows", [])
                        )

                        if matched_selection:
                            # Get filenames and match with dataset rows to get correct filename format
                            selected_filenames: List[str] = []
                            dataset_filename_col = _resolve_filename_column(dataset)

                            if dataset_filename_col:
                                # Build a lookup dict for faster matching: basename -> full filename
                                basename_to_fullname = {}
                                for ds_fname in dataset[dataset_filename_col].dropna():
                                    basename = Path(str(ds_fname)).stem
                                    basename_to_fullname[basename] = str(ds_fname)

                                # Match selected rows
                                for row_idx in matched_selection:
                                    if 0 <= row_idx < len(flagged_preview):
                                        matched_fname = flagged_preview.iloc[row_idx].get("filename")
                                        if pd.notna(matched_fname):
                                            matched_basename = Path(str(matched_fname)).stem
                                            if matched_basename in basename_to_fullname:
                                                selected_filenames.append(basename_to_fullname[matched_basename])

                            if selected_filenames:
                                _render_image_gallery(
                                    filenames=selected_filenames,
                                    filename_column=dataset_filename_col or "filename",
                                    range_label="Selected matched records",
                                )
                            else:
                                st.caption("No matching dataset files found for selected records.")
                        else:
                            st.caption("Select rows in Matched records table to preview thumbnails.")
                    else:
                        if run_label:
                            st.caption(f"No matched records from run '{run_label}' in selected bins.")
                        else:
                            st.caption("No matched records in selected bins.")
    elif messages:
        for message in messages[:3]:
            st.info(message)


def _resolve_filename_column(df: pd.DataFrame) -> Optional[str]:
    """Return dataset column name matching 'filename' (case-insensitive)."""
    for column in df.columns:
        if column.lower() == "filename":
            return column
    return None


def _render_image_gallery(
    *,
    filenames: List[str],
    filename_column: str,
    range_label: str,
) -> None:
    """Render image thumbnails for the provided dataset records."""
    if not filenames:
        return

    image_root = st.session_state.get("image_root")
    if not image_root:
        return

    root_path = Path(image_root)
    if not root_path.exists() or not root_path.is_dir():
        st.warning("Configured image directory is invalid or not accessible.")
        return

    unique_filenames = list(dict.fromkeys(filenames))
    existing_paths: Dict[str, Path] = {}
    missing: List[str] = []
    for name in unique_filenames:
        candidate = (root_path / name).resolve()
        if candidate.exists():
            existing_paths[name] = candidate
        else:
            missing.append(name)

    if not existing_paths:
        st.warning("No matching image files were found in the configured directory.")
        if missing:
            st.caption(f"Missing examples: {', '.join(missing[:5])}")
        return

    thumb_count = len(existing_paths)
    expander_label = f"Thumbnails ({thumb_count}) - {range_label}"
    with st.expander(expander_label, expanded=False):
        names_order = list(existing_paths.keys())
        selection_state = st.session_state.setdefault("image_preview_selection", {})
        selection_key = f"{filename_column}|{range_label}"
        if selection_key not in selection_state and names_order:
            selection_state[selection_key] = names_order[0]

        columns_per_row = THUMBNAILS_PER_ROW
        for start in range(0, len(names_order), columns_per_row):
            row_names = names_order[start : start + columns_per_row]
            row_columns = st.columns(columns_per_row)
            for slot, name in zip(row_columns, row_names):
                path = existing_paths[name]
                try:
                    thumbnail_bytes = _load_thumbnail_bytes(str(path), max_size=THUMBNAIL_MAX_SIZE)
                    slot.image(thumbnail_bytes, caption=name, width=THUMBNAIL_MAX_SIZE)
                    if slot.button(
                        "Full size",
                        key=f"full_image_btn|{selection_key}|{name}",
                    ):
                        selection_state[selection_key] = name
                except Exception as exc:  # pragma: no cover
                    slot.warning(f"Failed to load {name}: {exc}")

        selected_name = selection_state.get(selection_key)
        selected_path = existing_paths.get(selected_name) if selected_name else None
        if selected_path is not None:
            try:
                full_image = _load_full_image(selected_path)
                st.image(full_image, caption=f"Full resolution: {selected_path.name}", use_container_width=False)
                mime_type, _ = mimetypes.guess_type(str(selected_path))
                st.download_button(
                    label="Download image",
                    data=_load_image_bytes(str(selected_path)),
                    file_name=selected_path.name,
                    mime=mime_type or "application/octet-stream",
                    key=f"download_image|{selection_key}",
                )
            except Exception as exc:  # pragma: no cover
                st.warning(f"Failed to load full resolution image: {exc}")

    if missing:
        st.caption(f"{len(missing)} files missing in directory (e.g., {', '.join(missing[:3])}).")


@st.cache_data(show_spinner=False)
def _load_image_bytes(path: str) -> bytes:
    """Read image file bytes with caching."""
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def _load_thumbnail_bytes(path: str, max_size: int = THUMBNAIL_MAX_SIZE) -> bytes:
    """Generate a high-quality thumbnail for faster preview."""
    data = _load_image_bytes(path)
    with Image.open(io.BytesIO(data)) as img:
        image = ImageOps.exif_transpose(img)
        image = image.convert("RGB")
        preview = image.copy()
        preview.thumbnail((max_size, max_size), RESAMPLE_FILTER)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return buffer.getvalue()


def _load_full_image(path: Path) -> Image.Image:
    """Load the original image while respecting EXIF orientation."""
    data = _load_image_bytes(str(path))
    with Image.open(io.BytesIO(data)) as img:
        image = ImageOps.exif_transpose(img)
        return image.convert("RGB")
