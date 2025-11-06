"""Text analysis page comparing YOLO ground truth and predictions."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

from core import datasets as ds_state
from core import text_analysis
from core.labels import build_label_run_frames

THUMBNAILS_PER_ROW = 5
THUMBNAIL_MAX_SIZE = 160


def _build_confusion_matrix(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Build confusion matrix from replacement pairs data.

    Args:
        summary_df: DataFrame with 'token' column containing "X>Y" patterns

    Returns:
        Tuple of (confusion matrix DataFrame, list of all unique characters)
    """
    # Parse tokens to get ground_truth -> prediction mappings
    matrix_data = {}
    all_chars = set()

    for _, row in summary_df.iterrows():
        token = row["token"]
        count = row["count"]

        if ">" in token:
            parts = token.split(">")
            if len(parts) == 2:
                ground_truth = parts[0].strip()
                prediction = parts[1].strip()
                all_chars.add(ground_truth)
                all_chars.add(prediction)

                if ground_truth not in matrix_data:
                    matrix_data[ground_truth] = {}
                matrix_data[ground_truth][prediction] = count

    # Sort characters for consistent display
    sorted_chars = sorted(all_chars)

    # Build matrix DataFrame
    matrix_rows = []
    for gt_char in sorted_chars:
        row = {}
        for pred_char in sorted_chars:
            row[pred_char] = matrix_data.get(gt_char, {}).get(pred_char, 0)
        matrix_rows.append(row)

    confusion_df = pd.DataFrame(matrix_rows, index=sorted_chars)

    return confusion_df, sorted_chars


def _sort_replacement_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort replacement tokens by first character group, then by count.

    This ensures same-color slices are adjacent in the pie chart.

    Args:
        df: DataFrame with 'token' and 'count' columns

    Returns:
        Sorted DataFrame
    """
    # Extract first character from token (e.g., "8>0" -> "8")
    df = df.copy()
    df["_first_char"] = df["token"].apply(
        lambda x: x.split(">")[0] if ">" in x else x
    )

    # Sort by first character, then by count within each group (descending)
    df = df.sort_values(
        ["_first_char", "count"],
        ascending=[True, False]
    ).drop(columns=["_first_char"]).reset_index(drop=True)

    return df


def _generate_replacement_color_map(tokens: List[str]) -> Dict[str, str]:
    """
    Generate color map for replacement pairs, grouping by first character.

    Each group (same first character) uses the same color.
    Colors progressively get lighter from first group to last group.

    Args:
        tokens: List of replacement pair strings (e.g., ["8>0", "8>1", "0>8"])
                Should already be sorted in the desired display order.

    Returns:
        Dictionary mapping each token to its color
    """
    # Start with a deep color
    base_color = "#1f4788"  # Deep blue

    # Group tokens by first character, preserving the input order
    first_char_groups: Dict[str, List[str]] = {}
    first_char_order: List[str] = []  # Track the order first characters appear

    for token in tokens:
        if ">" in token:
            first_char = token.split(">")[0]
            if first_char not in first_char_groups:
                first_char_groups[first_char] = []
                first_char_order.append(first_char)
            first_char_groups[first_char].append(token)

    # Assign progressively lighter shades to each group
    color_map = {}
    num_groups = len(first_char_order)

    for idx, first_char in enumerate(first_char_order):
        group_tokens = first_char_groups[first_char]

        # Calculate lightness factor: 0.0 for first group, 1.0 for last group
        if num_groups > 1:
            lightness_factor = idx / (num_groups - 1)
        else:
            lightness_factor = 0.0

        # Get the color for this group (progressively lighter)
        group_color = _lighten_color(base_color, lightness_factor)

        # All tokens in the same group get the same color
        for token in group_tokens:
            color_map[token] = group_color

    return color_map


def _lighten_color(hex_color: str, amount: float) -> str:
    """
    Lighten a color by increasing its brightness.

    Args:
        hex_color: Hex color string (e.g., "#1f4788")
        amount: Lightening amount (0.0 = original, 1.0 = much lighter)

    Returns:
        Lightened hex color string
    """
    import colorsys

    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB (0-1 range)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert to HSL for better lightness control
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Increase lightness progressively
    # Map amount (0.0 to 1.0) to lightness (current to 0.95)
    l = l + (0.95 - l) * amount

    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Convert to hex
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)

    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


def render_text_analysis_page() -> None:
    """Render text analysis visualizations."""
    st.title("\U0001f4dd Text Analysis")

    ds_state.ensure_dataset_state()
    ds_state.deduplicate_datasets()
    entries = list(ds_state.list_datasets().values())
    if not entries:
        st.info("Load a dataset on the Data Source page first.")
        return

    original_entry = ds_state.get_active_dataset()
    original_id = original_entry.id if original_entry else None

    entries_with_runs = []
    for entry in entries:
        runs = ds_state.get_dataset_label_runs(entry.id)
        entry.label_runs = runs
        if runs:
            entries_with_runs.append(entry)
    if not entries_with_runs:
        st.info("Upload predicted label runs for at least one dataset to analyze text results.")
        return

    labels = ds_state.build_display_labels(entries_with_runs)
    tabs = st.tabs(labels)
    rendered_any = False
    for tab, entry, label in zip(tabs, entries_with_runs, labels):
        with tab:
            ds_state.set_active_dataset(entry.id)
            st.caption(
                f"Ground truth: {len(entry.ground_truth)} | Label runs: {len(entry.label_runs)}"
            )
            rendered_any = _render_text_analysis_for_entry(entry, entry.id) or rendered_any

    if original_id and ds_state.get_dataset(original_id):
        ds_state.set_active_dataset(original_id)
    elif entries_with_runs:
        ds_state.set_active_dataset(entries_with_runs[-1].id)

    if not rendered_any:
        st.info("No text analysis available for the current datasets.")


def _render_text_analysis_for_entry(entry, key_suffix: str) -> bool:
    dataset: pd.DataFrame = entry.dataset
    ground_truth: Dict[str, str] = entry.ground_truth or {}
    label_runs: List[Dict] = entry.label_runs or []
    entry.label_runs = label_runs

    if not ground_truth:
        st.info("Configure a YOLO label directory in Data Source to enable text analysis.")
        return False
    if not label_runs:
        st.info("Upload predicted label runs on the Error Overview page to analyze text results.")
        return False

    detail_df, _ = build_label_run_frames(label_runs, dataset)
    if detail_df.empty:
        st.info("No label run details are available.")
        return False

    run_names = [run["name"] for run in label_runs]
    run_options = ["All runs", *run_names]
    selected_run = st.selectbox(
        "Label run", options=run_options, key=f"text_analysis_run_{key_suffix}"
    )
    if selected_run == "All runs":
        predictions_df = detail_df
    else:
        predictions_df = detail_df[detail_df["run"] == selected_run]

    try:
        analysis = text_analysis.analyze_text_predictions(predictions_df, ground_truth)
    except ImportError as exc:
        st.error(str(exc))
        return False
    comparisons_df: pd.DataFrame = analysis["comparisons"]  # type: ignore[assignment]
    type_summary_df: pd.DataFrame = analysis["type_summary"]  # type: ignore[assignment]
    detail_summary_df: pd.DataFrame = analysis["detail_summary"]  # type: ignore[assignment]
    metrics: Dict[str, float] = analysis["metrics"]  # type: ignore[assignment]
    debug_info: Dict = analysis.get("debug_info", {})  # type: ignore[assignment]

    if comparisons_df.empty:
        st.warning("⚠️ No overlapping records between predictions and ground truth for the selected run.")

        # Display debug information
        with st.expander("🐛 Debug Information", expanded=True):
            st.write(f"**Total predictions:** {debug_info.get('total_predictions', 0)}")
            st.write(f"**Total ground truth files:** {debug_info.get('total_ground_truth', 0)}")

            skipped_no_gt = debug_info.get('skipped_no_gt', [])
            skipped_empty = debug_info.get('skipped_empty_pred', [])
            gt_keys = debug_info.get('ground_truth_keys', [])

            if skipped_no_gt:
                st.write(f"**Files in predictions but NOT in ground truth:** {len(skipped_no_gt)}")
                with st.expander(f"Show {len(skipped_no_gt)} unmatched prediction files"):
                    st.code("\n".join(sorted(skipped_no_gt)[:50]))
                    if len(skipped_no_gt) > 50:
                        st.caption(f"... and {len(skipped_no_gt) - 50} more")

            if skipped_empty:
                st.write(f"**Files with empty predictions:** {len(skipped_empty)}")
                with st.expander(f"Show {len(skipped_empty)} empty prediction files"):
                    st.code("\n".join(sorted(skipped_empty)[:50]))
                    if len(skipped_empty) > 50:
                        st.caption(f"... and {len(skipped_empty) - 50} more")

            if gt_keys:
                st.write(f"**Sample ground truth filenames (first 20):**")
                st.code("\n".join(gt_keys))

            st.info("💡 **Tip:** Make sure the prediction filenames match the ground truth filenames (without extensions).")

        return False

    st.subheader("Overall metrics")
    col_total, col_wrong, col_plate, col_char, col_errors = st.columns(5)
    total_value = int(metrics.get("total", 0))
    col_total.metric("Total plates", f"{total_value:,}")

    wrong_value = int(metrics.get("wrong", 0))
    missing_value = int(metrics.get("missing", 0))
    col_wrong.metric("Wrong plates", f"{wrong_value:,}")

    plate_ratio = float(metrics.get("plate_accuracy", 0.0))
    plate_ratio = max(0.0, min(1.0, plate_ratio))
    plate_display = f"{plate_ratio * 100:.2f}%"
    col_plate.metric("Plate accuracy", plate_display)

    col_char.metric("Char accuracy (avg.)", f"{metrics['char_accuracy'] * 100:.2f}%")
    col_errors.metric("Total errors", f"{int(metrics['total_errors']):,}")

    st.subheader("Error type distribution")
    if type_summary_df.empty or type_summary_df["count"].sum() == 0:
        st.info("No character-level errors detected.")
    else:
        pie_fig = px.pie(
            type_summary_df,
            names="error_type",
            values="count",
            hole=0.35,
            color="error_type",
            color_discrete_map={
                "delete": "#ff7f0e",
                "insert": "#1f77b4",
                "replace": "#2ca02c",
            },
        )
        pie_fig.update_layout(legend_title="Error type")
        st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("Error type breakdown")
    rendered_any = False
    error_configs = {
        "delete": ("Deleted characters", "deleted_char"),
        "insert": ("Inserted characters", "inserted_char"),
        "replace": ("Replacement pairs", "replacement_pair"),
    }
    for error_type, (title, label_column) in error_configs.items():
        records_df = _extract_error_records(comparisons_df, error_type)
        if records_df.empty:
            continue
        summary_df = (
            records_df.groupby("token")
            .agg(
                count=("filename", "count"),
                filenames=("filename", lambda values: sorted(set(values))),
            )
            .reset_index()
        )

        # For replacement pairs, use confusion matrix heatmap
        if error_type == "replace":
            confusion_matrix, char_labels = _build_confusion_matrix(summary_df)

            # Create heatmap using plotly
            import plotly.graph_objects as go

            heatmap_fig = go.Figure(data=go.Heatmap(
                z=confusion_matrix.values,
                x=char_labels,
                y=char_labels,
                colorscale='Blues',
                text=confusion_matrix.values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='Ground Truth: %{y}<br>Prediction: %{x}<br>Count: %{z}<extra></extra>',
                colorbar=dict(title="Count")
            ))

            heatmap_fig.update_layout(
                title=title,
                xaxis_title="Predicted Character",
                yaxis_title="Ground Truth Character",
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'},  # GT on top
                width=600,
                height=600,
            )

            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            # For delete and insert, keep pie chart
            summary_df = summary_df.sort_values("count", ascending=False)
            pie_fig = px.pie(
                summary_df,
                names="token",
                values="count",
                title=title,
            )
            pie_fig.update_traces(
                textinfo="label",
                hovertemplate="%{label}: %{value}<extra></extra>",
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        table_df = summary_df.copy()
        table_df[label_column] = table_df.pop("token")
        display_df = table_df.assign(
            display_filenames=lambda df: df["filenames"].apply(
                lambda items: ", ".join(sorted({Path(item).name for item in items if item}))
            ),
        ).rename(columns={"count": "occurrences"})
        table_key = f"text_error_table_{key_suffix}_{error_type}"
        st.dataframe(
            display_df.drop(columns=["filenames"]).rename(columns={"display_filenames": "filenames"}),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            key=table_key,
        )

        selected_rows = (
            st.session_state.get(table_key, {})
            .get("selection", {})
            .get("rows", [])
        )
        if selected_rows:
            selected_files: List[str] = []
            for idx in selected_rows:
                selected_files.extend(table_df.iloc[idx]["filenames"])
            unique_files = sorted(set(selected_files))
            _render_error_thumbnails(unique_files, f"{title} thumbnails")

        rendered_any = True

    if not rendered_any:
        st.info("No character-level errors recorded.")

    st.subheader("Per-plate comparison")
    formatted = comparisons_df.assign(
        **{
            "char_error_rate (%)": (comparisons_df["char_error_rate"] * 100).round(2),
        }
    )
    st.dataframe(
        formatted[
            [
                "filename",
                "dataset_filename",
                "ground_truth",
                "prediction",
                "matching",
                "edit_distance",
                "delete_count",
                "insert_count",
                "replace_count",
                "char_error_rate (%)",
                "error_tokens",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    return True




def _extract_error_records(comparisons_df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    """Return a dataframe with character occurrences for the specified error type."""
    rows: List[Dict[str, str]] = []
    if comparisons_df.empty:
        return pd.DataFrame(columns=["token", "filename"])

    prefix_map = {
        "delete": "del:",
        "insert": "ins:",
        "replace": "rep:",
    }
    prefix = prefix_map.get(error_type)
    if prefix is None:
        return pd.DataFrame(columns=["token", "filename"])

    for _, row in comparisons_df.iterrows():
        tokens = str(row.get("error_tokens") or "")
        if not tokens:
            continue
        filename = str(row.get("filename") or "")
        for token in tokens.split(";"):
            token = token.strip()
            if token.startswith(prefix):
                value = token[len(prefix):].strip()
                if value:
                    dataset_name = str(row.get("dataset_filename") or filename)
                    rows.append({"token": value, "filename": dataset_name})
    if not rows:
        return pd.DataFrame(columns=["token", "filename"])
    return pd.DataFrame(rows)


def _render_error_thumbnails(filenames: List[str], title: str) -> None:
    """Display thumbnails for the given filenames if assets are configured."""
    image_root = st.session_state.get("image_root")
    if not image_root:
        st.info("Configure an image directory on the Data Source page to preview thumbnails.")
        return

    root_path = Path(image_root)
    if not root_path.exists() or not root_path.is_dir():
        st.warning("Configured image directory is invalid or not accessible.")
        return

    existing_paths: List[Path] = []
    missing: List[str] = []
    for name in filenames:
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = (root_path / name).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file():
            existing_paths.append(candidate)
        else:
            missing.append(name)

    if not existing_paths:
        st.warning("No matching image files were found for the selected rows.")
        if missing:
            st.caption(f"Missing examples: {', '.join(missing[:5])}")
        return

    with st.expander(f"{title} ({len(existing_paths)})", expanded=False):
        columns_per_row = THUMBNAILS_PER_ROW
        for start in range(0, len(existing_paths), columns_per_row):
            row_paths = existing_paths[start : start + columns_per_row]
            row_columns = st.columns(columns_per_row)
            for slot, path in zip(row_columns, row_paths):
                try:
                    thumbnail_bytes = _load_error_thumbnail(str(path), max_size=THUMBNAIL_MAX_SIZE)
                    slot.image(thumbnail_bytes, caption=path.name, width=THUMBNAIL_MAX_SIZE)
                except Exception as exc:  # pragma: no cover
                    slot.warning(f"Failed to load {path.name}: {exc}")

    if missing:
        st.caption(f"{len(missing)} files missing in directory (e.g., {', '.join(missing[:3])}).")


@st.cache_data(show_spinner=False)
def _load_error_image_bytes(path: str) -> bytes:
    """Read image bytes with caching."""
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def _load_error_thumbnail(path: str, max_size: int = THUMBNAIL_MAX_SIZE) -> bytes:
    """Return a thumbnail image as bytes."""
    data = _load_error_image_bytes(path)
    with Image.open(io.BytesIO(data)) as img:
        image = ImageOps.exif_transpose(img)
        image = image.convert("RGB")
        preview = image.copy()
        try:
            resample_filter = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            resample_filter = Image.LANCZOS  # type: ignore[attr-defined]
        preview.thumbnail((max_size, max_size), resample_filter)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return buffer.getvalue()
