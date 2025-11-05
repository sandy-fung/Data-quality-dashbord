"""Data source page for managing datasets built from image and label folders."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from core import datasets as ds_state
from core import importers, text_analysis


def _format_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def render_data_source_page() -> bool:
    """Render the data source page and return True when state changes."""
    ds_state.ensure_dataset_state()
    ds_state.deduplicate_datasets()

    st.title("\U0001f4c1 Data Source")
    st.markdown(
        "Manage multiple datasets generated from plate images and YOLO label folders. "
        "Each dataset carries its own ground-truth mapping for downstream analysis."
    )

    changed = False
    changed = _render_scan_form() or changed
    changed = _render_dataset_overview() or changed
    changed = _render_active_summary() or changed
    return changed


def _render_dataset_overview() -> bool:
    """Display dataset registry and allow switching or deletion."""
    entries = list(ds_state.list_datasets().values())
    if not entries:
        st.info("No datasets loaded yet. Add a dataset below to get started.")
        return False

    labels = ds_state.build_display_labels(entries)
    options = dict(zip(labels, [entry.id for entry in entries]))
    label_by_id = {entry.id: label for entry, label in zip(entries, labels)}

    active_entry = ds_state.get_active_dataset()
    active_label = label_by_id.get(active_entry.id) if active_entry else labels[0]
    default_index = labels.index(active_label) if active_label in labels else 0

    selected_label = st.selectbox(
        "Active dataset",
        options=labels,
        index=default_index,
        key="dataset_selector",
    )
    selected_id = options[selected_label]
    ds_state.set_active_dataset(selected_id)
    entry = ds_state.get_active_dataset()

    changed = False
    with st.expander("Dataset details", expanded=False):
        st.caption(f"Created at: {entry.created_at}")
        st.caption(f"Image directory: {entry.image_dir or 'N/A'}")
        st.caption(f"Label directory: {entry.label_dir or 'N/A'}")
        st.caption(f"Ground-truth entries: {len(entry.ground_truth)}")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.session_state.setdefault(f"rename_input_{entry.id}", entry.name)
            new_name = st.text_input(
                "Display name",
                value=st.session_state[f"rename_input_{entry.id}"],
                key=f"dataset_name_input_{entry.id}",
            )
            if st.button("Save name", key=f"dataset_rename_btn_{entry.id}"):
                final_name = new_name.strip() or entry.name
                ds_state.update_dataset_entry(entry.id, name=final_name)
                st.session_state[f"rename_input_{entry.id}"] = final_name
                st.success(f"Renamed dataset to '{final_name}'.")
                changed = True
        with col2:
            if st.button("Delete dataset", type="secondary", key=f"dataset_delete_btn_{entry.id}"):
                ds_state.delete_dataset(entry.id)
                st.warning(f"Deleted dataset '{entry.name}'.")
                changed = True
                st.experimental_rerun()

    return changed





def _render_scan_form() -> bool:
    """Render the form used to add a new dataset."""
    st.subheader("Add dataset")
    last_image_dir = st.session_state.get("last_image_dir", "")
    last_label_dir = st.session_state.get("last_label_dir", "")

    with st.form("dataset_scan_form"):
        dataset_name = st.text_input(
            "Dataset name",
            placeholder="e.g., Factory_Batch_2024_11",
            key="dataset_scan_name",
        )
        image_dir = st.text_input(
            "Image directory",
            value=last_image_dir,
            placeholder="D:/dataset/picked/test/images",
            key="dataset_scan_images",
        )
        label_dir = st.text_input(
            "Label directory (optional)",
            value=last_label_dir,
            placeholder="D:/dataset/picked/test/labels",
            key="dataset_scan_labels",
        )
        submitted = st.form_submit_button("Scan folders", type="primary", use_container_width=True)

    if not submitted:
        return False

    image_dir = image_dir.strip()
    label_dir = label_dir.strip() or None

    try:
        df, summary = importers.prepare_dataset_from_directories(image_dir, label_dir)
    except importers.ImportErrorInfo as exc:
        st.error(str(exc))
        return False
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to scan directories: {exc}")
        return False

    ground_truth, resolved_label = _resolve_ground_truth(label_dir, df)
    resolved_image = _resolve_path(image_dir) if image_dir else None
    summary_payload = {
        "columns": len(df.columns),
        "rows": len(df),
        "tables": summary,
    }

    existing = ds_state.find_dataset_by_paths(resolved_image, resolved_label)
    if existing:
        final_name = dataset_name.strip() if dataset_name else existing.name
        ds_state.update_dataset_entry(
            existing.id,
            name=final_name,
            dataset=df,
            summary=summary_payload,
            image_dir=resolved_image,
            label_dir=resolved_label,
            ground_truth=ground_truth,
        )
        st.session_state[f"rename_input_{existing.id}"] = final_name
        st.success(
            f"Updated dataset '{final_name}' containing {len(df)} rows "
            f"and {len(ground_truth)} ground-truth mappings."
        )
        ds_state.set_active_dataset(existing.id)
        entry = ds_state.get_active_dataset()
        st.session_state["last_image_dir"] = image_dir
        if label_dir:
            st.session_state["last_label_dir"] = label_dir
        return True

    entry = ds_state.create_dataset_entry(
        name=dataset_name.strip() if dataset_name else Path(image_dir).name,
        dataset=df,
        summary=summary_payload,
        image_dir=resolved_image,
        label_dir=resolved_label,
        ground_truth=ground_truth,
    )
    st.session_state[f"rename_input_{entry.id}"] = entry.name

    st.session_state["last_image_dir"] = image_dir
    if label_dir:
        st.session_state["last_label_dir"] = label_dir

    st.success(
        f"Added dataset '{entry.name}' containing {len(df)} rows "
        f"and {len(entry.ground_truth)} ground-truth mappings."
    )
    return True


def _resolve_ground_truth(label_dir: str | None, df: pd.DataFrame) -> Tuple[Dict[str, str], str | None]:
    """Resolve ground-truth mapping using label directory and dataset fallback."""
    mapping: Dict[str, str] = {}
    resolved_label = _resolve_path(label_dir) if label_dir else None

    if resolved_label:
        try:
            mapping = text_analysis.load_label_directory(Path(resolved_label))
        except ValueError as exc:
            st.warning(f"Failed to load label directory: {exc}")
            mapping = {}

    if not mapping and "filename" in df.columns and "plate_text" in df.columns:
        fallback: Dict[str, str] = {}
        for filename, plate_text in df[["filename", "plate_text"]].itertuples(index=False):
            if not plate_text or not isinstance(plate_text, str):
                continue
            base_name = Path(str(filename)).stem
            if base_name:
                fallback[base_name] = plate_text
        mapping = fallback

    return mapping, resolved_label


def _resolve_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists() or not path.is_dir():
        return None
    return str(path.resolve())


def _render_active_summary() -> bool:
    """Render preview and column summary for the active dataset."""
    entry = ds_state.get_active_dataset()
    if entry is None:
        return False

    dataset = entry.dataset
    if dataset.empty:
        st.info("Active dataset is empty.")
        return False

    st.subheader("Preview")
    preview_df = dataset.drop(columns=["image_path", "label_path"], errors="ignore")
    st.dataframe(preview_df, use_container_width=True)

    st.subheader("Column summary")
    summary = entry.summary.get("tables", {})
    if not summary:
        st.warning("Summary unavailable; try rescanning the folders.")
        return False

    hidden_columns = {"image_path", "label_path"}
    records = []
    for column, meta in summary.items():
        if column in hidden_columns:
            continue
        base = {
            "column": column,
            "count": meta.get("count"),
            "missing": meta.get("missing"),
        }
        if "mean" in meta:
            base.update(
                {
                    "mean": _format_scalar(meta.get("mean")),
                    "std": _format_scalar(meta.get("std")),
                    "min": _format_scalar(meta.get("min")),
                    "max": _format_scalar(meta.get("max")),
                }
            )
        elif "top_values" in meta:
            base["top_values"] = ", ".join(
                f"{key}: {_format_scalar(value)}" for key, value in meta["top_values"].items()
            )
        elif "examples" in meta:
            base["examples"] = "; ".join(meta["examples"])
            base["non_hashable_values"] = meta.get("non_hashable_values")
        elif "min" in meta:
            base.update(
                {
                    "min": _format_scalar(meta.get("min")),
                    "max": _format_scalar(meta.get("max")),
                }
            )
        records.append(base)

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        numeric_cols = {"count", "missing", "non_hashable_values"}
        for column in summary_df.columns:
            if column in numeric_cols:
                summary_df[column] = (
                    pd.to_numeric(summary_df[column], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
            else:
                summary_df[column] = summary_df[column].fillna("").astype(str)
    st.dataframe(summary_df, use_container_width=True)
    return False
