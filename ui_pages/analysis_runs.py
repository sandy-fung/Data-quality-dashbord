"""Error overview page for managing label runs."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

from core import datasets as ds_state
from core import plots
from core.labels import build_label_run_frames, parse_uploaded_label_files


def render_analysis_runs_page() -> bool:
    """Render error overview dashboard; return True when state changes."""
    st.title("\U0001f6a8 Error Overview")
    st.caption("Name a run, upload label files, and inspect affected records.")

    ds_state.ensure_dataset_state()
    ds_state.deduplicate_datasets()
    entries = list(ds_state.list_datasets().values())
    if not entries:
        st.info("Add a dataset on the Data Source page to manage label runs.")
        return False

    labels = ds_state.build_display_labels(entries)
    options = dict(zip(labels, [entry.id for entry in entries]))
    label_by_id = {entry.id: label for entry, label in zip(entries, labels)}

    active_entry = ds_state.get_active_dataset()
    active_label = label_by_id.get(active_entry.id) if active_entry else labels[0]
    default_index = labels.index(active_label) if active_label in labels else 0

    col_dataset, col_info = st.columns([2, 3])
    with col_dataset:
        selected_label = st.selectbox(
            "Dataset",
            options=labels,
            index=default_index,
            help="Choose which dataset to associate with runs.",
        )
    selected_id = options[selected_label]
    if active_entry is None or selected_id != active_entry.id:
        ds_state.set_active_dataset(selected_id)
        active_entry = ds_state.get_active_dataset()

    entry = active_entry
    entry = active_entry
    if entry is None or entry.dataset is None or entry.dataset.empty:
        st.info("Selected dataset is empty. Scan folders on Data Source to populate it.")
        return False

    with col_info:
        st.caption(f"Rows: {len(entry.dataset)} | Ground truth: {len(entry.ground_truth)} entries")
        st.caption(f"Created at: {entry.created_at}")

    label_runs = entry.label_runs or []
    entry.label_runs = label_runs

    prefix = f"label_run_{entry.id}_"
    nonce_key = f"{prefix}form_nonce"
    flash_key = f"{prefix}flash"
    dirty_key = f"{prefix}dirty"
    st.session_state.setdefault(nonce_key, 0)
    st.session_state.setdefault(flash_key, None)
    st.session_state.setdefault(dirty_key, False)

    flash_message = st.session_state.pop(flash_key, None)
    runs_changed = st.session_state.pop(dirty_key, False)
    if flash_message:
        st.success(flash_message)

    dataset: pd.DataFrame = entry.dataset
    dataset_total = len(dataset)

    runs_changed = _render_run_form(label_runs, nonce_key, flash_key, dirty_key) or runs_changed
    runs_changed = _render_run_management(label_runs, dirty_key) or runs_changed

    _, summary_df = build_label_run_frames(label_runs, dataset)

    if not summary_df.empty:
        st.subheader("Run summary")
        st.caption("Run totals compared to overall dataset size.")
        run_counts = summary_df["label_files"].astype(int).tolist()
        summary_view = summary_df.rename(
            columns={
                "label_files": "New Additions",
                "matched": "Matched Records",
                "unmatched": "Unmatched Records",
            }
        )
        st.dataframe(summary_view, use_container_width=True)

        fig = plots.run_accuracy_chart(
            labels=summary_df["run"].tolist(),
            run_counts=run_counts,
            dataset_total=dataset_total,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No label runs added yet.")

    if runs_changed:
        ds_state.update_dataset_entry(entry.id, label_runs=label_runs)

    return runs_changed


def _render_run_form(
    label_runs: List[Dict],
    nonce_key: str,
    flash_key: str,
    dirty_key: str,
) -> bool:
    """Render run creation form."""
    nonce = st.session_state[nonce_key]
    with st.form(f"add_label_run_form_{nonce}"):
        st.subheader("Add label run")
        st.caption("Upload all erroneous label files (.txt) for a single run.")
        run_name = st.text_input(
            "Run name",
            placeholder="run_YYYYMMDD_HHMMSS",
            help="Leave blank to auto-generate using the current timestamp.",
            key=f"label_run_name_{nonce}",
        )
        run_description = st.text_area(
            "Description (optional)",
            placeholder="Brief notes about this run...",
            height=68,
            key=f"label_run_description_{nonce}",
        )
        uploaded_files = st.file_uploader(
            "Labels (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            help="One .txt per erroneous filename. The file name must match the dataset filename.",
            key=f"label_run_uploader_{nonce}",
        )
        submitted = st.form_submit_button("Add run", type="primary", use_container_width=True)

    if not submitted:
        return False

    if not uploaded_files:
        st.warning("Please choose at least one label file before submitting.")
        return False

    final_name = run_name.strip() if run_name and run_name.strip() else _default_run_name()
    existing = {run["name"] for run in label_runs}
    if final_name in existing:
        st.error(f"Run '{final_name}' already exists. Pick a different name.")
        return False

    try:
        records = parse_uploaded_label_files(uploaded_files)
    except ValueError as exc:
        st.error(str(exc))
        return False

    label_runs.append(
        {
            "name": final_name,
            "description": run_description.strip() if run_description else "",
            "created_at": datetime.now().isoformat(),
            "labels": [record.__dict__ for record in records],
        }
    )
    st.session_state[nonce_key] += 1
    st.session_state[flash_key] = f"Added run '{final_name}' with {len(records)} label files."
    st.session_state[dirty_key] = True
    st.rerun()
    return True


def _render_run_management(label_runs: List[Dict], dirty_key: str) -> bool:
    """Provide controls to remove existing runs."""
    if not label_runs:
        return False

    changed = False
    with st.expander("Manage runs"):
        st.warning("Deleting a run cannot be undone.")
        run_names = [run["name"] for run in label_runs]
        run_to_delete = st.selectbox(
            "Select run",
            options=run_names,
            index=None,
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "Delete run",
                type="secondary",
                disabled=run_to_delete is None,
            ):
                label_runs[:] = [run for run in label_runs if run["name"] != run_to_delete]
                st.warning(f"Deleted run '{run_to_delete}'.")
                st.session_state[dirty_key] = True
                changed = True
        with col2:
            if st.button("Delete all runs", type="secondary"):
                label_runs.clear()
                st.warning("Deleted all runs.")
                st.session_state[dirty_key] = True
                changed = True
    return changed


def _default_run_name() -> str:
    """Generate default run name from current timestamp."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
