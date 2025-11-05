"""Helpers for managing multiple datasets in session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


DATASETS_KEY = "datasets"
ACTIVE_DATASET_KEY = "active_dataset_id"
COUNTER_KEY = "dataset_counter"


@dataclass
class DatasetEntry:
    id: str
    name: str
    dataset: pd.DataFrame
    summary: Dict[str, Any]
    image_dir: Optional[str]
    label_dir: Optional[str]
    ground_truth: Dict[str, str]
    created_at: str
    label_runs: List[Dict[str, Any]] = field(default_factory=list)


def ensure_dataset_state() -> None:
    """Ensure dataset-related keys exist in session state."""
    st.session_state.setdefault(DATASETS_KEY, {})
    st.session_state.setdefault(ACTIVE_DATASET_KEY, None)
    st.session_state.setdefault(COUNTER_KEY, 0)


def list_datasets() -> Dict[str, DatasetEntry]:
    """Return the dataset registry."""
    ensure_dataset_state()
    return st.session_state[DATASETS_KEY]


def get_dataset(dataset_id: str) -> Optional[DatasetEntry]:
    """Return a dataset entry by identifier."""
    return list_datasets().get(dataset_id)


def get_active_dataset() -> Optional[DatasetEntry]:
    """Return the currently active dataset entry."""
    ensure_dataset_state()
    dataset_id = st.session_state.get(ACTIVE_DATASET_KEY)
    if not dataset_id:
        return None
    return get_dataset(dataset_id)


def set_active_dataset(dataset_id: Optional[str]) -> None:
    """Set the active dataset and sync legacy keys."""
    ensure_dataset_state()
    datasets = list_datasets()
    if dataset_id is None or dataset_id not in datasets:
        st.session_state[ACTIVE_DATASET_KEY] = None
        st.session_state.pop("dataset", None)
        st.session_state.pop("dataset_summary", None)
        st.session_state.pop("dataset_path", None)
        st.session_state.pop("image_root", None)
        st.session_state.pop("label_root", None)
        st.session_state.pop("ground_truth_labels", None)
        return

    st.session_state[ACTIVE_DATASET_KEY] = dataset_id
    entry = datasets[dataset_id]
    st.session_state.dataset = entry.dataset
    st.session_state.dataset_summary = entry.summary
    st.session_state.dataset_path = entry.name
    st.session_state.image_root = entry.image_dir
    st.session_state.label_root = entry.label_dir
    st.session_state.ground_truth_labels = entry.ground_truth


def create_dataset_entry(
    *,
    name: str,
    dataset: pd.DataFrame,
    summary: Dict[str, Any],
    image_dir: Optional[str],
    label_dir: Optional[str],
    ground_truth: Dict[str, str],
    dataset_id: Optional[str] = None,
    created_at: Optional[str] = None,
    label_runs: Optional[List[Dict[str, Any]]] = None,
) -> DatasetEntry:
    """Create and register a dataset entry, returning the stored object."""
    ensure_dataset_state()
    if dataset_id is None:
        counter = st.session_state[COUNTER_KEY] + 1
        st.session_state[COUNTER_KEY] = counter
        dataset_id = f"ds_{counter:04d}"
    else:
        try:
            numeric = int(dataset_id.split("_")[-1])
            st.session_state[COUNTER_KEY] = max(st.session_state[COUNTER_KEY], numeric)
        except ValueError:
            pass
    entry = DatasetEntry(
        id=dataset_id,
        name=name or dataset_id,
        dataset=dataset,
        summary=summary,
        image_dir=image_dir,
        label_dir=label_dir,
        ground_truth=ground_truth,
        created_at=created_at or datetime.now().isoformat(),
        label_runs=label_runs or [],
    )

    # Initialize is_wrong column
    update_is_wrong_column(entry)

    st.session_state[DATASETS_KEY][dataset_id] = entry
    set_active_dataset(dataset_id)
    return entry


def update_dataset_entry(dataset_id: str, **updates: Any) -> None:
    """Update mutable fields on an existing dataset entry."""
    entry = get_dataset(dataset_id)
    if entry is None:
        return
    for key, value in updates.items():
        if hasattr(entry, key):
            setattr(entry, key, value)

    # Update is_wrong column when label_runs or dataset changes
    if "label_runs" in updates or "dataset" in updates:
        update_is_wrong_column(entry)

    if dataset_id == st.session_state.get(ACTIVE_DATASET_KEY):
        set_active_dataset(dataset_id)


def delete_dataset(dataset_id: str) -> None:
    """Remove a dataset entry and reset active selection if needed."""
    ensure_dataset_state()
    datasets = list_datasets()
    if dataset_id in datasets:
        datasets.pop(dataset_id)
    if st.session_state.get(ACTIVE_DATASET_KEY) == dataset_id:
        if datasets:
            new_active = next(iter(datasets))
            set_active_dataset(new_active)
        else:
            set_active_dataset(None)


def list_dataset_options() -> Dict[str, str]:
    """Return mapping of dataset display labels to identifiers."""
    entries = list(list_datasets().values())
    return {entry.name: entry.id for entry in entries}


def get_dataset_label_runs(dataset_id: Optional[str]) -> list:
    """Return label runs list for a dataset, creating if missing."""
    entry = get_dataset(dataset_id) if dataset_id else None
    if entry is None:
        return []
    if entry.label_runs is None:
        entry.label_runs = []
    return entry.label_runs


def build_display_labels(entries: List[DatasetEntry]) -> List[str]:
    """Return display labels for datasets using their names."""
    return [entry.name or "Dataset" for entry in entries]


def find_dataset_by_paths(image_dir: Optional[str], label_dir: Optional[str]) -> Optional[DatasetEntry]:
    """Return the first dataset entry matching the given directories."""
    for entry in list_datasets().values():
        if entry.image_dir == image_dir and entry.label_dir == label_dir:
            return entry
    return None


def deduplicate_datasets() -> None:
    """Remove duplicate datasets that point to the same directories."""
    datasets = list(list_datasets().values())
    seen: Dict[Tuple[Optional[str], Optional[str]], str] = {}
    # Iterate reverse to keep the most recently created entry
    for entry in reversed(datasets):
        key = (entry.image_dir, entry.label_dir)
        if key in seen:
            delete_dataset(entry.id)
        else:
            seen[key] = entry.id


def update_is_wrong_column(entry: DatasetEntry) -> None:
    """
    Update the 'is_wrong' column in the dataset based on label_runs.

    A row is marked as wrong (is_wrong=True) if its filename appears
    in any label_run, indicating it was flagged as an error.
    """
    if entry is None or entry.dataset is None or entry.dataset.empty:
        return

    dataset = entry.dataset

    # Collect all filenames from label_runs
    wrong_filenames = set()
    for run in entry.label_runs or []:
        for label_record in run.get("labels", []):
            # Store the basename (without extension) for matching
            filename = label_record.get("filename", "")
            if filename:
                wrong_filenames.add(filename)

    # Initialize or update is_wrong column
    if "filename" in dataset.columns:
        def is_filename_wrong(value) -> bool:
            """Check if a filename is marked as wrong in any label run."""
            if not isinstance(value, str) or not value:
                return False
            basename = Path(value).stem
            return basename in wrong_filenames

        dataset["is_wrong"] = dataset["filename"].apply(is_filename_wrong)
    else:
        # If no filename column, set all to False
        dataset["is_wrong"] = False
