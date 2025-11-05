"""Utilities for handling uploaded label runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import pandas as pd


@dataclass
class LabelRecord:
    """Simplified representation of a single label file."""

    filename: str
    source_name: str
    label_text: str


def parse_uploaded_label_files(files: Iterable) -> List[LabelRecord]:
    """
    Convert uploaded Streamlit files into label records.

    Raises ValueError when the payload cannot be decoded as UTF-8.
    """
    records: List[LabelRecord] = []
    for file in files:
        raw_bytes = file.getvalue()
        try:
            text = raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Failed to decode '{file.name}' as UTF-8: {exc}") from exc

        resolved_name = _resolve_label_basename(file.name)
        records.append(
            LabelRecord(
                filename=resolved_name,
                source_name=file.name,
                label_text=text,
            )
        )
    return records


def build_label_run_frames(
    label_runs: List[Dict],
    dataset: pd.DataFrame | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct detail and summary dataframes for label runs.

    Returns (detail_df, summary_df).
    """
    detail_rows: List[Dict] = []
    for run in label_runs:
        for record in run.get("labels", []):
            detail_rows.append(
                {
                    "run": run["name"],
                    "filename": record["filename"],
                    "label_text": record["label_text"],
                    "source_name": record["source_name"],
                    "description": run.get("description") or "",
                    "created_at": run.get("created_at"),
                }
            )

    if not detail_rows:
        empty_cols = [
            "run",
            "filename",
            "label_text",
            "source_name",
            "description",
            "created_at",
            "matched",
        ]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(
            columns=["run", "label_files", "matched", "unmatched"]
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df["matched"] = False

    if dataset is not None and not dataset.empty and "filename" in dataset.columns:
        dataset_with_keys = dataset.assign(
            _row_index=lambda df: df.index,
            _basename=lambda df: df["filename"].map(_basename_or_none),
        ).dropna(subset=["_basename"])

        dataset_lookup = dataset_with_keys.drop_duplicates(subset=["_basename"]).set_index("_basename")

        matched_mask = detail_df["filename"].isin(dataset_lookup.index)
        detail_df.loc[matched_mask, "matched"] = True

        merged = detail_df[matched_mask].join(
            dataset_lookup.add_prefix("data_"), on="filename", how="left"
        )
        unmatched = detail_df[~matched_mask]
        detail_df = pd.concat([merged, unmatched], ignore_index=True, sort=False)
    summary_df = (
        detail_df.groupby("run", dropna=False)
        .agg(
            label_files=("filename", "count"),
            matched=("matched", "sum"),
        )
        .reset_index()
    )
    summary_df["matched"] = summary_df["matched"].astype(int)
    summary_df["unmatched"] = summary_df["label_files"] - summary_df["matched"]
    summary_df.sort_values(by="matched", ascending=False, inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    return detail_df, summary_df


def _basename_or_none(value) -> str | None:
    """Return base name for comparisons, or None when unavailable."""
    if isinstance(value, str) and value:
        return Path(value).stem
    return None


def _resolve_label_basename(raw_name: str) -> str:
    """Strip extension from label file to match dataset basenames."""
    path = Path(raw_name)
    if path.suffix:
        return path.stem
    return str(path)
