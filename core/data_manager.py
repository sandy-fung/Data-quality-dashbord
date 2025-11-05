"""Session persistence manager for JSON quality dashboard."""

from __future__ import annotations

import gzip
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from core import datasets as ds_state


class DataManager:
    """Provides compressed persistence for Streamlit session data."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session_file = self.data_dir / "session_snapshot.json.gz"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.version = "1.0"

    def save_session(self, auto: bool = False) -> Tuple[bool, str]:
        """Save active session to disk."""
        try:
            payload = self._serialize_session(auto)
            self._write_snapshot(payload)
            st.session_state.last_save_time = datetime.now()
            size_kb = self.session_file.stat().st_size / 1024
            return True, f"Session saved ({size_kb:.1f} KB)"
        except Exception as exc:  # noqa: BLE001
            return False, f"Failed to save session: {exc}"

    def load_session(self) -> Tuple[bool, str]:
        """Load session data from snapshot."""
        if not self.session_file.exists():
            return False, "No saved session found"

        try:
            with gzip.open(self.session_file, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            return False, f"Failed to load session: {exc}"

        if payload.get("version") != self.version:
            payload = self._migrate(payload)

        self._restore_session(payload)
        active_entry = ds_state.get_active_dataset()
        dataset_rows = len(active_entry.dataset) if active_entry else 0
        run_count = len(active_entry.label_runs) if active_entry else 0
        dataset_name = active_entry.name if active_entry else "dataset"
        return True, f"Loaded {dataset_rows} rows and {run_count} label runs for '{dataset_name}'"

    def export_data(self) -> Optional[bytes]:
        """Return serialized snapshot for download."""
        try:
            payload = self._serialize_session(auto=False)
        except Exception:
            return None

        json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
        buffer = gzip.compress(json_bytes)
        return buffer

    def import_data(self, buffer: bytes) -> Tuple[bool, str]:
        """Import snapshot from uploaded bytes."""
        try:
            with gzip.open(io.BytesIO(buffer), "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except gzip.BadGzipFile:
            try:
                payload = json.loads(buffer.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                return False, f"Import failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Import failed: {exc}"

        # Backup current snapshot before overwriting.
        if self.session_file.exists():
            self._create_backup()

        try:
            self._restore_session(payload)
            self._write_snapshot(payload)
        except Exception as exc:  # noqa: BLE001
            return False, f"Import failed: {exc}"

        active_entry = ds_state.get_active_dataset()
        dataset_rows = len(active_entry.dataset) if active_entry else 0
        dataset_name = active_entry.name if active_entry else "dataset"
        return True, f"Imported dataset '{dataset_name}' with {dataset_rows} rows"

    def get_session_info(self) -> Dict[str, Any]:
        """Provide basic info about current snapshot."""
        if not self.session_file.exists():
            return {"exists": False}

        try:
            stats = self.session_file.stat()
            with gzip.open(self.session_file, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            datasets_payload = payload.get("datasets", [])
            total_rows = sum(len(item.get("dataset", [])) for item in datasets_payload)
            run_count = sum(len(item.get("label_runs", [])) for item in datasets_payload)
            return {
                "exists": True,
                "size_kb": stats.st_size / 1024,
                "version": payload.get("version"),
                "last_updated": payload.get("last_updated"),
                "dataset_rows": total_rows,
                "dataset_count": len(datasets_payload),
                "label_run_count": run_count,
            }
        except Exception:  # noqa: BLE001
            return {"exists": True, "error": "Unable to read snapshot metadata"}

    def clear_all(self) -> bool:
        """Remove snapshot and backups."""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            for backup in self.backup_dir.glob("*.json.gz"):
                backup.unlink()
            return True
        except Exception:
            return False

    # Internal helpers ---------------------------------------------------- #

    def _serialize_session(self, auto: bool) -> Dict[str, Any]:
        ds_state.ensure_dataset_state()
        datasets_payload = []
        for entry in ds_state.list_datasets().values():
            dataset_records = entry.dataset.to_dict(orient="records") if not entry.dataset.empty else []
            datasets_payload.append(
                {
                    "id": entry.id,
                    "name": entry.name,
                    "dataset": dataset_records,
                    "summary": entry.summary,
                    "image_dir": entry.image_dir,
                    "label_dir": entry.label_dir,
                    "ground_truth": entry.ground_truth,
                    "created_at": entry.created_at,
                    "label_runs": entry.label_runs,
                }
            )

        payload: Dict[str, Any] = {
            "version": self.version,
            "last_updated": datetime.now().isoformat(),
            "auto": auto,
            "datasets": datasets_payload,
            "active_dataset_id": st.session_state.get(ds_state.ACTIVE_DATASET_KEY),
            "settings": {
                "auto_save": st.session_state.get("auto_save", True),
            },
        }
        return payload

    def _restore_session(self, payload: Dict[str, Any]) -> None:
        ds_state.ensure_dataset_state()
        st.session_state[ds_state.DATASETS_KEY] = {}
        st.session_state[ds_state.ACTIVE_DATASET_KEY] = None
        st.session_state[ds_state.COUNTER_KEY] = 0

        datasets_payload = payload.get("datasets", [])
        for item in datasets_payload:
            records = item.get("dataset", [])
            df = pd.DataFrame(records)
            summary = item.get("summary") or {
                "columns": len(df.columns),
                "rows": len(df),
                "tables": {},
            }
            ds_state.create_dataset_entry(
                name=item.get("name", item.get("id", "")),
                dataset=df,
                summary=summary,
                image_dir=item.get("image_dir"),
                label_dir=item.get("label_dir"),
                ground_truth=item.get("ground_truth") or {},
                dataset_id=item.get("id"),
                created_at=item.get("created_at"),
                label_runs=item.get("label_runs") or [],
            )

        active_id = payload.get("active_dataset_id")
        ds_state.set_active_dataset(active_id)
        st.session_state.auto_save = payload.get("settings", {}).get("auto_save", True)

    def _write_snapshot(self, payload: Dict[str, Any]) -> None:
        if self.session_file.exists():
            self._create_backup()

        json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
        with gzip.open(self.session_file, "wb") as handle:
            handle.write(json_bytes)

    def _create_backup(self, keep: int = 3) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"session_{timestamp}.json.gz"
        if self.session_file.exists():
            backup_file.write_bytes(self.session_file.read_bytes())

        backups = sorted(self.backup_dir.glob("session_*.json.gz"))
        for old in backups[:-keep]:
            old.unlink()

    def _migrate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "datasets" not in payload:
            dataset_records = payload.get("dataset", [])
            summary = payload.get("dataset_summary") or {}
            if isinstance(summary, dict):
                summary_payload = {
                    "columns": summary.get("columns", 0),
                    "rows": summary.get("rows", len(dataset_records)),
                    "tables": summary.get("tables", {}),
                }
            else:
                summary_payload = {
                    "columns": 0,
                    "rows": len(dataset_records),
                    "tables": {},
                }
            dataset_entry = {
                "id": "ds_0001",
                "name": payload.get("dataset_path") or "Imported dataset",
                "dataset": dataset_records,
                "summary": summary_payload,
                "image_dir": None,
                "label_dir": None,
                "ground_truth": {},
                "created_at": payload.get("last_updated") or datetime.now().isoformat(),
                "label_runs": payload.get("label_runs", []),
            }
            payload = {
                "version": self.version,
                "last_updated": payload.get("last_updated"),
                "auto": payload.get("auto", False),
                "datasets": [dataset_entry],
                "active_dataset_id": dataset_entry["id"],
                "settings": payload.get("settings", {"auto_save": True}),
            }
        payload["version"] = self.version
        return payload


# Shared instance
data_manager = DataManager()
