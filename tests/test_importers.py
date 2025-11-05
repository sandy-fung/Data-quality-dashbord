"""Tests for core.importers module."""

from pathlib import Path
import json

import pandas as pd
import pytest
from PIL import Image

from core import importers


def test_resolve_input_path_returns_path(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.json"
    file_path.write_text("{}", encoding="utf-8")
    resolved = importers.resolve_input_path(str(file_path))
    assert resolved == file_path


def test_resolve_input_path_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(importers.ImportErrorInfo):
        importers.resolve_input_path(str(tmp_path / "missing.json"))


def test_load_json_records_ndjson(tmp_path: Path) -> None:
    file_path = tmp_path / "rows.json"
    file_path.write_text('{"a":1}\n{"a":2}', encoding="utf-8")
    records = importers.load_json_records(file_path, lines=True)
    assert len(records) == 2
    assert records[0]["a"] == 1


def test_build_dataframe_parses_datetime() -> None:
    records = [
        {"timestamp": "2024-01-01T00:00:00Z", "value": 10},
        {"timestamp": "2024-01-02T00:00:00Z", "value": 12},
    ]
    df = importers.build_dataframe(records)
    assert df['timestamp'].tolist() == ['2024-01-01T00:00:00Z', '2024-01-02T00:00:00Z']


def test_summarize_dataframe_numeric_and_categorical() -> None:
    records = [
        {"value": 1, "category": "A"},
        {"value": 3, "category": "A"},
        {"value": 5, "category": "B"},
    ]
    df = importers.build_dataframe(records)
    summary = importers.summarize_dataframe(df)

    assert "value" in summary
    assert summary["value"]["count"] == 3
    assert summary["value"]["max"] == 5.0

    assert "category" in summary
    assert "top_values" in summary["category"]
    assert summary["category"]["top_values"]["A"] == 2



def test_summarize_dataframe_handles_non_hashable() -> None:
    df = pd.DataFrame({"raw": [{"a": 1}, {"b": 2}]})
    summary = importers.summarize_dataframe(df)
    assert "examples" in summary["raw"]
    assert summary["raw"]["non_hashable_values"] == 2


def test_build_dataframe_flatten_features(tmp_path: Path) -> None:
    sample = [
        {
            "filename": "sample.jpg",
            "width": 100,
            "height": 50,
            "plate_text": "ABC123",
            "features": {"iou": 0.12, "brightness_mean": 120.5, "brightness_std": 12.3},
        }
    ]
    file_path = tmp_path / "sample.json"
    file_path.write_text(json.dumps(sample), encoding='utf-8')

    df, summary = importers.prepare_dataset(str(file_path))
    assert "IOU" in df.columns
    assert "Brightness" in df.columns
    assert "Brightness_Std" in df.columns
    assert "Width" in df.columns
    assert "Height" in df.columns
    assert summary["IOU"]["count"] == 1


def test_prepare_dataset_from_directories(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image_path = images_dir / "plate.jpg"
    Image.new("RGB", (100, 40), color="white").save(image_path)

    label_path = labels_dir / "plate.txt"
    label_path.write_text("0 0.4 0.5 0.2 0.4\n10 0.6 0.5 0.2 0.4\n", encoding="utf-8")

    df, summary = importers.prepare_dataset_from_directories(str(images_dir), str(labels_dir))

    assert len(df) == 1
    row = df.iloc[0]
    assert pytest.approx(row["Width"]) == 100.0
    assert pytest.approx(row["Height"]) == 40.0
    assert row["plate_text"] == "0A"
    assert "Brightness" in df.columns
    assert "Blurriness" in df.columns
    assert summary["Width"]["count"] == 1

