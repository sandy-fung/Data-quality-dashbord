"""Tests for core.labels helpers."""

import pandas as pd
import pytest

from core import labels


class _FakeUpload:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    def getvalue(self) -> bytes:
        return self._text.encode("utf-8")


def test_parse_uploaded_label_files_strips_extension() -> None:
    uploads = [
        _FakeUpload("foo.jpg.txt", "label A"),
        _FakeUpload("bar.json", "label B"),
    ]

    records = labels.parse_uploaded_label_files(uploads)
    assert [record.filename for record in records] == ["foo.jpg", "bar"]
    assert [record.source_name for record in records] == ["foo.jpg.txt", "bar.json"]
    assert [record.label_text for record in records] == ["label A", "label B"]


def test_parse_uploaded_label_files_raises_on_decode_error() -> None:
    class _BadUpload:
        name = "bad.txt"

        @staticmethod
        def getvalue() -> bytes:
            return b"\xff\xfe"

    with pytest.raises(ValueError):
        labels.parse_uploaded_label_files([_BadUpload()])


def test_build_label_run_frames_matches_dataset() -> None:
    run_payload = {
        "name": "run_1",
        "description": "first test",
        "created_at": "2025-01-01T00:00:00",
        "labels": [
            {"filename": "a", "source_name": "a.jpg.txt", "label_text": "err"},
            {"filename": "b", "source_name": "b.jpg.txt", "label_text": "err"},
        ],
    }
    dataset = pd.DataFrame(
        [
            {"filename": "a.jpg", "value": 1},
            {"filename": "c.jpg", "value": 2},
        ]
    )

    detail_df, summary_df = labels.build_label_run_frames([run_payload], dataset)

    assert summary_df.loc[0, "label_files"] == 2
    assert summary_df.loc[0, "matched"] == 1
    assert summary_df.loc[0, "unmatched"] == 1
    assert detail_df["matched"].tolist() == [True, False]
    assert "data_value" in detail_df.columns
    assert detail_df.loc[0, "data_value"] == 1
