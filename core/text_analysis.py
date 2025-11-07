"""Utilities for text-based error analysis using YOLO label files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz import distance as rf_distance
except ImportError:
    rf_distance = None  # type: ignore[assignment]

CLS_MAP: Dict[int, str] = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
}


@dataclass
class TextComparison:
    """Result of a ground-truth vs prediction comparison."""

    run: str
    filename: str
    dataset_filename: str
    base_name: str
    ground_truth: str
    prediction: str
    edit_distance: int
    char_length: int
    matching: bool
    delete_count: int
    insert_count: int
    replace_count: int
    error_tokens: List[str]


def load_label_directory(root: Path) -> Dict[str, str]:
    """Load YOLO label files into a mapping of filename -> text."""
    if not root.exists():
        raise ValueError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory, got: {root}")

    mapping: Dict[str, str] = {}
    files = list(root.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt label files found in {root}")

    for label_file in files:
        content = label_file.read_text(encoding="utf-8", errors="ignore")
        labels = parse_yolo_content(content)
        if not labels:
            continue
        mapping[label_file.stem] = labels_to_text(labels)
    return mapping


def parse_yolo_content(content: str) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """Parse YOLO label text into (cls, bbox) tuples."""
    labels: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if not content:
        return labels

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            bbox = tuple(float(x) for x in parts[1:5])
        except ValueError:
            continue
        labels.append((cls_id, bbox))
    return labels


def labels_to_text(labels: Iterable[Tuple[int, Tuple[float, float, float, float]]]) -> str:
    """Convert a sequence of YOLO labels into an ordered string."""
    sorted_labels = sorted(labels, key=lambda item: item[1][0])
    chars: List[str] = []
    for cls_id, _ in sorted_labels:
        chars.append(CLS_MAP.get(int(cls_id), "?"))
    return "".join(chars)


def parse_prediction_text(text: str) -> str:
    """Convert raw prediction label text into an ordered string."""
    return labels_to_text(parse_yolo_content(text))


def compare_texts(ground_truth: str, prediction: str) -> Tuple[int, List[Tuple[str, str, Optional[str]]]]:
    """Return edit distance and detailed errors between two strings."""
    if rf_distance is None:
        raise ImportError("rapidfuzz is required for text comparisons. Install with `uv pip install rapidfuzz`.")
    edit_distance = rf_distance.Levenshtein.distance(ground_truth, prediction)
    operations = rf_distance.Levenshtein.editops(ground_truth, prediction)
    details: List[Tuple[str, str, Optional[str]]] = []
    for op_type, src_pos, dest_pos in operations:
        if op_type == "delete" and src_pos < len(ground_truth):
            details.append(("delete", ground_truth[src_pos], None))
        elif op_type == "insert" and dest_pos < len(prediction):
            details.append(("insert", prediction[dest_pos], None))
        elif op_type == "replace":
            gt_char = ground_truth[src_pos] if src_pos < len(ground_truth) else ""
            pred_char = prediction[dest_pos] if dest_pos < len(prediction) else ""
            if gt_char or pred_char:
                details.append(("replace", gt_char, pred_char or None))
    return edit_distance, details


def analyze_text_predictions(
    predictions: pd.DataFrame,
    ground_truth_map: Dict[str, str],
) -> Dict[str, pd.DataFrame | Dict[str, float]]:
    """
    Compare predictions with ground-truth map and return analysis artefacts.

    Returns a dict containing:
        - comparisons: per-file comparison dataframe
        - type_summary: counts per error type
        - detail_summary: detailed counts per label
        - metrics: aggregate metrics (dict)
        - debug_info: debugging information (dict)
    """
    if rf_distance is None:
        raise ImportError("rapidfuzz is required for text analysis. Install with `uv pip install rapidfuzz`.")

    if predictions is None or predictions.empty:
        return {
            "comparisons": pd.DataFrame(),
            "type_summary": pd.DataFrame(),
            "detail_summary": pd.DataFrame(),
            "metrics": {
                "total": 0,
                "compared": 0,
                "matched": 0,
                "wrong": 0,
                "missing": 0,
                "plate_accuracy": 0.0,
                "char_accuracy": 0.0,
                "total_errors": 0,
            },
            "debug_info": {
                "total_predictions": 0,
                "total_ground_truth": 0,
                "skipped_no_gt": [],
                "skipped_empty_pred": [],
            },
        }

    comparisons: List[TextComparison] = []
    type_counter: Counter[str] = Counter()
    detail_counter: Dict[str, Counter[str]] = defaultdict(Counter)

    # Debug tracking
    skipped_no_gt: List[str] = []
    skipped_empty_pred: List[str] = []

    for _, row in predictions.iterrows():
        run_name = str(row.get("run") or "unknown")
        label_filename = str(row.get("filename") or "")
        dataset_filename = str(row.get("data_filename") or row.get("filename") or "")
        if not label_filename and not dataset_filename:
            continue
        base_name_source = dataset_filename or label_filename
        base_name = Path(base_name_source).stem
        gt_text = ground_truth_map.get(base_name)
        if not gt_text:
            skipped_no_gt.append(base_name)
            continue

        prediction_text = parse_prediction_text(str(row.get("label_text") or ""))
        if prediction_text == "":
            skipped_empty_pred.append(base_name)
            continue

        edit_distance, errors = compare_texts(gt_text, prediction_text)
        char_len = max(len(gt_text), 1)
        delete_count = insert_count = replace_count = 0
        token_details: List[str] = []
        for etype, first, second in errors:
            if etype == "delete":
                delete_count += 1
                type_counter["delete"] += 1
                detail_counter["delete"][first] += 1
                token_details.append(f"del:{first}")
            elif etype == "insert":
                insert_count += 1
                type_counter["insert"] += 1
                detail_counter["insert"][first] += 1
                token_details.append(f"ins:{first}")
            elif etype == "replace":
                replace_count += 1
                token = f"{first}->{second}"
                type_counter["replace"] += 1
                detail_counter["replace"][token] += 1
                token_details.append(f"rep:{token}")

        comparisons.append(
            TextComparison(
                run=run_name,
                filename=label_filename,
                dataset_filename=dataset_filename,
                base_name=base_name,
                ground_truth=gt_text,
                prediction=prediction_text,
                edit_distance=edit_distance,
                char_length=len(gt_text),
                matching=edit_distance == 0,
                delete_count=delete_count,
                insert_count=insert_count,
                replace_count=replace_count,
                error_tokens=token_details,
            )
        )

    if not comparisons:
        return {
            "comparisons": pd.DataFrame(),
            "type_summary": pd.DataFrame(),
            "detail_summary": pd.DataFrame(),
            "metrics": {
                "total": 0,
                "compared": 0,
                "matched": 0,
                "wrong": 0,
                "missing": 0,
                "plate_accuracy": 0.0,
                "char_accuracy": 0.0,
                "total_errors": 0,
            },
            "debug_info": {
                "total_predictions": len(predictions),
                "total_ground_truth": len(ground_truth_map),
                "skipped_no_gt": list(set(skipped_no_gt)),
                "skipped_empty_pred": list(set(skipped_empty_pred)),
                "ground_truth_keys": list(ground_truth_map.keys())[:20],  # Show first 20
            },
        }

    comparisons_df = pd.DataFrame(
        {
            "run": [item.run for item in comparisons],
            "filename": [item.filename for item in comparisons],
            "dataset_filename": [item.dataset_filename for item in comparisons],
            "base_name": [item.base_name for item in comparisons],
            "ground_truth": [item.ground_truth for item in comparisons],
            "prediction": [item.prediction for item in comparisons],
            "matching": [item.matching for item in comparisons],
            "edit_distance": [item.edit_distance for item in comparisons],
            "char_length": [item.char_length for item in comparisons],
            "delete_count": [item.delete_count for item in comparisons],
            "insert_count": [item.insert_count for item in comparisons],
            "replace_count": [item.replace_count for item in comparisons],
            "error_tokens": ["; ".join(item.error_tokens) for item in comparisons],
        }
    )
    comparisons_df["char_error_rate"] = comparisons_df["edit_distance"] / comparisons_df["char_length"].replace(0, 1)

    type_summary_df = (
        pd.DataFrame(
            [{"error_type": key, "count": value} for key, value in type_counter.items()],
        )
        if type_counter
        else pd.DataFrame(columns=["error_type", "count"])
    )

    detail_rows: List[Dict[str, str | int]] = []
    for error_type, counter in detail_counter.items():
        for token, count in counter.items():
            detail_rows.append({"error_type": error_type, "token": token, "count": count})
    detail_summary_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(columns=["error_type", "token", "count"])

    total_ground_truth = len(ground_truth_map)
    compared_unique = comparisons_df["base_name"].nunique()
    matched_unique = comparisons_df.loc[comparisons_df["matching"], "base_name"].nunique()
    wrong_unique = max(compared_unique - matched_unique, 0)
    total_errors = int(type_summary_df["count"].sum()) if not type_summary_df.empty else 0

    if total_ground_truth > 0:
        plate_accuracy_ratio = 1.0 - (wrong_unique / total_ground_truth)
        plate_accuracy_ratio = max(0.0, min(1.0, plate_accuracy_ratio))
    else:
        plate_accuracy_ratio = 0.0

    # Calculate character accuracy: 1 - (errors in run / total chars in dataset)
    # Use all ground truth chars as denominator, not just compared files
    total_dataset_characters = float(sum(len(text) for text in ground_truth_map.values()))
    wrong_characters = float(comparisons_df["edit_distance"].sum())
    if total_dataset_characters > 0:
        char_accuracy_ratio = 1.0 - (wrong_characters / total_dataset_characters)
        char_accuracy_ratio = max(0.0, min(1.0, char_accuracy_ratio))
    else:
        char_accuracy_ratio = 0.0

    metrics = {
        "total": total_ground_truth,
        "compared": compared_unique,
        "matched": matched_unique,
        "wrong": wrong_unique,
        "missing": max(total_ground_truth - compared_unique, 0),
        "plate_accuracy": plate_accuracy_ratio,
        "char_accuracy": char_accuracy_ratio,
        "total_errors": total_errors,
    }

    debug_info = {
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth_map),
        "skipped_no_gt": list(set(skipped_no_gt)),
        "skipped_empty_pred": list(set(skipped_empty_pred)),
        "ground_truth_keys": list(ground_truth_map.keys())[:20],  # Show first 20
    }

    return {
        "comparisons": comparisons_df,
        "type_summary": type_summary_df,
        "detail_summary": detail_summary_df,
        "metrics": metrics,
        "debug_info": debug_info,
    }


__all__ = [
    "CLS_MAP",
    "load_label_directory",
    "parse_yolo_content",
    "labels_to_text",
    "parse_prediction_text",
    "analyze_text_predictions",
]
