"""Data import utilities for the JSON quality dashboard."""

from __future__ import annotations

import json
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from PIL import Image


class ImportErrorInfo(Exception):
    """Raised when data import fails."""


def resolve_input_path(raw_path: str) -> Path:
    """Resolve user-supplied path string."""
    if not raw_path:
        raise ImportErrorInfo("Empty path provided")

    normalized = raw_path.strip()

    path = Path(normalized).expanduser()
    if not path.exists():
        raise ImportErrorInfo(f"Path does not exist: {path}")
    if path.is_dir():
        raise ImportErrorInfo(f"Expected a file path, got directory: {path}")
    return path


def load_json_records(
    file_path: Path,
    lines: bool = True,
    encoding: str = "utf-8",
) -> List[Dict]:
    """
    Load JSON records from file.

    Supports standard array JSON and newline-delimited JSON (default).
    """
    try:
        if lines:
            return _load_ndjson(file_path, encoding=encoding)
        return _load_standard_json(file_path, encoding=encoding)
    except (json.JSONDecodeError, ImportErrorInfo) as exc:
        if lines:
            return _load_standard_json(file_path, encoding=encoding)
        raise ImportErrorInfo(f"Invalid JSON format: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ImportErrorInfo(f"Encoding issue while reading file: {exc}") from exc


def _load_ndjson(file_path: Path, encoding: str) -> List[Dict]:
    records: List[Dict] = []
    with file_path.open("r", encoding=encoding) as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except (json.JSONDecodeError, ImportErrorInfo) as exc:
                raise ImportErrorInfo(f"Line {idx} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ImportErrorInfo(f"Line {idx} does not contain an object")
            records.append(record)
    return records


def _load_standard_json(file_path: Path, encoding: str) -> List[Dict]:
    with file_path.open("r", encoding=encoding) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [obj for obj in payload if isinstance(obj, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ImportErrorInfo("Unsupported JSON structure; expected list or object")


def build_dataframe(records: List[Dict]) -> pd.DataFrame:
    """Convert record list into a pandas DataFrame with flattened columns."""
    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)
    rename_map = {
        "features.blurriness": "Blurriness",
        "features.brightness_mean": "Brightness",
        "features.brightness_std": "Brightness_Std",
        "features.tilt_abs_deg": "Tilt",
        "features.iou": "IOU",
        "width": "Width",
        "height": "Height",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df


def _maybe_parse_datetime(series: pd.Series) -> pd.Series:
    return series


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Dict]:
    """Produce column-wise summaries for display."""
    summary: Dict[str, Dict] = {}
    if df.empty:
        return summary

    for column in df.columns:
        series = df[column]
        meta: Dict[str, Optional[float]] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(series):
            meta.update(
                {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            meta.update(
                {
                    "min": series.min().isoformat() if series.notna().any() else None,
                    "max": series.max().isoformat() if series.notna().any() else None,
                }
            )
        else:
            hashable_mask = series.dropna().map(lambda value: isinstance(value, Hashable))
            if hashable_mask.all():
                top_values = series.value_counts(dropna=True).head(5)
                meta["top_values"] = top_values.to_dict()
            else:
                sample = series.dropna().head(3)
                meta["examples"] = [str(value) for value in sample]
                meta["non_hashable_values"] = int((~hashable_mask).sum())

        summary[column] = meta
    return summary


def prepare_dataset(raw_path: str) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Load and summarize JSON dataset from path string."""
    path = resolve_input_path(raw_path)
    records = load_json_records(path)
    df = build_dataframe(records)
    summary = summarize_dataframe(df)
    return df, summary


YOLO_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def prepare_dataset_from_directories(
    image_dir: str,
    label_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Build a dataset by scanning image and label directories.

    Args:
        image_dir: Directory containing plate images (recursively scanned).
        label_dir: Directory containing YOLO-format label files (optional).
    """
    image_root = Path(image_dir).expanduser()
    if not image_root.exists() or not image_root.is_dir():
        raise ImportErrorInfo(f"Image directory is invalid: {image_dir}")

    label_root: Optional[Path] = None
    if label_dir:
        candidate = Path(label_dir).expanduser()
        if not candidate.exists():
            raise ImportErrorInfo(f"Label directory does not exist: {label_dir}")
        if not candidate.is_dir():
            raise ImportErrorInfo(f"Label path is not a directory: {label_dir}")
        label_root = candidate

    records = _collect_image_label_records(image_root, label_root)
    if not records:
        raise ImportErrorInfo("No image files were found in the provided directory.")

    df = build_dataframe(records)
    summary = summarize_dataframe(df)
    return df, summary


def _collect_image_label_records(image_root: Path, label_root: Optional[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_name = _relative_key(path, image_root)
        metadata = _extract_image_features(path)
        label_path = _resolve_label_path(path, image_root, label_root)
        label_data = _extract_label_features(label_path)
        record: Dict[str, Any] = {
            "filename": relative_name,
            "image_path": str(path.resolve()),
            "label_path": str(label_path.resolve()) if label_path else None,
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "plate_text": label_data.get("plate_text"),
            "features": {
                "tilt_abs_deg": label_data.get("tilt_abs_deg"),
                "iou": label_data.get("mean_iou"),
                "brightness_mean": metadata.get("brightness_mean"),
                "brightness_std": metadata.get("brightness_std"),
                "blurriness": metadata.get("blurriness"),
            },
        }
        records.append(record)
    return records


def _relative_key(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _extract_image_features(path: Path) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "width": None,
        "height": None,
        "brightness_mean": None,
        "brightness_std": None,
        "blurriness": None,
    }
    try:
        with Image.open(path) as image:
            width, height = image.size
            gray = image.convert("L")
            resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS  # type: ignore[attr-defined]
            gray_small = gray.resize((256, 256), resample=resample)
            gray_arr = np.asarray(gray_small, dtype=np.float32)
            result["width"] = float(width)
            result["height"] = float(height)
            if gray_arr.size > 0:
                result["brightness_mean"] = float(gray_arr.mean())
                result["brightness_std"] = float(gray_arr.std())
                result["blurriness"] = _compute_blurriness(gray_arr)
    except Exception:  # pragma: no cover - best effort
        return result
    return result


def _compute_blurriness(gray_arr: np.ndarray) -> Optional[float]:
    if gray_arr.size == 0:
        return None
    if cv2 is not None:  # pragma: no cover
        lap = cv2.Laplacian(gray_arr.astype(np.float32), cv2.CV_64F)
        return float(lap.var()) if lap.size else None
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(gray_arr, 1, mode="edge")
    laplacian = (
        kernel[0, 0] * padded[:-2, :-2]
        + kernel[0, 1] * padded[:-2, 1:-1]
        + kernel[0, 2] * padded[:-2, 2:]
        + kernel[1, 0] * padded[1:-1, :-2]
        + kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[1, 2] * padded[1:-1, 2:]
        + kernel[2, 0] * padded[2:, :-2]
        + kernel[2, 1] * padded[2:, 1:-1]
        + kernel[2, 2] * padded[2:, 2:]
    )
    return float(laplacian.var()) if laplacian.size else None


def _resolve_label_path(path: Path, image_root: Path, label_root: Optional[Path]) -> Optional[Path]:
    if label_root is None:
        return None
    candidates: List[Path] = []
    try:
        relative = path.relative_to(image_root)
        candidates.append(label_root / relative.with_suffix(".txt"))
    except ValueError:
        pass
    candidates.append(label_root / f"{path.stem}.txt")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _extract_label_features(label_path: Optional[Path]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {
        "plate_text": None,
        "tilt_abs_deg": None,
        "mean_iou": None,
    }
    labels = _read_yolo_labels(label_path)
    if not labels:
        return result
    result["plate_text"] = _decode_plate_text(labels)
    result["tilt_abs_deg"] = _compute_tilt(labels)
    result["mean_iou"] = _compute_mean_iou(labels)
    return result


def _read_yolo_labels(label_path: Optional[Path]) -> List[Tuple[int, float, float, float, float]]:
    if label_path is None or not label_path.exists():
        return []
    labels: List[Tuple[int, float, float, float, float]] = []
    try:
        content = label_path.read_text(encoding="utf-8")
    except Exception:
        return labels
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = (float(parts[idx]) for idx in range(1, 5))
            labels.append((cls_id, x, y, w, h))
        except ValueError:
            continue
    return labels


def _decode_plate_text(labels: List[Tuple[int, float, float, float, float]]) -> Optional[str]:
    if not labels:
        return None
    sorted_labels = sorted(labels, key=lambda item: item[1])
    chars: List[str] = []
    for cls_id, *_ in sorted_labels:
        if 0 <= cls_id < len(YOLO_CHARSET):
            chars.append(YOLO_CHARSET[cls_id])
    return "".join(chars) if chars else None


def _compute_tilt(labels: List[Tuple[int, float, float, float, float]]) -> Optional[float]:
    if len(labels) < 2:
        return None
    xs = np.asarray([item[1] for item in labels], dtype=np.float64)
    ys = np.asarray([item[2] for item in labels], dtype=np.float64)
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if abs(denom) < 1e-10:
        return None
    slope = ((xs - x_mean) * (ys - y_mean)).sum() / denom
    angle = -np.degrees(np.arctan(slope))
    return float(abs(angle))


def _compute_mean_iou(labels: List[Tuple[int, float, float, float, float]]) -> Optional[float]:
    boxes: List[Tuple[float, float, float, float]] = []
    for _, x, y, w, h in labels:
        if w <= 0 or h <= 0:
            continue
        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x + 0.5 * w
        y2 = y + 0.5 * h
        boxes.append((x1, y1, x2, y2))
    if len(boxes) < 2:
        return None
    total = 0.0
    count = 0
    for idx in range(len(boxes) - 1):
        total += _iou_xyxy(boxes[idx], boxes[idx + 1])
        count += 1
    return float(total / count) if count else None


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union







