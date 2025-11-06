#!/usr/bin/env python3
"""
Independent image quality classifier tool.

Classifies images into three levels based on quality metrics:
- Brightness: dark / medium / bright
- Contrast: low / medium / high
- Sharpness: blurry / normal / sharp
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Image file extensions to process
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Optional OpenCV support for faster Laplacian computation
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


def extract_image_features(image_path: Path) -> Dict[str, Optional[float]]:
    """
    Extract quality metrics from an image.

    Reuses logic from core/importers.py for consistency.

    Returns:
        Dictionary with brightness_mean, brightness_std, and blurriness values.
    """
    result: Dict[str, Optional[float]] = {
        "width": None,
        "height": None,
        "brightness_mean": None,
        "brightness_std": None,
        "brightness_median_filtered": None,
        "blurriness": None,
    }

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            # Convert to grayscale
            gray = image.convert("L")

            # Resize to 256x256 for consistent processing
            resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS  # type: ignore[attr-defined]
            gray_small = gray.resize((256, 256), resample=resample)
            gray_arr = np.asarray(gray_small, dtype=np.float32)

            result["width"] = float(width)
            result["height"] = float(height)

            if gray_arr.size > 0:
                result["brightness_mean"] = float(gray_arr.mean())
                result["brightness_std"] = float(gray_arr.std())

                # Calculate filtered median: remove outliers (mean ± 2*std) then take median
                mean = gray_arr.mean()
                std = gray_arr.std()
                filtered = gray_arr[(gray_arr >= mean - 2*std) & (gray_arr <= mean + 2*std)]
                result["brightness_median_filtered"] = float(np.median(filtered)) if filtered.size > 0 else None

                result["blurriness"] = compute_blurriness(gray_arr)

    except Exception as e:
        print(f"Warning: Could not process {image_path}: {e}", file=sys.stderr)
        return result

    return result


def compute_blurriness(gray_arr: np.ndarray) -> Optional[float]:
    """
    Compute image sharpness using Laplacian variance.

    Higher values indicate sharper images.
    Uses OpenCV if available, otherwise falls back to NumPy implementation.
    """
    if gray_arr.size == 0:
        return None

    # Prefer OpenCV for faster computation
    if cv2 is not None:
        lap = cv2.Laplacian(gray_arr.astype(np.float32), cv2.CV_64F)
        return float(lap.var()) if lap.size else None

    # NumPy fallback implementation
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


def process_single_image(image_path: Path) -> Dict[str, Any]:
    """
    Extract features from an image without classification.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing only raw metrics
    """
    features = extract_image_features(image_path)

    return {
        "filename": image_path.name,
        "path": str(image_path.resolve()),
        "metrics": {
            "brightness_mean": features["brightness_mean"],
            "brightness_std": features["brightness_std"],
            "brightness_median_filtered": features["brightness_median_filtered"],
            "blurriness": features["blurriness"],
            "width": features["width"],
            "height": features["height"],
        },
    }


def collect_image_paths(input_path: Path) -> List[Path]:
    """
    Collect all image file paths from input (file or directory).

    Args:
        input_path: Path to image file or directory

    Returns:
        List of image file paths
    """
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        else:
            print(f"Error: File {input_path} is not a supported image format", file=sys.stderr)
            return []

    elif input_path.is_dir():
        # Recursively scan directory for images
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(input_path.rglob(f"*{ext}"))
            images.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(images)

    else:
        print(f"Error: Path {input_path} is not a file or directory", file=sys.stderr)
        return []




def process_images(
    input_path: Path,
    output_path: Path,
    verbose: bool = False
) -> None:
    """
    Main processing function: extract metrics from images and save to JSON.

    Args:
        input_path: Input image file or directory
        output_path: Output JSON file path
        verbose: Show progress information
    """
    # Collect image paths
    image_paths = collect_image_paths(input_path)

    if not image_paths:
        print("Error: No images found to process", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Found {len(image_paths)} image(s) to process")

    # Process each image
    results = []
    for idx, img_path in enumerate(image_paths, 1):
        if verbose:
            print(f"Processing [{idx}/{len(image_paths)}]: {img_path.name}")

        result = process_single_image(img_path)
        results.append(result)

    # Build output JSON (metrics only, no classification)
    output_data = {
        "summary": {
            "total_images": len(results),
            "timestamp": datetime.now().isoformat(),
            "input_path": str(input_path.resolve()),
        },
        "results": results,
    }

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n✓ Results saved to: {output_path}")
        print(f"  Total images processed: {len(results)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract quality metrics from images (brightness, contrast, sharpness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  %(prog)s -i photo.jpg -o result.json

  # Directory batch processing
  %(prog)s -i ./images/ -o results.json -v
        """
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input image file or directory"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress information"
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate output path
    output_path = Path(args.output).expanduser()
    if output_path.exists() and not args.verbose:
        # Prompt for confirmation if output exists (unless verbose mode)
        pass  # In CLI tool, we'll just overwrite

    # Process images
    try:
        process_images(input_path, output_path, args.verbose)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
