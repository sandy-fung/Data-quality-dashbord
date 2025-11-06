# Image Quality Classifier

Image quality analysis tool with brightness, contrast, and sharpness metrics, featuring a Streamlit GUI with 256-bin histogram visualization.

## Installation

```bash
uv pip install pillow numpy opencv-python streamlit plotly pandas
```

## Usage

### 1. Extract Image Metrics

```bash
# Batch process directory
uv run python image_classifier.py -i ./images/ -o results.json -v

# Single image
uv run python image_classifier.py -i photo.jpg -o result.json
```

### 2. Launch GUI

```bash
uv run streamlit run classifier_viewer.py
```

Open browser at http://localhost:8501 and upload `results.json` for interactive analysis.

## GUI Features

- Statistical dashboard with distribution charts
- Multi-dimensional filtering (brightness, contrast, sharpness)
- 256-bin brightness histogram with logarithmic scale
- YOLO prediction results display
- Export filtered results (CSV/JSON)

## Configuration

Edit `classifier_config.json` to adjust classification thresholds for brightness, contrast, and sharpness levels.
