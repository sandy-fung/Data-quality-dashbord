# Image Quality Classifier

Image quality analysis tool with brightness, contrast, and sharpness metrics, featuring a Streamlit GUI with 256-bin histogram visualization.

## Installation

```bash
uv pip install pillow numpy opencv-python streamlit plotly pandas
```

## Usage

### Option 1: GUI with Direct Image Analysis (Recommended)

```bash
uv run streamlit run classifier_viewer.py
```

Open browser at http://localhost:8501:
1. Select "Analyze images directly"
2. Upload images (drag & drop or browse)
3. Results are analyzed and displayed automatically

### Option 2: CLI + GUI Workflow

```bash
# Step 1: Extract metrics using CLI
uv run python image_classifier.py -i ./images/ -o results.json -v

# Step 2: Launch GUI
uv run streamlit run classifier_viewer.py
```

In the GUI:
1. Select "Upload JSON file"
2. Upload `results.json` for analysis

## GUI Features

- **Dual Input Modes**: Analyze images directly or upload pre-generated JSON
- Statistical dashboard with distribution charts
- Multi-dimensional filtering (brightness, contrast, sharpness)
- 256-bin brightness histogram with logarithmic scale
- YOLO prediction results display
- Export options:
  - **Full JSON**: Complete analysis results (use this to save direct analysis)
  - **Filtered CSV**: Filtered results in spreadsheet format
  - **Filtered JSON**: Filtered results for further processing
  - **File Paths**: Text list of image paths

## Configuration

Edit `classifier_config.json` to adjust classification thresholds for brightness, contrast, and sharpness levels.
