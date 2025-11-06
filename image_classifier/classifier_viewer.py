#!/usr/bin/env python3
"""
Streamlit GUI for visualizing image classification results.

Usage:
    streamlit run classifier_viewer.py
"""

from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Image Classification Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# YOLO class ID to character mapping
CLS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load classification thresholds from JSON config file."""
    if config_path is None:
        # Use default config file in the same directory
        config_path = str(Path(__file__).parent / "classifier_config.json")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config["thresholds"]
    except FileNotFoundError:
        st.error(f"Config file not found: {config_path}")
        st.error("Please ensure classifier_config.json exists in the same directory.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format in config file: {e}")
        sys.exit(1)
    except KeyError as e:
        st.error(f"Missing required key in config file: {e}")
        st.error("Config file must contain 'thresholds' key.")
        sys.exit(1)


def parse_yolo_prediction(image_path: str) -> str:
    """
    Parse YOLO prediction from txt file in image directory.

    YOLO format: class_id x_center y_center width height (one detection per line)

    Args:
        image_path: Path to the image file

    Returns:
        Predicted string sorted by x coordinate, or "N/A" if file not found/invalid
    """
    try:
        img_path = Path(image_path)
        txt_path = img_path.with_suffix('.txt')

        if not txt_path.exists():
            return "N/A"

        # Read and parse all detections
        detections = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])

                    # Map class_id to character
                    if class_id in CLS_MAP:
                        char = CLS_MAP[class_id]
                        detections.append((x_center, char))
                except (ValueError, IndexError):
                    continue

        if not detections:
            return "N/A"

        # Sort by x coordinate and concatenate characters
        detections.sort(key=lambda x: x[0])
        result = ''.join(char for _, char in detections)

        return result if result else "N/A"

    except Exception:
        return "N/A"


def classify_value(value: Optional[float], levels: Dict[str, List[float]]) -> Optional[str]:
    """
    Classify a metric value into one of the levels based on thresholds.

    Args:
        value: The metric value to classify
        levels: Dictionary mapping level names to [min, max] ranges

    Returns:
        Level name (e.g., "dark", "medium", "bright") or None if value is None
    """
    if value is None:
        return None

    for level_name, (min_val, max_val) in levels.items():
        if min_val <= value < max_val:
            return level_name

    # If value is above all ranges, return the highest level
    return list(levels.keys())[-1]


def load_results(uploaded_file) -> Optional[Dict[str, Any]]:
    """Load classification results from uploaded JSON file."""
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        return data
    except Exception as e:
        st.error(f"Failed to load JSON file: {e}")
        return None


def create_dataframe(results: List[Dict], thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert results list to pandas DataFrame with dynamic classification.

    Args:
        results: List of image results with metrics
        thresholds: Classification thresholds from config

    Returns:
        DataFrame with metrics and dynamically computed classifications
    """
    rows = []
    for result in results:
        metrics = result['metrics']

        # Dynamically classify based on thresholds
        brightness_class = classify_value(
            metrics['brightness_mean'],
            thresholds['brightness']['levels']
        )
        brightness_median_class = classify_value(
            metrics['brightness_median_filtered'],
            thresholds['brightness_median']['levels']
        )
        contrast_class = classify_value(
            metrics['brightness_std'],
            thresholds['contrast']['levels']
        )
        sharpness_class = classify_value(
            metrics['blurriness'],
            thresholds['sharpness']['levels']
        )

        row = {
            'filename': result['filename'],
            'path': result['path'],
            'brightness_value': metrics['brightness_mean'],
            'brightness_median': metrics['brightness_median_filtered'],
            'contrast_value': metrics['brightness_std'],
            'sharpness_value': metrics['blurriness'],
            'width': metrics['width'],
            'height': metrics['height'],
            'brightness_class': brightness_class,
            'brightness_median_class': brightness_median_class,
            'contrast_class': contrast_class,
            'sharpness_class': sharpness_class,
            'prediction': parse_yolo_prediction(result['path']),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Compute classification statistics from DataFrame.

    Args:
        df: DataFrame with classification columns

    Returns:
        Dictionary mapping metric names to level counts
    """
    statistics = {}

    # Count classifications for each metric
    for metric in ['brightness', 'brightness_median', 'contrast', 'sharpness']:
        class_col = f'{metric}_class'
        if class_col in df.columns:
            counts = df[class_col].value_counts().to_dict()
            statistics[metric] = counts

    return statistics


def plot_statistics(statistics: Dict[str, Dict[str, int]]) -> go.Figure:
    """Create bar chart showing classification distribution."""
    fig = go.Figure()

    metrics = ['brightness', 'brightness_median', 'contrast', 'sharpness']
    colors = {
        'brightness': '#636EFA',
        'brightness_median': '#AB63FA',
        'contrast': '#EF553B',
        'sharpness': '#00CC96'
    }

    for metric in metrics:
        if metric in statistics:
            levels = list(statistics[metric].keys())
            counts = list(statistics[metric].values())

            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=levels,
                y=counts,
                marker_color=colors.get(metric, '#636EFA'),
                text=counts,
                textposition='outside',
            ))

    fig.update_layout(
        title="Classification Distribution",
        xaxis_title="Classification Level",
        yaxis_title="Count",
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1,
        showlegend=True,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def plot_metric_distribution(df: pd.DataFrame) -> go.Figure:
    """Create histogram showing metric value distributions."""
    fig = go.Figure()

    metrics_config = [
        ('brightness_value', 'Brightness', '#636EFA'),
        ('brightness_median', 'Brightness (Median)', '#AB63FA'),
        ('contrast_value', 'Contrast', '#EF553B'),
        ('sharpness_value', 'Sharpness', '#00CC96'),
    ]

    for column, name, color in metrics_config:
        fig.add_trace(go.Histogram(
            x=df[column],
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=20,
        ))

    fig.update_layout(
        title="Metric Value Distributions",
        xaxis_title="Value",
        yaxis_title="Count",
        barmode='overlay',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def calculate_image_histogram(image_path: str) -> Optional[np.ndarray]:
    """
    Calculate brightness histogram from image.

    Args:
        image_path: Path to image file

    Returns:
        Histogram array with 256 bins or None if error
    """
    try:
        import numpy as np

        img = Image.open(image_path)
        gray = img.convert('L')
        gray_arr = np.array(gray)
        hist, _ = np.histogram(gray_arr.flatten(), bins=256, range=(0, 256))
        return hist
    except Exception:
        return None


def plot_brightness_histogram(hist) -> go.Figure:
    """
    Create detailed brightness histogram with 256 bins and quartile markers.

    Args:
        hist: Histogram array with 256 bins

    Returns:
        Plotly figure with full 256-bin histogram
    """
    import numpy as np

    # Bin centers (0-255)
    bin_centers = np.arange(256)

    # Assign colors based on quartile regions (low transparency)
    colors = np.where(
        bin_centers < 64, 'rgba(70, 130, 180, 0.8)',   # Dark: blue
        np.where(
            bin_centers < 128, 'rgba(100, 180, 100, 0.8)',  # Mid-dark: green
            np.where(
                bin_centers < 192, 'rgba(255, 200, 50, 0.8)',  # Mid-bright: yellow
                'rgba(255, 140, 0, 0.8)'  # Bright: orange
            )
        )
    )

    # Create main histogram with 256 bins
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        marker=dict(
            color=colors.tolist(),
            line=dict(width=0)  # No border for compact display
        ),
        hovertemplate=(
            '<b>Brightness</b>: %{x}<br>'
            '<b>Pixel Count</b>: %{y:,}<br>'
            '<extra></extra>'
        ),
        showlegend=False
    ))

    # Layout optimization
    fig.update_layout(
        xaxis=dict(
            title='Brightness (0 = Black, 255 = White)',
            tickmode='array',
            tickvals=[0, 64, 128, 192, 255],
            ticktext=['0', '64', '128', '192', '255'],
            range=[-5, 260]  # Slightly extended to show boundaries
        ),
        yaxis=dict(
            title='Pixel Count (log scale)',
            type='log'
        ),
        height=220,
        margin=dict(l=40, r=20, t=30, b=50),
        bargap=0,  # No gap for continuous appearance
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        hovermode='x unified',
        showlegend=False
    )

    return fig


def filter_results(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered = df.copy()

    # Classification filters
    if filters['brightness_classes']:
        filtered = filtered[filtered['brightness_class'].isin(filters['brightness_classes'])]

    if filters['brightness_median_classes']:
        filtered = filtered[filtered['brightness_median_class'].isin(filters['brightness_median_classes'])]

    if filters['contrast_classes']:
        filtered = filtered[filtered['contrast_class'].isin(filters['contrast_classes'])]

    if filters['sharpness_classes']:
        filtered = filtered[filtered['sharpness_class'].isin(filters['sharpness_classes'])]

    # Value range filters
    if filters['brightness_range']:
        filtered = filtered[
            (filtered['brightness_value'] >= filters['brightness_range'][0]) &
            (filtered['brightness_value'] <= filters['brightness_range'][1])
        ]

    if filters['brightness_median_range']:
        filtered = filtered[
            (filtered['brightness_median'] >= filters['brightness_median_range'][0]) &
            (filtered['brightness_median'] <= filters['brightness_median_range'][1])
        ]

    if filters['contrast_range']:
        filtered = filtered[
            (filtered['contrast_value'] >= filters['contrast_range'][0]) &
            (filtered['contrast_value'] <= filters['contrast_range'][1])
        ]

    if filters['sharpness_range']:
        filtered = filtered[
            (filtered['sharpness_value'] >= filters['sharpness_range'][0]) &
            (filtered['sharpness_value'] <= filters['sharpness_range'][1])
        ]

    return filtered


def main():
    """Main Streamlit application."""

    st.title("Image Classification Viewer")
    st.markdown("Visualize and explore image quality classification results")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload classification results (JSON)",
        type=['json'],
        help="Upload the result.json file generated by image_classifier.py"
    )

    if uploaded_file is None:
        st.info("Please upload a classification results JSON file to begin")
        st.markdown("---")
        st.markdown("""
        ### How to generate results:
        ```bash
        # Classify images
        uv run python image_classifier.py -i ./images/ -o results.json -v

        # Then upload results.json to this viewer
        ```
        """)
        return

    # Load data
    data = load_results(uploaded_file)
    if data is None:
        return

    # Load classification thresholds
    thresholds = load_config()

    # Extract components
    summary = data.get('summary', {})
    results = data.get('results', [])

    if not results:
        st.warning("No results found in the uploaded file")
        return

    # Create DataFrame with dynamic classification
    df = create_dataframe(results, thresholds)

    # Compute statistics from classified data
    statistics = compute_statistics(df)

    # Initialize session state
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0

    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    # === SUMMARY SECTION ===
    st.header("Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Images", summary.get('total_images', len(results)))
    with col2:
        st.metric("Timestamp", summary.get('timestamp', 'N/A')[:10])
    with col3:
        avg_brightness = df['brightness_value'].mean()
        st.metric("Avg Brightness", f"{avg_brightness:.1f}")
    with col4:
        avg_brightness_median = df['brightness_median'].mean()
        st.metric("Avg Brightness (Median)", f"{avg_brightness_median:.1f}")
    with col5:
        avg_sharpness = df['sharpness_value'].mean()
        st.metric("Avg Sharpness", f"{avg_sharpness:.1f}")

    # === STATISTICS CHARTS ===
    st.header("Statistics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_stats = plot_statistics(statistics)
        st.plotly_chart(fig_stats, use_container_width=True)

    with chart_col2:
        fig_dist = plot_metric_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # === FILTERS SIDEBAR ===
    with st.sidebar:
        st.header("Filters")

        st.subheader("Classification Levels")

        # Brightness filter
        brightness_options = df['brightness_class'].dropna().unique().tolist()
        brightness_selected = st.multiselect(
            "Brightness",
            options=brightness_options,
            default=brightness_options,
            key=f'brightness_filter_{st.session_state.reset_counter}'
        )

        # Brightness median filter
        brightness_median_options = df['brightness_median_class'].dropna().unique().tolist()
        brightness_median_selected = st.multiselect(
            "Brightness (Median)",
            options=brightness_median_options,
            default=brightness_median_options,
            key=f'brightness_median_filter_{st.session_state.reset_counter}'
        )

        # Contrast filter
        contrast_options = df['contrast_class'].dropna().unique().tolist()
        contrast_selected = st.multiselect(
            "Contrast",
            options=contrast_options,
            default=contrast_options,
            key=f'contrast_filter_{st.session_state.reset_counter}'
        )

        # Sharpness filter
        sharpness_options = df['sharpness_class'].dropna().unique().tolist()
        sharpness_selected = st.multiselect(
            "Sharpness",
            options=sharpness_options,
            default=sharpness_options,
            key=f'sharpness_filter_{st.session_state.reset_counter}'
        )

        st.markdown("---")
        st.subheader("Metric Value Ranges")

        # Brightness range
        brightness_min, brightness_max = float(df['brightness_value'].min()), float(df['brightness_value'].max())
        brightness_range = st.slider(
            "Brightness Value",
            min_value=brightness_min,
            max_value=brightness_max,
            value=(brightness_min, brightness_max),
            key=f'brightness_range_{st.session_state.reset_counter}'
        )

        # Brightness median range
        brightness_median_min, brightness_median_max = float(df['brightness_median'].min()), float(df['brightness_median'].max())
        brightness_median_range = st.slider(
            "Brightness (Median) Value",
            min_value=brightness_median_min,
            max_value=brightness_median_max,
            value=(brightness_median_min, brightness_median_max),
            key=f'brightness_median_range_{st.session_state.reset_counter}'
        )

        # Contrast range
        contrast_min, contrast_max = float(df['contrast_value'].min()), float(df['contrast_value'].max())
        contrast_range = st.slider(
            "Contrast Value",
            min_value=contrast_min,
            max_value=contrast_max,
            value=(contrast_min, contrast_max),
            key=f'contrast_range_{st.session_state.reset_counter}'
        )

        # Sharpness range
        sharpness_min, sharpness_max = float(df['sharpness_value'].min()), float(df['sharpness_value'].max())
        sharpness_range = st.slider(
            "Sharpness Value",
            min_value=sharpness_min,
            max_value=sharpness_max,
            value=(sharpness_min, sharpness_max),
            key=f'sharpness_range_{st.session_state.reset_counter}'
        )

        # Reset button
        if st.button("Reset Filters", use_container_width=True):
            st.session_state.reset_counter += 1
            st.rerun()

    # Apply filters
    filters = {
        'brightness_classes': brightness_selected,
        'brightness_median_classes': brightness_median_selected,
        'contrast_classes': contrast_selected,
        'sharpness_classes': sharpness_selected,
        'brightness_range': brightness_range,
        'brightness_median_range': brightness_median_range,
        'contrast_range': contrast_range,
        'sharpness_range': sharpness_range,
    }

    filtered_df = filter_results(df, filters)

    # === MAIN CONTENT AREA ===
    st.header(f"Images ({len(filtered_df)} / {len(df)})")

    if filtered_df.empty:
        st.warning("No images match the current filters")
        return

    # === GRID LAYOUT: 3 images per row ===
    # Convert filtered results to list for easier iteration
    filtered_results = []
    for idx, row in filtered_df.iterrows():
        result = next((r for r in results if r['filename'] == row['filename']), None)
        if result:
            filtered_results.append((idx, row, result))

    # Display images in grid (3 per row)
    num_cols = 3
    for i in range(0, len(filtered_results), num_cols):
        cols = st.columns(num_cols)

        for col_idx in range(num_cols):
            item_idx = i + col_idx
            if item_idx >= len(filtered_results):
                break

            idx, row, result = filtered_results[item_idx]

            with cols[col_idx]:
                # Container for each image card with border and background
                with st.container():
                    # First row: Filename (full width)
                    st.markdown(f"**Filename:** {result['filename']}")

                    # Second row: Image (left) + Attributes table (right)
                    img_col, attr_col = st.columns([1, 1])

                    with img_col:
                        # Try to load and display original image
                        try:
                            img_path = Path(result['path'])
                            if img_path.exists():
                                img = Image.open(img_path)

                                # Convert image to base64 for HTML display
                                buffered = BytesIO()
                                img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()

                                # Display using HTML with CSS to prevent stretching
                                # Fixed container height ensures grid alignment
                                st.markdown(f'''
                                <div style="width: 100%; height: 250px;
                                            display: flex; align-items: center;
                                            justify-content: center;
                                            background-color: #f8f9fa;
                                            border: 1px solid #dee2e6;
                                            border-radius: 4px;">
                                    <img src="data:image/png;base64,{img_str}"
                                         style="max-width: 100%; max-height: 100%;
                                                object-fit: contain;">
                                </div>
                                ''', unsafe_allow_html=True)
                            else:
                                st.warning("Not found")
                        except Exception as e:
                            st.error(f"Error: {e}")

                    with attr_col:
                        # Display attributes in table format with color coding
                        # Helper function to get color for classification
                        def get_color(metric, value):
                            colors = {
                                'brightness': {
                                    'dark': '#B3D9FF',
                                    'medium': '#FFE599',
                                    'bright': '#FFB3B3'
                                },
                                'contrast': {
                                    'low': '#E8E8E8',
                                    'medium': '#C0C0C0',
                                    'high': '#A0A0A0'
                                },
                                'sharpness': {
                                    'blurry': '#FFB3B3',
                                    'sharp': '#B3FFB3'
                                }
                            }
                            return colors.get(metric, {}).get(value, '#FFFFFF')

                        # Get classification values from DataFrame row
                        brightness_val = row['brightness_class']
                        brightness_median_val = row['brightness_median_class']
                        contrast_val = row['contrast_class']
                        sharpness_val = row['sharpness_class']
                        prediction_val = row['prediction']

                        # Get metric values from result
                        brightness_metric = result['metrics']['brightness_mean']
                        brightness_median_metric = result['metrics']['brightness_median_filtered']
                        contrast_metric = result['metrics']['brightness_std']
                        sharpness_metric = result['metrics']['blurriness']

                        # Get colors
                        brightness_color = get_color('brightness', brightness_val)
                        brightness_median_color = get_color('brightness', brightness_median_val)
                        contrast_color = get_color('contrast', contrast_val)
                        sharpness_color = get_color('sharpness', sharpness_val)

                        # Create HTML table with colored backgrounds and values
                        table_html = f"""
                        <style>
                            .attr-table {{
                                width: 100%;
                                border-collapse: collapse;
                                font-size: 14px;
                                background-color: #F8F9FA;
                                border: 1px solid #DEE2E6;
                            }}
                            .attr-table th, .attr-table td {{
                                padding: 6px 8px;
                                text-align: left;
                                border: 1px solid #DEE2E6;
                            }}
                            .attr-table th {{
                                background-color: #E9ECEF;
                                font-weight: bold;
                            }}
                            .colored-cell {{
                                font-weight: 500;
                            }}
                        </style>
                        <table class="attr-table">
                            <tr>
                                <th>Attribute</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Brightness</td>
                                <td class="colored-cell" style="background-color: {brightness_color};">{brightness_val} ({brightness_metric:.1f})</td>
                            </tr>
                            <tr>
                                <td>Brightness (Median)</td>
                                <td class="colored-cell" style="background-color: {brightness_median_color};">{brightness_median_val} ({f'{brightness_median_metric:.1f}' if brightness_median_metric is not None else 'N/A'})</td>
                            </tr>
                            <tr>
                                <td>Contrast</td>
                                <td class="colored-cell" style="background-color: {contrast_color};">{contrast_val} ({contrast_metric:.1f})</td>
                            </tr>
                            <tr>
                                <td>Sharpness</td>
                                <td class="colored-cell" style="background-color: {sharpness_color};">{sharpness_val} ({sharpness_metric:.1f})</td>
                            </tr>
                            <tr>
                                <td>Prediction</td>
                                <td style="background-color: #E8F4F8; font-weight: 500;">{prediction_val}</td>
                            </tr>
                            <tr>
                                <td>Size</td>
                                <td>{int(result['metrics']['width'])}x{int(result['metrics']['height'])}</td>
                            </tr>
                        </table>
                        """

                        st.markdown(table_html, unsafe_allow_html=True)

                    # Third row: Histogram (full width)
                    hist = calculate_image_histogram(result['path'])
                    if hist is not None:
                        fig = plot_brightness_histogram(hist)
                        st.plotly_chart(fig, use_container_width=True, key=f"hist_{item_idx}")

                    st.markdown("---")

    # === EXPORT SECTION ===
    st.markdown("---")
    st.header("Export")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        # Export filtered results as CSV
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="filtered_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with export_col2:
        # Export filtered results as JSON
        filtered_results = [
            r for r in results
            if r['filename'] in filtered_df['filename'].values
        ]
        export_data = {
            'summary': {
                'total_images': len(filtered_results),
                'filters_applied': filters,
            },
            'results': filtered_results
        }
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name="filtered_results.json",
            mime="application/json",
            use_container_width=True
        )

    with export_col3:
        # Copy file paths
        file_paths = "\n".join(filtered_df['path'].tolist())
        st.download_button(
            label="Download File Paths",
            data=file_paths,
            file_name="image_paths.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Show path list
    with st.expander("View Filtered File Paths"):
        st.code(file_paths, language='text')


if __name__ == "__main__":
    main()
