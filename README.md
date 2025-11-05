# JSON Quality Dashboard

Streamlit dashboard for exploring large JSON datasets with quality metrics and interactive visualizations. Designed to mimic the workflow of the legacy "年久失修" project while keeping the lean, modular structure of the `dataset_analyzer` application.

## Features

- Load JSON data from local paths such as `D:/dataset/picked/test/master.json`
- Persist parsed datasets with gzip-compressed session snapshots
- Interactive pages for dataset inspection, run comparisons, and trend visualization
- Plotly-based charts for distribution, time series, and categorical analysis
- Modular core layer (`core/`) covering import, statistics, plotting, and persistence

## Installation

```bash
uv sync
```

```bash
uv run streamlit run app.py
```

### Using pip

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
json_quality_dashboard/
├── app.py             # Main Streamlit entry point
├── core/              # Core logic modules
├── ui_pages/          # Streamlit page components
├── tests/             # Unit tests
├── mock/              # Sample JSON snippets
├── pyproject.toml     # Project metadata
└── requirements.txt   # Runtime dependencies
```

## Testing

```bash
pytest -q
```

## License

MIT
