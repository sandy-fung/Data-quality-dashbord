"""Streamlit entry point for JSON Quality Dashboard."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from core import datasets as ds_state
from core.data_manager import data_manager
from ui_pages.data_source import render_data_source_page
from ui_pages.dataset_preview import render_dataset_preview_page
from ui_pages.analysis_runs import render_analysis_runs_page
from ui_pages.trends import render_trends_page
from ui_pages.text_analysis import render_text_analysis_page


st.set_page_config(
    page_title="JSON Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Ensure all expected streamlit session keys exist."""
    defaults = {
        "dataset": None,
        "dataset_summary": {},
        "dataset_path": None,
        "auto_save": True,
        "last_save_time": None,
        "session_loaded": False,
        "last_image_dir": "",
        "last_label_dir": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    ds_state.ensure_dataset_state()


def render_sidebar() -> str:
    """Render sidebar controls and return the selected page."""
    with st.sidebar:
        st.title("📊 JSON Dashboard")
        st.caption("Quality analytics for large JSON datasets")

        session_info = data_manager.get_session_info()
        ds_state.ensure_dataset_state()
        entry = ds_state.get_active_dataset()
        dataset_count = len(ds_state.list_datasets())
        if session_info.get("exists"):
            st.success("Snapshot available")
            st.caption(
                f"Datasets: {dataset_count} | "
                f"Rows: {session_info.get('dataset_rows', 0)} | "
                f"Label runs: {session_info.get('label_run_count', 0)}"
            )
        else:
            st.info("No snapshot saved yet")
            st.caption(f"Datasets in session: {dataset_count}")

        st.divider()

        # Check if any dataset has label runs for Text Analysis
        has_label_runs = any(
            ds_state.get_dataset_label_runs(entry.id)
            for entry in ds_state.list_datasets().values()
        )

        pages = ["📁 Data Source", "📋 Dataset Preview", "🚨 Error Overview", "📈 Trends"]
        if has_label_runs:
            pages.append("📝 Text Analysis")

        previous_page = st.session_state.get("navigation_page")
        if not previous_page or previous_page not in pages:
            previous_page = pages[0]
        page = st.radio(
            "Navigation",
            options=pages,
            index=pages.index(previous_page),
            key="navigation_page",
            help="Choose a page to explore",
        )

        st.divider()

        st.checkbox(
            "Auto-save on changes",
            key="auto_save",
            value=st.session_state.auto_save,
            help="Automatically persist changes after imports or edits",
        )

        if st.button("💾 Save Snapshot", width="stretch"):
            success, message = data_manager.save_session()
            icon = "✅" if success else "❌"
            st.toast(message, icon=icon)

        if st.button("📂 Load Snapshot", width="stretch"):
            success, message = data_manager.load_session()
            icon = "✅" if success else "❌"
            st.toast(message, icon=icon)

        export_bytes = data_manager.export_data()
        if export_bytes:
            st.download_button(
                label="⬇️ Export Snapshot",
                data=export_bytes,
                file_name=f"json_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz",
                mime="application/gzip",
                width="stretch",
            )

        uploaded = st.file_uploader(
            "⬆️ Import Snapshot",
            type=["json", "gz"],
            accept_multiple_files=False,
            key="snapshot_import",
        )
        if uploaded is not None:
            success, message = data_manager.import_data(uploaded.read())
            icon = "✅" if success else "❌"
            st.toast(message, icon=icon)
            if success:
                st.rerun()

        st.divider()

        if st.button("🧹 Reset All Data", type="secondary", width="stretch"):
            # Clear Streamlit caches first
            st.cache_data.clear()
            st.cache_resource.clear()

            # Clear snapshot files before clearing session state
            data_manager.clear_all()

            # Clear all datasets
            ds_state.ensure_dataset_state()
            for dataset_id in list(ds_state.list_datasets().keys()):
                ds_state.delete_dataset(dataset_id)

            # Clear session state completely
            st.session_state.clear()

            # Reinitialize session state with clean slate
            init_session_state()

            st.success("All data cleared successfully")
            st.rerun()

        dataset = entry.dataset if entry else None
        if dataset is not None and not dataset.empty:
            st.metric("Rows", len(dataset))
            st.metric("Columns", len(dataset.columns))
            st.caption(f"Active dataset: {entry.name}")
        else:
            st.metric("Rows", 0)
            st.metric("Columns", 0)

        if st.session_state.last_save_time:
            st.caption(f"Last saved: {st.session_state.last_save_time.strftime('%H:%M:%S')}")

    return page


def auto_save_if_needed() -> None:
    """Persist session automatically when the flag is enabled."""
    if st.session_state.auto_save:
        success, message = data_manager.save_session(auto=True)
        icon = "💾" if success else "⚠️"
        st.toast(message, icon=icon)


def main() -> None:
    init_session_state()
    page = render_sidebar()

    if page == "📁 Data Source":
        changed = render_data_source_page()
        if changed:
            auto_save_if_needed()
    elif page == "📋 Dataset Preview":
        render_dataset_preview_page()
    elif page == "🚨 Error Overview":
        changed = render_analysis_runs_page()
        if changed:
            auto_save_if_needed()
    elif page == "📈 Trends":
        render_trends_page()
    elif page == "📝 Text Analysis":
        render_text_analysis_page()


if __name__ == "__main__":
    main()
