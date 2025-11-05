"""Dataset preview page for viewing all loaded datasets."""

from __future__ import annotations

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

from core import datasets as ds_state


def render_dataset_preview_page() -> bool:
    """Render the dataset preview page and return True when state changes."""
    st.title("📋 Dataset Preview")
    st.markdown(
        "View data previews for all loaded datasets. "
        "Each dataset is displayed in a separate tab for easy navigation."
    )

    ds_state.ensure_dataset_state()
    ds_state.deduplicate_datasets()

    entries = list(ds_state.list_datasets().values())
    if not entries:
        st.info("No datasets loaded yet. Add a dataset on the Data Source page to get started.")
        return False

    # Create tabs for each dataset
    tab_labels = [f"{entry.name} ({len(entry.dataset)} rows)" for entry in entries]
    tabs = st.tabs(tab_labels)

    for tab, entry in zip(tabs, entries):
        with tab:
            _render_dataset_preview(entry)

    return False


def _render_dataset_preview(entry) -> None:
    """Render preview for a single dataset entry."""
    dataset = entry.dataset

    if dataset.empty:
        st.info(f"Dataset '{entry.name}' is empty.")
        return

    # Display basic information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(dataset))
    with col2:
        st.metric("Columns", len(dataset.columns))
    with col3:
        st.metric("Ground-truth entries", len(entry.ground_truth))
    with col4:
        st.metric("Label runs", len(entry.label_runs))

    with st.expander("Dataset details", expanded=False):
        st.caption(f"**Created at:** {entry.created_at}")
        st.caption(f"**Image directory:** {entry.image_dir or 'N/A'}")
        st.caption(f"**Label directory:** {entry.label_dir or 'N/A'}")

    st.divider()

    # Display data preview
    st.subheader("Data Preview")

    # Ensure is_wrong column exists in the dataset
    if "is_wrong" not in dataset.columns:
        # If is_wrong doesn't exist, create it and initialize to False
        ds_state.update_is_wrong_column(entry)

    # Remove technical columns for better readability
    preview_df = dataset.drop(columns=["image_path", "label_path"], errors="ignore")

    # Reorder columns: put is_wrong right after filename
    if "filename" in preview_df.columns and "is_wrong" in preview_df.columns:
        # Get all other columns except filename and is_wrong
        other_cols = [col for col in preview_df.columns if col not in ["filename", "is_wrong"]]
        # Reorder: filename, is_wrong, then others
        preview_df = preview_df[["filename", "is_wrong"] + other_cols]
    elif "is_wrong" in preview_df.columns:
        # If no filename column, put is_wrong first
        other_cols = [col for col in preview_df.columns if col != "is_wrong"]
        preview_df = preview_df[["is_wrong"] + other_cols]

    # Configure AgGrid options
    gb = GridOptionsBuilder.from_dataframe(preview_df)

    # Global column configuration
    gb.configure_default_column(
        filterable=True,        # Enable filtering
        sortable=True,          # Enable sorting
        resizable=True,         # Allow column resizing
        editable=False,         # Read-only
    )

    # Special configuration for is_wrong column
    if "is_wrong" in preview_df.columns:
        gb.configure_column(
            "is_wrong",
            header_name="Error",
            filter="agSetColumnFilter",  # Checkbox filter for boolean
            width=100,
        )

    # Pin filename column to the left
    if "filename" in preview_df.columns:
        gb.configure_column(
            "filename",
            header_name="Filename",
            filter="agTextColumnFilter",
            pinned="left",
            width=200,
        )

    # Configure numeric columns with number filter
    for col in preview_df.columns:
        if preview_df[col].dtype in ['int64', 'float64']:
            gb.configure_column(
                col,
                filter="agNumberColumnFilter",  # Number filter with range support
                type=["numericColumn"],
            )

    # Enable pagination
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=50,
    )

    # Enable side bar for advanced filtering
    gb.configure_side_bar(
        filters_panel=True,
        columns_panel=True,
    )

    # Enable selection
    gb.configure_selection(
        selection_mode='multiple',
        use_checkbox=True,
    )

    grid_options = gb.build()

    # Display AgGrid
    st.caption("💡 Tip: Click column headers to filter and sort. Use the sidebar (☰) for advanced filters.")

    grid_response = AgGrid(
        preview_df,
        gridOptions=grid_options,
        height=400,
        width='100%',
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        theme='streamlit',
        allow_unsafe_jscode=True,
    )

    # Show info about filtered data
    filtered_df = grid_response['data']
    if len(filtered_df) < len(preview_df):
        st.info(f"Showing {len(filtered_df)} of {len(preview_df)} rows (filtered).")
    else:
        st.caption(f"Displaying {len(preview_df)} rows, {len(preview_df.columns)} columns. Hidden: image_path, label_path")
