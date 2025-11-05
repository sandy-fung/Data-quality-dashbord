"""Dataset preview page for viewing all loaded datasets."""

from __future__ import annotations

import streamlit as st

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
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Rows", len(dataset))
    with col2:
        st.metric("Columns", len(dataset.columns))
    with col3:
        st.metric("Ground-truth entries", len(entry.ground_truth))
    with col4:
        st.metric("Label runs", len(entry.label_runs))
    with col5:
        # Count how many rows are marked as wrong
        wrong_count = dataset["is_wrong"].sum() if "is_wrong" in dataset.columns else 0
        st.metric("Errors", int(wrong_count))

    with st.expander("Dataset details", expanded=False):
        st.caption(f"**Created at:** {entry.created_at}")
        st.caption(f"**Image directory:** {entry.image_dir or 'N/A'}")
        st.caption(f"**Label directory:** {entry.label_dir or 'N/A'}")

    st.divider()

    # Display data preview
    st.subheader("Data Preview")

    # Add filter option for showing only errors
    show_filter = "is_wrong" in dataset.columns and dataset["is_wrong"].sum() > 0
    if show_filter:
        col_filter, col_spacer = st.columns([1, 3])
        with col_filter:
            show_only_errors = st.checkbox(
                "Show only errors",
                value=False,
                key=f"show_errors_{entry.id}",
                help="Display only rows marked as wrong (is_wrong=True)",
            )
    else:
        show_only_errors = False

    # Remove technical columns for better readability
    preview_df = dataset.drop(columns=["image_path", "label_path"], errors="ignore")

    # Apply filter if needed
    if show_only_errors and "is_wrong" in preview_df.columns:
        preview_df = preview_df[preview_df["is_wrong"] == True]
        if preview_df.empty:
            st.info("No errors found in this dataset.")
            return

    # Display the dataframe
    st.dataframe(
        preview_df,
        use_container_width=True,
        height=400,  # Fixed height for consistent display
    )

    # Optional: Show column info
    caption_parts = [f"Displaying {len(preview_df)} rows, {len(preview_df.columns)} columns."]
    caption_parts.append("Hidden: image_path, label_path")
    st.caption(" ".join(caption_parts))
