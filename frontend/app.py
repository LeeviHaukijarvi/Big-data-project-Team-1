"""
Streamlit dashboard for OpenWebText2 topic analysis.
Displays processed data from Azure Blob Storage with visualizations
for text analysis and placeholders for Phase 2 topic modeling.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load .env from parent directory (try multiple locations)
possible_paths = [
    Path(__file__).parent.parent / ".env",  # relative to script
    Path.cwd().parent / ".env",              # relative to cwd
    Path.cwd() / ".env",                     # current directory
]
for env_path in possible_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

# Page configuration
st.set_page_config(
    page_title="OpenWebText2 Topic Analysis",
    page_icon="📊",
    layout="wide",
)

# Azure configuration from environment
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "blob")
PROCESSED_PREFIX = "PROCESSED/"


@st.cache_resource
def get_blob_service_client() -> Optional[BlobServiceClient]:
    """Create and cache Azure Blob Service client."""
    if not AZURE_STORAGE_CONNECTION_STRING:
        return None
    try:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    except Exception as e:
        st.error(f"Failed to connect to Azure: {e}")
        return None


@st.cache_data(ttl=300)
def list_processed_files() -> list[str]:
    """List all parquet files in PROCESSED/ folder."""
    client = get_blob_service_client()
    if not client:
        return []

    try:
        container_client = client.get_container_client(AZURE_CONTAINER_NAME)
        blobs = [
            b.name for b in container_client.list_blobs(name_starts_with=PROCESSED_PREFIX)
            if b.name.endswith(".parquet")
        ]
        return sorted(blobs)
    except Exception as e:
        st.error(f"Failed to list files: {e}")
        return []


@st.cache_data(ttl=300)
def load_parquet_file(blob_name: str, sample_size: int = 1000) -> Optional[pd.DataFrame]:
    """Download and load a parquet file from Azure."""
    client = get_blob_service_client()
    if not client:
        return None

    tmp_path = None
    try:
        container_client = client.get_container_client(AZURE_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)

        # Create temp file, close it, then write/read (Windows compatibility)
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Download to temp file
        with open(tmp_path, "wb") as f:
            stream = blob_client.download_blob()
            f.write(stream.readall())

        # Read parquet
        table = pq.read_table(tmp_path)
        df = table.to_pandas()

        # Sample if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        return df
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


def render_sidebar():
    """Render sidebar with connection status and controls."""
    st.sidebar.title("Dashboard Controls")

    # Azure connection status
    st.sidebar.subheader("Azure Connection")
    client = get_blob_service_client()
    if client:
        st.sidebar.success("Connected")
    else:
        st.sidebar.error("Not connected")
        st.sidebar.caption("Set AZURE_STORAGE_CONNECTION_STRING in .env")
        return None, 0

    # File selector
    st.sidebar.subheader("Data Selection")
    files = list_processed_files()

    if not files:
        st.sidebar.warning("No processed files found")
        return None, 0

    selected_file = st.sidebar.selectbox(
        "Select batch file",
        files,
        format_func=lambda x: x.replace(PROCESSED_PREFIX, "")
    )

    # Sample size slider
    sample_size = st.sidebar.slider(
        "Sample size",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of records to load for visualization"
    )

    st.sidebar.divider()
    st.sidebar.caption("OpenWebText2 Pipeline Dashboard")

    return selected_file, sample_size


def render_overview_tab(df: pd.DataFrame):
    """Render the Overview tab with summary statistics."""
    st.header("Data Overview")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        avg_score = df["score"].mean() if "score" in df.columns else 0
        st.metric("Avg Score", f"{avg_score:.2f}")

    with col3:
        if "original_length" in df.columns:
            avg_orig = df["original_length"].mean()
            st.metric("Avg Original Length", f"{avg_orig:,.0f}")
        else:
            st.metric("Avg Original Length", "N/A")

    with col4:
        if "normalized_length" in df.columns:
            avg_norm = df["normalized_length"].mean()
            st.metric("Avg Normalized Length", f"{avg_norm:,.0f}")
        else:
            st.metric("Avg Normalized Length", "N/A")

    st.divider()

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")
        if "score" in df.columns:
            fig = px.histogram(
                df,
                x="score",
                nbins=50,
                title="Distribution of Content Quality Scores",
                labels={"score": "Score", "count": "Count"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Score column not available")

    with col2:
        st.subheader("Text Length Comparison")
        if "original_length" in df.columns and "normalized_length" in df.columns:
            length_df = pd.DataFrame({
                "Type": ["Original"] * len(df) + ["Normalized"] * len(df),
                "Length": list(df["original_length"]) + list(df["normalized_length"])
            })
            fig = px.box(
                length_df,
                x="Type",
                y="Length",
                title="Original vs Normalized Text Lengths",
                color="Type"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Length columns not available")

    # Length reduction stats
    if "original_length" in df.columns and "normalized_length" in df.columns:
        st.subheader("Normalization Impact")
        df["reduction_pct"] = (
            (df["original_length"] - df["normalized_length"]) / df["original_length"] * 100
        )
        avg_reduction = df["reduction_pct"].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Length Reduction", f"{avg_reduction:.1f}%")
        with col2:
            total_orig = df["original_length"].sum()
            total_norm = df["normalized_length"].sum()
            st.metric(
                "Total Characters Saved",
                f"{(total_orig - total_norm):,}"
            )


def render_samples_tab(df: pd.DataFrame):
    """Render the Text Samples tab with filterable data."""
    st.header("Text Samples")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        if "score" in df.columns:
            min_score, max_score = float(df["score"].min()), float(df["score"].max())
            score_range = st.slider(
                "Filter by score range",
                min_value=min_score,
                max_value=max_score,
                value=(min_score, max_score)
            )
            filtered_df = df[
                (df["score"] >= score_range[0]) & (df["score"] <= score_range[1])
            ]
        else:
            filtered_df = df

    with col2:
        st.metric("Filtered Records", f"{len(filtered_df):,}")

    st.divider()

    # Pagination
    page_size = 10
    total_pages = max(1, len(filtered_df) // page_size + (1 if len(filtered_df) % page_size else 0))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))

    st.caption(f"Showing records {start_idx + 1} to {end_idx} of {len(filtered_df)}")

    # Display records
    page_df = filtered_df.iloc[start_idx:end_idx]

    for idx, row in page_df.iterrows():
        with st.expander(
            f"Record {row.get('id', idx)} | Score: {row.get('score', 'N/A'):.2f}"
            if 'score' in row else f"Record {row.get('id', idx)}"
        ):
            if "text" in row:
                text = str(row["text"])
                # Show snippet first, full text on expansion
                if len(text) > 500:
                    st.text(text[:500] + "...")
                    if st.button(f"Show full text", key=f"full_{idx}"):
                        st.text_area("Full text", text, height=300, key=f"area_{idx}")
                else:
                    st.text(text)

            # Show metadata
            col1, col2 = st.columns(2)
            with col1:
                if "original_length" in row:
                    st.caption(f"Original length: {row['original_length']:,}")
            with col2:
                if "normalized_length" in row:
                    st.caption(f"Normalized length: {row['normalized_length']:,}")


def render_topics_tab():
    """Render the Topics tab (placeholder for Phase 2)."""
    st.header("Topic Modeling")

    st.info(
        """
        **Topic modeling coming in Phase 2**

        This tab will display:
        - Topic clusters discovered via LDA
        - Top words per topic with relevance scores
        - Topic distribution across the corpus
        - Interactive topic exploration

        The modeling service will process normalized text through PySpark LDA
        to extract meaningful topics from the OpenWebText2 corpus.
        """
    )

    # Placeholder visualization
    st.subheader("Preview: Topic Distribution (Placeholder)")

    placeholder_data = pd.DataFrame({
        "Topic": [f"Topic {i}" for i in range(1, 11)],
        "Document Count": [150, 120, 100, 90, 85, 75, 70, 60, 55, 45]
    })

    fig = px.bar(
        placeholder_data,
        x="Topic",
        y="Document Count",
        title="Example Topic Distribution (Placeholder Data)",
        color="Document Count",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("This is placeholder data. Real topic modeling results will appear after Phase 2 implementation.")


def render_drift_tab():
    """Render the Drift Timeline tab (placeholder for Phase 3)."""
    st.header("Drift Detection Timeline")

    st.info(
        """
        **Drift detection coming in Phase 3**

        This tab will display:
        - Timeline view of topic changes over time windows
        - Drift magnitude indicators per window
        - NPMI coherence scores for topic quality
        - Alerts for significant topic shifts

        The system will split data into time windows and detect
        when topic distributions shift significantly.
        """
    )

    # Placeholder timeline
    st.subheader("Preview: Topic Drift Over Time (Placeholder)")

    import numpy as np
    np.random.seed(42)

    placeholder_drift = pd.DataFrame({
        "Window": [f"W{i}" for i in range(1, 13)],
        "Topic 1": np.random.uniform(0.1, 0.3, 12),
        "Topic 2": np.random.uniform(0.15, 0.35, 12),
        "Topic 3": np.random.uniform(0.1, 0.25, 12),
    })

    fig = go.Figure()
    for col in ["Topic 1", "Topic 2", "Topic 3"]:
        fig.add_trace(go.Scatter(
            x=placeholder_drift["Window"],
            y=placeholder_drift[col],
            mode="lines+markers",
            name=col
        ))

    fig.update_layout(
        title="Example Topic Proportions Over Time (Placeholder Data)",
        xaxis_title="Time Window",
        yaxis_title="Topic Proportion",
        legend_title="Topics"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("This is placeholder data. Real drift detection results will appear after Phase 3 implementation.")


def main():
    """Main application entry point."""
    st.title("OpenWebText2 Topic Analysis Dashboard")

    # Sidebar
    selected_file, sample_size = render_sidebar()

    if not selected_file:
        st.warning("Please configure Azure connection and select a data file from the sidebar.")
        st.stop()

    # Load data
    with st.spinner(f"Loading data from {selected_file}..."):
        df = load_parquet_file(selected_file, sample_size)

    if df is None or df.empty:
        st.error("Failed to load data. Please check the file and try again.")
        st.stop()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Text Samples",
        "Topics (Phase 2)",
        "Drift Timeline (Phase 3)"
    ])

    with tab1:
        render_overview_tab(df)

    with tab2:
        render_samples_tab(df)

    with tab3:
        render_topics_tab()

    with tab4:
        render_drift_tab()


if __name__ == "__main__":
    main()
