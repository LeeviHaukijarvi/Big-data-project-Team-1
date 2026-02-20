"""
Streamlit dashboard for OpenWebText2 topic analysis.
Displays processed data from Azure Blob Storage with visualizations
for text analysis and placeholders for Phase 2 topic modeling.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
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
FEATURES_PREFIX = "FEATURES/"


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
def list_feature_files() -> list[str]:
    """List all parquet files in FEATURES/ folder."""
    client = get_blob_service_client()
    if not client:
        return []

    try:
        container_client = client.get_container_client(AZURE_CONTAINER_NAME)
        blobs = [
            b.name for b in container_client.list_blobs(name_starts_with=FEATURES_PREFIX)
            if b.name.endswith(".parquet")
        ]
        return sorted(blobs)
    except Exception as e:
        st.error(f"Failed to list feature files: {e}")
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
        return None, None, 0

    # Data source selection
    st.sidebar.subheader("Data Selection")

    processed_files = list_processed_files()
    feature_files = list_feature_files()

    # Choose data source
    data_source = st.sidebar.radio(
        "Data source",
        ["Processed (text only)", "Features (with topics)"],
        index=1 if feature_files else 0
    )

    if data_source == "Features (with topics)" and feature_files:
        files = feature_files
        prefix = FEATURES_PREFIX
    else:
        files = processed_files
        prefix = PROCESSED_PREFIX

    if not files:
        st.sidebar.warning("No files found")
        return None, None, 0

    selected_file = st.sidebar.selectbox(
        "Select batch file",
        files,
        format_func=lambda x: x.replace(prefix, "")
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

    has_topics = "Features" in data_source and feature_files
    return selected_file, has_topics, sample_size


def render_overview_tab(df: pd.DataFrame):
    """Render the Overview tab with summary statistics."""
    st.header("Data Overview")

    # Model info if available
    if "model_version" in df.columns:
        model_version = df["model_version"].iloc[0] if len(df) > 0 else "N/A"
        st.info(f"Model Version: **{model_version}**")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        if "original_length" in df.columns:
            avg_orig = df["original_length"].mean()
            st.metric("Avg Original Length", f"{avg_orig:,.0f}")
        else:
            st.metric("Avg Original Length", "N/A")

    with col3:
        if "normalized_length" in df.columns:
            avg_norm = df["normalized_length"].mean()
            st.metric("Avg Normalized Length", f"{avg_norm:,.0f}")
        else:
            st.metric("Avg Normalized Length", "N/A")

    # Topic stats if available
    if "dominant_topic" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            num_topics = df["dominant_topic"].nunique()
            st.metric("Topics Detected", num_topics)
        with col2:
            if "drift_detected" in df.columns:
                drift_count = df[df["drift_detected"] == True]["window_id"].nunique() if "window_id" in df.columns else 0
                st.metric("Drift Events", drift_count)

    st.divider()

    # Text length comparison chart
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

    # Record count
    filtered_df = df
    st.metric("Total Records", f"{len(filtered_df):,}")

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
        topic_label = f" | Topic {row['dominant_topic']}" if 'dominant_topic' in row and row.get('dominant_topic', -1) >= 0 else ""
        with st.expander(f"Record {row.get('id', idx)}{topic_label}"):
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
            col1, col2, col3 = st.columns(3)
            with col1:
                if "original_length" in row:
                    st.caption(f"Original length: {row['original_length']:,}")
            with col2:
                if "normalized_length" in row:
                    st.caption(f"Normalized length: {row['normalized_length']:,}")
            with col3:
                if "dominant_topic" in row and row["dominant_topic"] >= 0:
                    st.caption(f"Topic: {row['dominant_topic']}")

            # Show top words if available
            if "top_words" in row:
                words = row["top_words"]
                if words is not None and isinstance(words, (list, np.ndarray)) and len(words) > 0:
                    st.caption(f"Top words: {', '.join(str(w) for w in words[:5])}")


def render_topics_tab(df: pd.DataFrame, has_topics: bool):
    """Render the Topics tab with real or placeholder data."""
    st.header("Topic Modeling")

    if not has_topics or "dominant_topic" not in df.columns:
        st.info(
            """
            **No topic data available**

            Select "Features (with topics)" data source from the sidebar,
            or wait for the modeling service to process data.
            """
        )
        return

    # Topic distribution
    st.subheader("Topic Distribution")

    topic_counts = df["dominant_topic"].value_counts().sort_index()
    topic_df = pd.DataFrame({
        "Topic": [f"Topic {i}" for i in topic_counts.index],
        "Document Count": topic_counts.values
    })

    fig = px.bar(
        topic_df,
        x="Topic",
        y="Document Count",
        title="Documents per Topic",
        color="Document Count",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top words per topic
    st.subheader("Top Words per Topic")

    if "top_words" in df.columns:
        # Get unique topics and their top words
        topic_words = {}
        for _, row in df.iterrows():
            topic_id = row.get("dominant_topic", -1)
            words = row.get("top_words", [])
            # Handle numpy arrays and lists
            if isinstance(words, np.ndarray):
                words = words.tolist()
            if topic_id >= 0 and words and len(words) > 0 and topic_id not in topic_words:
                topic_words[topic_id] = words

        if topic_words:
            cols = st.columns(min(5, max(1, len(topic_words))))
            for i, (topic_id, words) in enumerate(sorted(topic_words.items())):
                with cols[i % len(cols)]:
                    st.markdown(f"**Topic {topic_id}**")
                    if isinstance(words, (list, np.ndarray)):
                        st.caption(", ".join(str(w) for w in words[:7]))
                    else:
                        st.caption(str(words))
        else:
            st.info("No topic words available yet")

    # Topic distribution heatmap
    st.subheader("Topic Probability Distribution")

    if "topic_distribution" in df.columns:
        # Sample for visualization
        sample_df = df.head(50)
        dist_data = []
        for idx, row in sample_df.iterrows():
            dist = row.get("topic_distribution", [])
            # Handle numpy arrays
            if isinstance(dist, np.ndarray):
                dist = dist.tolist()
            if isinstance(dist, list) and len(dist) > 0:
                for topic_idx, prob in enumerate(dist):
                    dist_data.append({
                        "Document": f"Doc {row.get('id', idx)}",
                        "Topic": f"T{topic_idx}",
                        "Probability": float(prob) if prob is not None else 0.0
                    })

        if dist_data:
            dist_df = pd.DataFrame(dist_data)
            fig = px.density_heatmap(
                dist_df,
                x="Topic",
                y="Document",
                z="Probability",
                title="Topic Probabilities per Document (sample)",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic distribution data available")


def render_drift_tab(df: pd.DataFrame, has_topics: bool):
    """Render the Drift Timeline tab with real or placeholder data."""
    st.header("Drift Detection Timeline")

    if not has_topics or "window_id" not in df.columns:
        st.info(
            """
            **No drift data available**

            Select "Features (with topics)" data source from the sidebar,
            or wait for the modeling service to process data.
            """
        )
        return

    # Drift summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        num_windows = df["window_id"].nunique()
        st.metric("Time Windows", num_windows)

    with col2:
        if "drift_detected" in df.columns:
            drift_events = df[df["drift_detected"] == True]["window_id"].nunique()
            st.metric("Drift Events", drift_events)
        else:
            st.metric("Drift Events", "N/A")

    with col3:
        if "js_divergence" in df.columns:
            avg_div = df.groupby("window_id")["js_divergence"].first().mean()
            st.metric("Avg JS Divergence", f"{avg_div:.4f}")
        else:
            st.metric("Avg JS Divergence", "N/A")

    st.divider()

    # JS Divergence over time
    st.subheader("Jensen-Shannon Divergence Over Time")

    if "js_divergence" in df.columns and "window_id" in df.columns:
        window_stats = df.groupby("window_id").agg({
            "js_divergence": "first",
            "drift_detected": "first" if "drift_detected" in df.columns else lambda x: False,
            "drift_magnitude": "first" if "drift_magnitude" in df.columns else lambda x: "none"
        }).reset_index()

        fig = go.Figure()

        # Add JS divergence line
        fig.add_trace(go.Scatter(
            x=window_stats["window_id"],
            y=window_stats["js_divergence"],
            mode="lines+markers",
            name="JS Divergence",
            line=dict(color="blue")
        ))

        # Add threshold line
        fig.add_hline(y=0.15, line_dash="dash", line_color="red",
                      annotation_text="Drift Threshold (0.15)")

        # Mark drift events
        drift_windows = window_stats[window_stats["drift_detected"] == True]
        if not drift_windows.empty:
            fig.add_trace(go.Scatter(
                x=drift_windows["window_id"],
                y=drift_windows["js_divergence"],
                mode="markers",
                name="Drift Detected",
                marker=dict(color="red", size=12, symbol="x")
            ))

        fig.update_layout(
            title="Topic Distribution Drift Over Time",
            xaxis_title="Window ID",
            yaxis_title="JS Divergence",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # Topic distribution per window
    st.subheader("Topic Distribution per Window")

    if "dominant_topic" in df.columns and "window_id" in df.columns:
        # Calculate topic proportions per window
        topic_by_window = df.groupby(["window_id", "dominant_topic"]).size().unstack(fill_value=0)
        topic_by_window = topic_by_window.div(topic_by_window.sum(axis=1), axis=0)

        fig = go.Figure()
        for topic in topic_by_window.columns:
            fig.add_trace(go.Scatter(
                x=topic_by_window.index,
                y=topic_by_window[topic],
                mode="lines+markers",
                name=f"Topic {topic}",
                stackgroup="one"
            ))

        fig.update_layout(
            title="Topic Proportions Over Time Windows",
            xaxis_title="Window ID",
            yaxis_title="Proportion",
            legend_title="Topics"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Drift events table
    if "drift_detected" in df.columns:
        st.subheader("Drift Events Log")
        drift_events = df[df["drift_detected"] == True].groupby("window_id").first().reset_index()

        if not drift_events.empty:
            display_cols = ["window_id", "js_divergence", "drift_magnitude"]
            display_cols = [c for c in display_cols if c in drift_events.columns]
            st.dataframe(drift_events[display_cols], use_container_width=True)
        else:
            st.success("No significant drift detected in the data.")


def main():
    """Main application entry point."""
    st.title("OpenWebText2 Topic Analysis Dashboard")

    # Sidebar
    selected_file, has_topics, sample_size = render_sidebar()

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
        "Topics",
        "Drift Timeline"
    ])

    with tab1:
        render_overview_tab(df)

    with tab2:
        render_samples_tab(df)

    with tab3:
        render_topics_tab(df, has_topics)

    with tab4:
        render_drift_tab(df, has_topics)


if __name__ == "__main__":
    main()
