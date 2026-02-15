"""
Centralized configuration for the data pipeline.
Provides Kafka broker settings, topic names, Azure storage config, and batch parameters.
"""
import os

# Kafka configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")

# Topic name mappings for each pipeline stage
KAFKA_TOPICS = {
    "raw": "raw-text",
    "normalized": "normalized-text",
    "modeled": "modeled-text",
    "storage": "storage-results"
}

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "pipeline-results")

# Processing batch size
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Data source - HuggingFace dataset ID
DATA_SOURCE = os.getenv("DATA_SOURCE", "Skylion007/openwebtext")
