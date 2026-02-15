"""
Data ingestion service for OpenWebText2 dataset.
Reads parquet files from Azure Blob Storage RAW/ folder and produces
messages to Kafka raw-text topic.
"""
import logging
import os
import sys
import tempfile

# Add /app to path to ensure shared module is importable
sys.path.insert(0, "/app")

import pyarrow as pa
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient
from shared.config import (
    KAFKA_BROKER, KAFKA_TOPICS, BATCH_SIZE,
    AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME
)
from shared.kafka_utils import create_producer

# Configure logging — suppress noisy Azure HTTP logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Maximum messages to ingest (for development/testing)
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "1000"))

# Prefix for raw parquet files in Azure
RAW_PREFIX = "RAW/"


def iter_azure_parquet_rows(connection_string: str, container_name: str):
    """
    List all blobs under RAW/ prefix, download each parquet file,
    and yield rows as dicts with keys: title, text.
    Files are processed in sorted order for deterministic runs.
    """
    client = BlobServiceClient.from_connection_string(connection_string)
    container_client = client.get_container_client(container_name)

    blobs = sorted(
        [b.name for b in container_client.list_blobs(name_starts_with=RAW_PREFIX)
         if b.name.endswith(".parquet")]
    )

    logger.info(f"Found {len(blobs)} parquet file(s) under {RAW_PREFIX}")

    for blob_name in blobs:
        blob_client = container_client.get_blob_client(blob_name)
        props = blob_client.get_blob_properties()
        size_mb = props.size / 1024 / 1024
        logger.info(f"Downloading {blob_name} ({size_mb:.0f} MB) to temp file...")

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
            stream = blob_client.download_blob(max_concurrency=4)
            downloaded = 0
            for chunk in stream.chunks():
                tmp.write(chunk)
                downloaded += len(chunk)
                if downloaded % (100 * 1024 * 1024) < len(chunk):
                    logger.info(f"  {downloaded / 1024 / 1024:.0f} / {size_mb:.0f} MB ({downloaded * 100 / props.size:.0f}%)")
            tmp.flush()
            logger.info(f"  Download complete: {downloaded / 1024 / 1024:.0f} MB")

            table = pq.read_table(tmp.name)

            # Apply data_preparation logic: cast float16 → float32
            new_fields = []
            for field in table.schema:
                if field.type == pa.float16():
                    new_fields.append(pa.field(field.name, pa.float32()))
                else:
                    new_fields.append(field)
            table = table.cast(pa.schema(new_fields))

            row_count = table.num_rows
            logger.info(f"{blob_name}: {row_count} rows, columns: {table.column_names}")

            text_col = table.column("text")
            title_col = table.column("title") if "title" in table.column_names else None
            score_col = table.column("score") if "score" in table.column_names else None
            reddit_scores_col = table.column("reddit_scores") if "reddit_scores" in table.column_names else None

            for i in range(row_count):
                row = {
                    "title": str(title_col[i]) if title_col else "",
                    "text": str(text_col[i]),
                }
                if score_col is not None:
                    val = score_col[i].as_py()
                    row["score"] = val if val is not None else 0.0
                if reddit_scores_col is not None:
                    val = reddit_scores_col[i].as_py()
                    row["reddit_scores"] = val if val is not None else []
                yield row


def main():
    """Main ingestion flow."""
    logger.info("Starting ingestion service (Azure Blob Storage source)")
    logger.info(f"Container: {AZURE_CONTAINER_NAME}, prefix: {RAW_PREFIX}")
    logger.info(f"Target topic: {KAFKA_TOPICS['raw']}")
    logger.info(f"Max messages: {MAX_MESSAGES}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    if not AZURE_STORAGE_CONNECTION_STRING:
        logger.error("AZURE_STORAGE_CONNECTION_STRING is not set. Cannot read from Azure.")
        sys.exit(1)

    producer = create_producer()

    try:
        idx = 0
        for item in iter_azure_parquet_rows(
            AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME
        ):
            message = {
                "id": idx,
                "title": item["title"],
                "text": item["text"],
                "score": item.get("score", 0.0),
                "reddit_scores": item.get("reddit_scores", []),
            }

            producer.send(KAFKA_TOPICS["raw"], value=message)
            idx += 1

            if idx % BATCH_SIZE == 0:
                logger.info(f"Produced {idx} messages to {KAFKA_TOPICS['raw']}")

            if idx >= MAX_MESSAGES:
                logger.info(f"Reached MAX_MESSAGES limit ({MAX_MESSAGES}), stopping")
                break

        producer.flush()
        logger.info(f"Ingestion complete. Total messages produced: {idx}")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise
    finally:
        producer.close()
        logger.info("Producer closed")


if __name__ == "__main__":
    main()
