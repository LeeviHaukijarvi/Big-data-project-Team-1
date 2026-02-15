"""
Storage service - consumes modeled text from Kafka and persists to Azure Blob Storage.

Writes parquet files to RAW/PROCESSED/FEATURES folders and JSON for METADATA.
Supports local fallback mode when Azure credentials are not configured.
"""
import sys
sys.path.insert(0, "/app")

from shared.config import KAFKA_BROKER, KAFKA_TOPICS, AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME, BATCH_SIZE
from shared.kafka_utils import create_consumer
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
import io
import json
import logging
import datetime
import os
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def create_blob_client():
    """
    Create Azure Blob Service client.

    Returns None if Azure credentials are not configured (fallback mode).
    """
    if not AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_CONNECTION_STRING.strip() == "":
        logger.warning("Azure connection string not configured. Using local fallback storage.")
        return None

    try:
        blob_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        logger.info(f"Connected to Azure Blob Storage, container: {AZURE_CONTAINER_NAME}")
        return blob_client
    except Exception as e:
        logger.error(f"Failed to create Azure Blob client: {e}")
        logger.warning("Falling back to local storage")
        return None


def split_batch(data: list, batch_num: int, timestamp: str) -> dict:
    """
    Split a batch of pipeline messages into the 4 Azure folder structures.

    Returns a dict with keys: raw, processed, features, metadata
    """
    raw = []
    processed = []
    features = []

    for msg in data:
        raw.append({
            "id": msg.get("id"),
            "title": msg.get("title", ""),
            "text": msg.get("text", ""),
            "score": msg.get("score", 0.0),
            "reddit_scores": json.dumps(msg.get("reddit_scores", [])),
        })
        processed.append({
            "id": msg.get("id"),
            "text": msg.get("text", ""),
            "original_length": msg.get("original_length"),
            "normalized_length": msg.get("normalized_length"),
            "score": msg.get("score", 0.0),
        })
        features.append({
            "id": msg.get("id"),
            "model_version": msg.get("model_version"),
            "topics": json.dumps(msg.get("topics", [])),
        })

    metadata = {
        "batch_num": batch_num,
        "timestamp": timestamp,
        "record_count": len(data),
        "id_range_min": raw[0]["id"] if raw else None,
        "id_range_max": raw[-1]["id"] if raw else None,
    }

    return {"raw": raw, "processed": processed, "features": features, "metadata": metadata}


def records_to_parquet_bytes(records: list) -> bytes:
    """Convert a list of dicts to parquet bytes in memory."""
    table = pa.Table.from_pylist(records)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def upload_to_azure(blob_client, data: list, batch_num: int):
    """
    Upload batch data to Azure Blob Storage using the RAW/PROCESSED/FEATURES/METADATA structure.
    Data folders use parquet format; METADATA uses JSON.
    """
    try:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        parts = split_batch(data, batch_num, timestamp)
        container_client = blob_client.get_container_client(AZURE_CONTAINER_NAME)
        base_name = f"batch_{batch_num:06d}_{timestamp}"

        # Parquet folders
        for folder in ("RAW", "PROCESSED", "FEATURES"):
            blob_name = f"{folder}/{base_name}.parquet"
            parquet_data = records_to_parquet_bytes(parts[folder.lower()])
            container_client.upload_blob(
                name=blob_name,
                data=parquet_data,
                overwrite=True
            )
            logger.info(f"Uploaded to Azure: {blob_name}")

        # Metadata as JSON (small, human-readable)
        meta_blob = f"METADATA/{base_name}_meta.json"
        container_client.upload_blob(
            name=meta_blob,
            data=json.dumps(parts["metadata"], indent=2),
            overwrite=True
        )
        logger.info(f"Uploaded to Azure: {meta_blob}")

        return base_name

    except Exception as e:
        logger.error(f"Failed to upload to Azure: {e}", exc_info=True)
        raise


def fallback_local_storage(data: list, batch_num: int):
    """
    Store batch data locally using the same RAW/PROCESSED/FEATURES/METADATA structure.
    Data folders use parquet format; METADATA uses JSON.
    """
    try:
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        parts = split_batch(data, batch_num, timestamp)
        base_name = f"batch_{batch_num:06d}_{timestamp}"

        # Parquet folders
        for folder in ("RAW", "PROCESSED", "FEATURES"):
            folder_path = os.path.join("/app/output", folder)
            os.makedirs(folder_path, exist_ok=True)
            filepath = os.path.join(folder_path, f"{base_name}.parquet")
            table = pa.Table.from_pylist(parts[folder.lower()])
            pq.write_table(table, filepath)
            logger.info(f"Stored locally: {filepath}")

        # Metadata as JSON
        meta_path = os.path.join("/app/output", "METADATA")
        os.makedirs(meta_path, exist_ok=True)
        meta_file = os.path.join(meta_path, f"{base_name}_meta.json")
        with open(meta_file, 'w') as f:
            json.dump(parts["metadata"], f, indent=2)
        logger.info(f"Stored locally: {meta_file}")

        return base_name

    except Exception as e:
        logger.error(f"Failed to store locally: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting storage service...")

        # Create consumer
        consumer = create_consumer(KAFKA_TOPICS["modeled"], "storage-group")

        # Try to create blob client
        blob_client = create_blob_client()

        # Ensure container exists if using Azure
        if blob_client:
            try:
                container_client = blob_client.get_container_client(AZURE_CONTAINER_NAME)
                container_client.create_container()
                logger.info(f"Created container: {AZURE_CONTAINER_NAME}")
            except ResourceExistsError:
                logger.info(f"Container already exists: {AZURE_CONTAINER_NAME}")
            except Exception as e:
                logger.warning(f"Could not verify/create container: {e}")

        # Batch processing
        batch_buffer = []
        batch_num = 1
        processed_count = 0

        logger.info(f"Consuming from {KAFKA_TOPICS['modeled']}, batch size: {BATCH_SIZE}")

        for message in consumer:
            try:
                msg = message.value
                batch_buffer.append(msg)

                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} messages")

                if len(batch_buffer) >= BATCH_SIZE:
                    if blob_client:
                        try:
                            upload_to_azure(blob_client, batch_buffer, batch_num)
                        except Exception as upload_err:
                            logger.warning(f"Azure upload failed, using local fallback: {upload_err}")
                            fallback_local_storage(batch_buffer, batch_num)
                    else:
                        fallback_local_storage(batch_buffer, batch_num)

                    batch_buffer = []
                    batch_num += 1

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

        # Flush remaining messages on shutdown
        if batch_buffer:
            logger.info(f"Flushing remaining {len(batch_buffer)} messages...")
            if blob_client:
                try:
                    upload_to_azure(blob_client, batch_buffer, batch_num)
                except Exception as upload_err:
                    logger.warning(f"Azure upload failed, using local fallback: {upload_err}")
                    fallback_local_storage(batch_buffer, batch_num)
            else:
                fallback_local_storage(batch_buffer, batch_num)

        logger.info(f"Storage service shutting down. Processed {processed_count} messages total.")
        consumer.close()

    except KeyboardInterrupt:
        logger.info("Storage service interrupted by user")
        if batch_buffer:
            logger.info(f"Flushing remaining {len(batch_buffer)} messages...")
            if blob_client:
                try:
                    upload_to_azure(blob_client, batch_buffer, batch_num)
                except Exception as upload_err:
                    logger.warning(f"Azure upload failed, using local fallback: {upload_err}")
                    fallback_local_storage(batch_buffer, batch_num)
            else:
                fallback_local_storage(batch_buffer, batch_num)
    except Exception as e:
        logger.error(f"Fatal error in storage service: {e}", exc_info=True)
        raise
