"""
Stream parquet files from HuggingFace directly to Azure Blob Storage.

Usage:
    python scripts/upload_parquet_to_azure.py

Requires:
    pip install azure-storage-blob requests tqdm

Environment variables (or set directly below):
    AZURE_STORAGE_CONNECTION_STRING
    AZURE_CONTAINER_NAME  (default: pipeline-results)
"""
import os
import sys
import time
import requests
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# ── Configuration ────────────────────────────────────────────────────────────

HF_BASE_URL = (
    "https://huggingface.co/datasets/Geralt-Targaryen/openwebtext2"
    "/resolve/filtered"
)
TOTAL_FILES = 27
FILE_PATTERN = "webtext2-{n:05d}-of-00027.parquet"
AZURE_FOLDER  = "RAW"
CHUNK_SIZE    = 4 * 1024 * 1024  # 4 MB chunks — streams, never loads full file
MAX_RETRIES   = 5
CONNECT_TIMEOUT = 30   # seconds to establish connection
READ_TIMEOUT    = None # no timeout between chunks — large files stall HF CDN

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER_NAME    = os.getenv("AZURE_CONTAINER_NAME", "pipeline-results")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_container_client():
    if not CONNECTION_STRING:
        sys.exit(
            "ERROR: AZURE_STORAGE_CONNECTION_STRING is not set.\n"
            "Export it or edit the CONNECTION_STRING variable at the top of this file."
        )
    client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = client.get_container_client(CONTAINER_NAME)
    try:
        container.create_container()
        print(f"Created container: {CONTAINER_NAME}")
    except ResourceExistsError:
        print(f"Using existing container: {CONTAINER_NAME}")
    return container


def blob_exists(container, blob_name: str) -> bool:
    blob = container.get_blob_client(blob_name)
    try:
        blob.get_blob_properties()
        return True
    except Exception:
        return False


LOG_EVERY_MB = 100  # print a progress line every N MB


def _stream_upload(container, url: str, blob_name: str):
    """Single attempt: stream url → Azure blob. Raises on any error."""
    with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
        resp.raise_for_status()
        total_bytes = int(resp.headers.get("content-length", 0))
        total_mb = total_bytes / 1024 / 1024
        blob_client = container.get_blob_client(blob_name)

        uploaded = 0
        next_log_bytes = LOG_EVERY_MB * 1024 * 1024

        def chunked():
            nonlocal uploaded, next_log_bytes
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                uploaded += len(chunk)
                if uploaded >= next_log_bytes:
                    pct = uploaded / total_bytes * 100 if total_bytes else 0
                    print(
                        f"         {uploaded / 1024 / 1024:6.0f} / {total_mb:.0f} MB"
                        f"  ({pct:.0f}%)",
                        flush=True,
                    )
                    next_log_bytes += LOG_EVERY_MB * 1024 * 1024
                yield chunk

        blob_client.upload_blob(chunked(), overwrite=True,
                                length=total_bytes if total_bytes else None)
    return total_bytes


def upload_file(container, filename: str, n: int):
    url       = f"{HF_BASE_URL}/{filename}"
    blob_name = f"{AZURE_FOLDER}/{filename}"

    if blob_exists(container, blob_name):
        print(f"  [{n:02d}/{TOTAL_FILES}] SKIP  {filename} (already in Azure)")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt == 1:
                print(f"  [{n:02d}/{TOTAL_FILES}] START {filename}")
            else:
                print(f"  [{n:02d}/{TOTAL_FILES}] RETRY {filename} (attempt {attempt}/{MAX_RETRIES})")

            total_bytes = _stream_upload(container, url, blob_name)
            size_mb = total_bytes / 1024 / 1024
            print(f"  [{n:02d}/{TOTAL_FILES}] DONE  {blob_name}  ({size_mb:.0f} MB)")
            return

        except Exception as e:
            print(f"  [{n:02d}/{TOTAL_FILES}] ERROR {filename}: {e}")
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt  # 2s, 4s, 8s, 16s
                print(f"  [{n:02d}/{TOTAL_FILES}]       retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {filename}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Uploading {TOTAL_FILES} parquet files → {CONTAINER_NAME}/{AZURE_FOLDER}/\n")
    container = get_container_client()

    failed = []
    for n in range(1, TOTAL_FILES + 1):
        filename = FILE_PATTERN.format(n=n)
        try:
            upload_file(container, filename, n)
        except Exception as e:
            print(f"  [{n:02d}/{TOTAL_FILES}] FAIL  {filename}: {e}")
            failed.append(filename)

    print(f"\nDone. {TOTAL_FILES - len(failed)}/{TOTAL_FILES} files uploaded.")
    if failed:
        print("Failed files (re-run to retry — skips already uploaded):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
