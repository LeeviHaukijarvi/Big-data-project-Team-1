# Pipeline Summary

## What We Built

A containerized data pipeline that processes the OpenWebText2 dataset through 4 stages, all orchestrated with Apache Kafka and running in Docker.

## Services (docker-compose)

| Service | What it does |
|---|---|
| **data-upload** | Streams 27 parquet files (~35 GB) from HuggingFace to Azure `RAW/` folder. Runs first, skips files already uploaded. |
| **ingestion** | Downloads parquet files from Azure `RAW/`, applies float16 to float32 fix (from `data_preparation.ipynb`), produces messages with all columns (`title`, `text`, `score`, `reddit_scores`) to Kafka `raw-text` topic. |
| **normalization** | Consumes `raw-text`, cleans text (lowercase, remove URLs/HTML/special chars), passes through `score` and `reddit_scores`, produces to `normalized-text`. |
| **modeling** | Currently a stub (passthrough). Consumes `normalized-text`, produces to `modeled-text`. |
| **storage** | Consumes `modeled-text`, writes parquet files to Azure in 4 folders: `RAW/`, `PROCESSED/`, `FEATURES/`, `METADATA/` (metadata as JSON). |

## How to Run

### 1. Create `.env` file

Copy from `.env.example` and fill in your Azure credentials:

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=bigdatateam1;AccountKey=...;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=blob
```

### 2. Start everything

```bash
docker compose up -d
```

This will:
- Upload parquet files to Azure (if not already there)
- Start Kafka + Zookeeper
- Run ingestion, normalization, modeling, storage

### 3. Check progress

```bash
# Watch ingestion downloading from Azure and producing messages
docker compose logs -f ingestion

# Watch storage writing parquet files to Azure
docker compose logs -f storage

# Check Kafka message counts
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic raw-text
```

## Key Configuration

| Setting | Location | Default | Description |
|---|---|---|---|
| `MAX_MESSAGES` | `docker-compose.yml` line 79 | `1000` | How many messages to ingest (increase for full runs) |
| `BATCH_SIZE` | `services/shared/config.py` | `100` | Messages per storage batch file |
| `KAFKA_BROKER` | `docker-compose.yml` | `localhost:29092` | Kafka broker address (host networking) |

## Data Schema

Matches `data_preparation.ipynb` output:

| Column | Type | Description |
|---|---|---|
| `title` | string | Article title |
| `text` | string | Article text (normalized in PROCESSED) |
| `score` | float | Content quality score |
| `reddit_scores` | list[int] | Reddit upvote scores |

## Azure Blob Output Structure

| Folder | Format | Content |
|---|---|---|
| `RAW/` | `.parquet` | id, title, text, score, reddit_scores |
| `PROCESSED/` | `.parquet` | id, text, original_length, normalized_length, score |
| `FEATURES/` | `.parquet` | id, model_version, topics |
| `METADATA/` | `.json` | batch_num, timestamp, record_count, id_range |

## Notebooks to Pipeline Mapping

| Notebook | Pipeline Service | Logic Used |
|---|---|---|
| `data_preparation.ipynb` | **ingestion** | float16 to float32 fix, all columns preserved |
| `normalization.ipynb` | **normalization** | Text cleaning (lowercase, URL/HTML removal) |
| `model.ipynb` | **modeling** | Stub now, PySpark LDA in Phase 2 |

## What's Next

- **Phase 2: Topic Modeling** - Replace modeling stub
- **Phase 3: Drift Detection** - Split data into time windows, detect topic shifts
- **Phase 4: Visualization** - Web frontend with topic timeline view

## Useful Commands

```bash
# Rebuild a specific service after code changes
docker compose build ingestion

# Restart a service
docker compose up -d storage

# Stop everything
docker compose down

# View all service statuses
docker compose ps

# Check what Kafka topics exist
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check messages per topic
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic raw-text
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic normalized-text
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic modeled-text
```
