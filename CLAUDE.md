# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A containerized data pipeline for processing the OpenWebText2 dataset (~35 GB, 27 parquet files). Uses Apache Kafka for message passing between 4 processing stages, all orchestrated via Docker Compose.

## Commands

### Running the Pipeline
```bash
# Start all services (uploads data, starts Kafka, runs pipeline)
docker compose up -d

# Rebuild a specific service after code changes
docker compose build ingestion

# Restart a specific service
docker compose up -d storage

# Stop everything
docker compose down

# View service logs
docker compose logs -f ingestion
docker compose logs -f storage
```

### Kafka Commands
```bash
# List topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check message counts per topic
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic raw-text
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic normalized-text
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic modeled-text
```

## Architecture

### Pipeline Flow
```
HuggingFace → data-upload → Azure RAW/ → ingestion → Kafka raw-text
  → normalization → Kafka normalized-text → modeling → Kafka modeled-text
  → storage → Azure (RAW/, PROCESSED/, FEATURES/, METADATA/)
```

### Services (docker-compose.yml)
- **data-upload**: Streams parquet files from HuggingFace to Azure `RAW/`. Runs first, skips existing files.
- **ingestion**: Downloads from Azure `RAW/`, applies float16→float32 fix, produces to `raw-text` topic.
- **normalization**: Cleans text (lowercase, URL/HTML removal, whitespace), produces to `normalized-text`.
- **modeling**: Currently a passthrough stub. Phase 2 will add PySpark LDA topic modeling.
- **storage**: Batches messages into parquet files, uploads to 4 Azure folders.

### Shared Code (`services/shared/`)
- `config.py`: Centralized config for Kafka broker, topics, Azure credentials, batch size.
- `kafka_utils.py`: Producer/consumer factory with retry logic for Docker startup delays.

### Key Configuration
| Setting | Location | Default |
|---------|----------|---------|
| `MAX_MESSAGES` | docker-compose.yml line 79 | 1000 |
| `BATCH_SIZE` | services/shared/config.py | 100 |
| `KAFKA_BROKER` | docker-compose.yml | localhost:29092 |

### Data Schema
| Column | Type | Description |
|--------|------|-------------|
| `title` | string | Article title |
| `text` | string | Article text (normalized in PROCESSED) |
| `score` | float | Content quality score |
| `reddit_scores` | list[int] | Reddit upvote scores |

### Azure Output Structure
| Folder | Format | Content |
|--------|--------|---------|
| `RAW/` | parquet | id, title, text, score, reddit_scores |
| `PROCESSED/` | parquet | id, text, original_length, normalized_length, score |
| `FEATURES/` | parquet | id, model_version, topics |
| `METADATA/` | JSON | batch_num, timestamp, record_count, id_range |

## Environment Setup

Copy `.env.example` to `.env` and set:
```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=blob
```

## Notebooks

The Jupyter notebooks contain the original prototypes that were migrated to services:
- `data_preparation.ipynb` → ingestion service (float16→float32 fix)
- `normalization.ipynb` → normalization service (text cleaning)
- `model.ipynb` → modeling service (PySpark LDA, Phase 2)

## Future Phases

- **Phase 2**: Replace modeling stub with PySpark LDA topic modeling
- **Phase 3**: Drift detection - split data into time windows, detect topic shifts
- **Phase 4**: Web frontend with topic timeline visualization
