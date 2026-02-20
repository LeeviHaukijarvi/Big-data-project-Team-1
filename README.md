# Big Data Project - OpenWebText2 Topic Analysis Pipeline

A containerized data pipeline for processing the OpenWebText2 dataset (~35 GB, 27 parquet files) with LDA topic modeling and drift detection. Uses Apache Kafka for message passing between processing stages, all orchestrated via Docker Compose.

## Live Demo

**Frontend Dashboard:** https://big-data-project-team-1.onrender.com

The Streamlit dashboard is hosted on Render.com and displays:
- Overview metrics and text length distributions
- Text samples with filtering
- Topic modeling results with top words per topic
- Drift detection timeline with Jensen-Shannon divergence

## Architecture

```
HuggingFace → data-upload → Azure RAW/ → ingestion → Kafka raw-text
  → normalization → Kafka normalized-text → modeling → Kafka modeled-text
  → storage → Azure (RAW/, PROCESSED/, FEATURES/, METADATA/)
```

### Services

| Service | Description |
|---------|-------------|
| **data-upload** | Streams parquet files from HuggingFace to Azure `RAW/` |
| **ingestion** | Downloads from Azure, applies float16→float32 fix, produces to Kafka |
| **normalization** | Text cleaning (lowercase, URL/HTML removal, whitespace normalization) |
| **modeling** | LDA topic modeling with drift detection using scikit-learn |
| **storage** | Batches messages into parquet files, uploads to Azure |
| **frontend** | Streamlit dashboard for visualization (hosted on Render.com) |

### Topic Modeling

The modeling service uses scikit-learn's Latent Dirichlet Allocation (LDA) to extract topics from text:
- Extracts 10 topics by default
- Online learning for incremental updates
- Top 10 words per topic for interpretation

### Drift Detection

Monitors topic distribution changes over time using Jensen-Shannon divergence:
- Sliding window approach (200 messages default)
- Drift magnitude levels: none, low, medium, high
- Threshold-based alerts for significant shifts

## Setup

### Prerequisites

- Docker and Docker Compose
- Azure Storage Account
- Python 3.11+ (for local development)

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=blob
BATCH_SIZE=100
MAX_MESSAGES=1000
NUM_TOPICS=10
```

### Running the Pipeline

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Running Frontend Locally

```bash
docker compose -f docker-compose.frontend.yml up -d
```

Open http://localhost:8501 in your browser.

## Data Schema

### PROCESSED/ Folder
| Column | Type | Description |
|--------|------|-------------|
| id | int | Record identifier |
| text | string | Normalized text |
| original_length | int | Pre-normalization length |
| normalized_length | int | Post-normalization length |

### FEATURES/ Folder
| Column | Type | Description |
|--------|------|-------------|
| id | int | Record identifier |
| dominant_topic | int | Most likely topic (0-9) |
| topic_distribution | array | Probability per topic |
| top_words | array | Top words for dominant topic |
| window_id | int | Drift detection window |
| drift_detected | bool | Whether drift was detected |
| js_divergence | float | Jensen-Shannon divergence score |
| drift_magnitude | string | none/low/medium/high |

## Project Structure

```
├── docker-compose.yml          # Main pipeline orchestration
├── docker-compose.frontend.yml # Separate frontend compose
├── frontend/
│   ├── app.py                  # Streamlit dashboard
│   ├── Dockerfile
│   └── requirements.txt
├── services/
│   ├── shared/                 # Shared config and utilities
│   ├── ingestion/              # Data ingestion service
│   ├── normalization/          # Text normalization service
│   ├── modeling/               # LDA topic modeling + drift detection
│   └── storage/                # Azure storage service
```

## Team

Big Data Project - Team 1
