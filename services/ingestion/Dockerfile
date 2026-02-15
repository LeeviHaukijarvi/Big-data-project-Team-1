FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Note: shared/ module is mounted as a volume at /app/shared in docker-compose.yml

CMD ["python", "ingest.py"]
