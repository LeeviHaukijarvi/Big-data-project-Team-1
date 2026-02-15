FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY upload_parquet_to_azure.py .
CMD ["python", "upload_parquet_to_azure.py"]
