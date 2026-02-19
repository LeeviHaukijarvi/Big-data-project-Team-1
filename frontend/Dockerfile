FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Render uses PORT env variable
ENV PORT=8501

EXPOSE $PORT

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
