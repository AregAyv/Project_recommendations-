FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV INTERNAL_API_PORT=9000
ENV API_BASE_URL=http://127.0.0.1:9000
ENV ARTIFACTS_PATH=/app/model_artifacts/model_artifacts.joblib

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir "setuptools<60" wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY steam_client.py .
COPY streamlit_app.py .
COPY model_artifacts/ ./model_artifacts/

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8080"]

