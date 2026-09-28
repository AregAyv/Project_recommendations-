# Python 3.11 -- matches the venv where lightfm's old build actually works
FROM python:3.11-slim

WORKDIR /app

# lightfm's setup.py needs a C compiler and an older setuptools to build cleanly
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<60" wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

# only what the API needs to run -- no training code, no raw dataset
COPY app.py .
COPY steam_client.py .
COPY model_artifacts/ model_artifacts/

ENV ARTIFACTS_PATH=model_artifacts/model_artifacts.joblib

# Cloud Run injects PORT; default to 8080 for local runs
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
