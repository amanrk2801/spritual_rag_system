FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/backend
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
RUN useradd -m appuser && mkdir -p /srv/data && chown -R appuser /srv
USER appuser
ENV DATA_DIR=/srv/data QDRANT_PATH=/srv/data/qdrant
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
# Multiple workers are safe only with a Qdrant server (QDRANT_URL); caches/rate limits are per worker.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2} --proxy-headers"]
