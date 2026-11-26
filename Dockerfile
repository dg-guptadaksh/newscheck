FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEWSCHECK_DB=/var/lib/newscheck/newscheck.sqlite3

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 newscheck \
    && install -d -o newscheck -g newscheck /var/lib/newscheck

WORKDIR /app
COPY --chown=newscheck:newscheck . .
RUN python -m pip install --no-cache-dir -r requirements.txt

USER newscheck
EXPOSE 7860
HEALTHCHECK --start-period=30s --interval=30s --timeout=5s CMD ["python", "-c", "import json, urllib.request; status = json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/status', timeout=3)); assert status['model']['mode'] == 'encoder' and status['retriever'] == 'MiniLM / FAISS'"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
