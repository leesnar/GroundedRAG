# Bakes in a pre-built vectorstore/ (run `python src/ingest.py` before building)
# rather than re-embedding the corpus on every container start -- see DECISIONS.md.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY vectorstore/ ./vectorstore/
COPY data/SOURCES.md ./data/SOURCES.md

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
# 7860 is the public demo (Streamlit, matches HF Spaces' default Docker SDK
# port); 8000 is the FastAPI service, reachable within the container network
# for programmatic/API use but not published by the Space itself.
EXPOSE 7860 8000

ENTRYPOINT ["/entrypoint.sh"]
