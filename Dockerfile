FROM python:3.11-slim

WORKDIR /app

COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/base.txt \
 && pip install --no-cache-dir -r requirements/storage.txt

COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "pulseiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
