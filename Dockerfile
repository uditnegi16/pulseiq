FROM python:3.12-slim

WORKDIR /app

# Chrome/Selenium are not needed in the API image - scraping runs as a
# separate scheduled job, not inside the request path.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "pulseiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
