FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0" \
    && pip install "chronos-forecasting==2.2.2" "fastapi>=0.115,<1" "uvicorn>=0.30,<1"

COPY app.py .

EXPOSE 8081

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
