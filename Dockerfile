# Pin xinference to a fixed version for reproducible builds.
FROM xprobe/xinference:v1.14.0

ENV PYTHONUNBUFFERED=1 \
    XINFERENCE_HOME=/data

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    boto3 \
    requests \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    httpx \
    runpod

COPY scripts/bootstrap.py /app/bootstrap.py
COPY scripts/rp_handler.py /app/rp_handler.py

EXPOSE 9997

CMD ["python", "-u", "/app/rp_handler.py"]
