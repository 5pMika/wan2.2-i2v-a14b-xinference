# Pin xinference to a fixed version for reproducible builds.
FROM xprobe/xinference:1.14.0

ENV PYTHONUNBUFFERED=1 \
    XINFERENCE_HOME=/data

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3

COPY scripts/bootstrap.py /app/bootstrap.py

EXPOSE 9997

ENTRYPOINT ["python3", "/app/bootstrap.py"]
