FROM xprobe/xinference:latest

ENV PYTHONUNBUFFERED=1 \
    XINFERENCE_HOME=/data

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3

COPY scripts/entrypoint.sh /app/entrypoint.sh
COPY scripts/bootstrap.py /app/bootstrap.py

RUN chmod +x /app/entrypoint.sh

EXPOSE 9997

ENTRYPOINT ["/app/entrypoint.sh"]
