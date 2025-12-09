# wan2.2-i2v-a14b on Xinference (Docker)

Containerized Xinference setup that auto-launches the Wan image-to-video model (default `wan2.2-i2v-a14b`) and exposes the OpenAI-compatible API.

## Prerequisites
- Docker & Docker Compose v2
- NVIDIA driver + `nvidia-container-toolkit`
- GPU with ≥20GB VRAM recommended for 14B video models

## Quick start (no Compose)
Build:
```bash
docker build -t wan2-2-i2v-a14b-xinference .
```

Run (GPU):
```bash
docker run --rm -d \
  --name xinference \
  --gpus all \
  -p 9997:9997 \
  -v $(pwd)/data:/data \
  -v $(pwd)/cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/cache/modelscope:/root/.cache/modelscope \
  -e XINFERENCE_PORT=9997 \
  -e XINFERENCE_HOST=0.0.0.0 \
  -e ENABLE_S3_MODEL=1 \
  -e MODEL_S3_BUCKET=noty3emlmi \
  -e MODEL_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io \
  -e MODEL_S3_PREFIX=wan2.2-i2v-a14b/ \
  -e MODEL_LOCAL_PATH=/data/models/wan2.2-i2v-a14b \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e VIDEO_MODEL_NAME=wan2.2-i2v-a14b \
  -e VIDEO_MODEL_PATH=/data/models/wan2.2-i2v-a14b \
  -e VIDEO_FALLBACK_MODEL=Wan2.1-i2v-14B-480p \
  wan2-2-i2v-a14b-xinference
```

Logs:
```bash
docker logs -f xinference
```

## Configuration
- Tune via environment variables.
- Persistent caches and models live in `./data` and `./cache/*` (mounted into the container).
- Key environment variables (defaults in compose):
  - `VIDEO_MODEL_NAME` (default `wan2.2-i2v-a14b`; use `Wan2.1-i2v-14B-480p` or `Wan2.1-i2v-14B-720p` if 2.2 is unavailable)
  - `VIDEO_FALLBACK_MODEL` (default `Wan2.1-i2v-14B-480p`, used if the primary launch fails)
  - `VIDEO_LAYERWISE_CAST`, `VIDEO_CPU_OFFLOAD`, `VIDEO_GROUP_OFFLOAD`, `VIDEO_USE_STREAM` (memory/perf knobs)
  - `VIDEO_MODEL_ENGINE`, `VIDEO_MODEL_FORMAT`, `VIDEO_MODEL_PATH`, `VIDEO_SIZE_IN_BILLIONS`, `VIDEO_QUANTIZATION` (custom launches)
  - `AUTO_LAUNCH_MODEL` (set `0` to skip auto-launch)
- Port mapping defaults to `9997` → `9997`.

## RunPod S3 network volume
The bootstrap step verifies and syncs model files from S3 before launching:
- Set S3 variables (examples align with RunPod network volume gateway):
  - `ENABLE_S3_MODEL=1`
  - `MODEL_S3_BUCKET` (default `noty3emlmi`)
  - `MODEL_S3_ENDPOINT` (default `https://s3api-eu-ro-1.runpod.io`)
  - `MODEL_S3_PREFIX` (default `wan2.2-i2v-a14b/`)
  - `MODEL_LOCAL_PATH` (defaults to `/data/models/wan2.2-i2v-a14b`)
  - Optional: `MODEL_REQUIRE_KEYS` (comma-separated relative keys to assert presence), `MODEL_SKIP_EXISTING=1` to avoid re-downloading existing files.
- Ensure AWS credentials are passed (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `MODEL_S3_REGION`).
- `VIDEO_MODEL_PATH` defaults to `/data/models/wan2.2-i2v-a14b` so Xinference loads from the synced local copy.

## How it works
- `Dockerfile` wraps `xprobe/xinference:latest` and adds a thin entrypoint.
- `scripts/entrypoint.sh` starts `xinference-local` and runs `scripts/bootstrap.py`.
- `scripts/bootstrap.py` waits for the API then calls `client.launch_model(...)` with env-driven params to preload the video model.

## Usage
1) List running models and grab the UID:
```bash
curl -s http://localhost:9997/v1/models | jq
```

2) Image → video (multipart):
```bash
curl -X POST \
  "http://localhost:9997/v1/video/generations/image" \
  -F model=<MODEL_UID> \
  -F image=@sample.jpg \
  -F prompt="a slow camera pan"
```

3) Text → video:
```bash
curl -X POST \
  "http://localhost:9997/v1/video/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_UID>",
    "prompt": "a sunset timelapse over mountains"
  }'
```

## Custom notes
- For RunPod or other hosts, bind your persistent volume to `./data` and `./cache/*` so downloads survive container restarts.
- To skip auto-launch and manage models manually, set `AUTO_LAUNCH_MODEL=0` and use `xinference launch ...` via `docker exec -it xinference bash`.
