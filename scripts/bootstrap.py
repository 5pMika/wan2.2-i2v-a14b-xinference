import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import requests
from botocore.exceptions import ClientError
from xinference.client import Client


LOG_PATH = (
    r"c:\Users\mikas\OneDrive\Projects\PROJEKTIT\runpod\wan2.2-i2v-a14b-xinference\.cursor\debug.log"
)
SESSION_ID = "debug-session"


def dbg_log(*, hypothesis_id: str, location: str, message: str, data: Optional[dict] = None, run_id: str = "run1") -> None:
    """Append a small NDJSON log line for debug mode."""
    payload = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        #region agent log
        with open(LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
        #endregion
    except Exception:
        # Best-effort logging; ignore failures to keep bootstrap running
        pass


def parse_bool(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "t", "yes", "y", "on"}


def configure_xinference_home() -> str:
    """Mirror entrypoint.sh logic to prefer the RunPod volume for cache/state."""
    home = os.getenv("XINFERENCE_HOME")
    if os.path.isdir("/runpod-volume"):
        if not home:
            home = "/runpod-volume/xinference"
        elif home.startswith("/data"):
            home = home.replace("/data", "/runpod-volume", 1)
    if not home:
        home = "/data"
    os.environ["XINFERENCE_HOME"] = home
    Path(home).mkdir(parents=True, exist_ok=True)
    return home


def start_server(host: str, port: str, log_level: str) -> subprocess.Popen:
    cmd = ["xinference-local", "-H", host, "-p", str(port), "--log-level", log_level]
    print(f"[bootstrap] starting xinference-local: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=os.environ.copy())


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if not proc or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        # Best effort shutdown; avoid masking the original error.
        pass


def wait_for_server(endpoint: str, timeout: int = 300, interval: int = 3) -> None:
    deadline = time.time() + timeout
    url = endpoint.rstrip("/") + "/v1/models"
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                print(f"[bootstrap] xinference is ready ({response.status_code})", flush=True)
                return
        except Exception as exc:  # noqa: BLE001
            print(f"[bootstrap] waiting for xinference: {exc}", flush=True)
        time.sleep(interval)
    raise RuntimeError(f"Timed out waiting for Xinference at {endpoint}")


def build_launch_kwargs() -> Dict[str, object]:
    launch_kwargs: Dict[str, object] = {
        "model_name": os.getenv("VIDEO_MODEL_NAME", "Wan2.1-i2v-14B-480p"),
        "model_type": os.getenv("VIDEO_MODEL_TYPE", "video"),
    }

    optional_fields = {
        "VIDEO_MODEL_ENGINE": "model_engine",
        "VIDEO_MODEL_FORMAT": "model_format",
        "VIDEO_MODEL_UID": "model_uid",
        "VIDEO_MODEL_PATH": "model_path",
        "VIDEO_SIZE_IN_BILLIONS": "size_in_billions",
        "VIDEO_QUANTIZATION": "quantization",
    }
    for env_key, arg_key in optional_fields.items():
        value = os.getenv(env_key)
        if value:
            launch_kwargs[arg_key] = value

    bool_fields = {
        "VIDEO_LAYERWISE_CAST": "layerwise_cast",
        "VIDEO_CPU_OFFLOAD": "cpu_offload",
        "VIDEO_GROUP_OFFLOAD": "group_offload",
        "VIDEO_USE_STREAM": "use_stream",
    }
    for env_key, arg_key in bool_fields.items():
        raw_value = os.getenv(env_key)
        if raw_value is not None:
            launch_kwargs[arg_key] = parse_bool(raw_value)

    return launch_kwargs


def ensure_s3_model() -> None:
    if not parse_bool(os.getenv("ENABLE_S3_MODEL", "1")):
        print("[bootstrap] S3 model sync disabled; skipping", flush=True)
        return

    bucket = os.getenv("MODEL_S3_BUCKET")
    prefix = os.getenv("MODEL_S3_PREFIX", "").lstrip("/")
    endpoint = os.getenv("MODEL_S3_ENDPOINT")
    region = os.getenv("MODEL_S3_REGION", "eu-ro-1")
    required_keys_env = os.getenv("MODEL_REQUIRE_KEYS", "")
    required_keys: List[str] = [k.strip() for k in required_keys_env.split(",") if k.strip()]
    # Default to RunPod volume if mounted; fall back to container disk.
    default_local_path = "/runpod-volume/models/wan2.2-i2v-a14b" if os.path.isdir("/runpod-volume") else "/data/models/wan2.2-i2v-a14b"
    env_local_path = os.getenv("MODEL_LOCAL_PATH")
    if env_local_path:
        # If someone set a /data path but the RunPod volume exists, prefer the volume to avoid filling the root disk.
        if env_local_path.startswith("/data/") and os.path.isdir("/runpod-volume"):
            local_path = env_local_path.replace("/data", "/runpod-volume", 1)
        else:
            local_path = env_local_path
    else:
        local_path = default_local_path
    skip_existing = parse_bool(os.getenv("MODEL_SKIP_EXISTING", "1"))

    if not bucket or not endpoint:
        raise RuntimeError("S3 model sync is enabled but MODEL_S3_BUCKET or MODEL_S3_ENDPOINT is missing.")

    normalized_prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
    dbg_log(
        hypothesis_id="H1",
        location="bootstrap.py:ensure_s3_model:entry",
        message="enter ensure_s3_model",
        data={
            "bucket": bucket,
            "prefix": normalized_prefix,
            "region": region,
            "endpoint": endpoint,
            "required_keys_count": len(required_keys),
            "skip_existing": skip_existing,
        },
    )
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region,
    )
    s3 = session.client("s3", endpoint_url=endpoint)

    def full_key(key: str) -> str:
        return key if key.startswith(normalized_prefix) or not normalized_prefix else normalized_prefix + key.lstrip("/")

    if required_keys:
        missing = []
        for key in required_keys:
            key_to_check = full_key(key)
            try:
                s3.head_object(Bucket=bucket, Key=key_to_check)
            except ClientError as exc:  # noqa: PERF203
                if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                    missing.append(key_to_check)
                else:
                    raise
        if missing:
            dbg_log(
                hypothesis_id="H2",
                location="bootstrap.py:ensure_s3_model:required_keys",
                message="required keys missing",
                data={"missing_count": len(missing)},
            )
            raise RuntimeError(f"S3 model check failed; missing keys: {missing}")

    paginator = s3.get_paginator("list_objects_v2")
    keys_to_fetch = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(normalized_prefix) :].lstrip("/") if normalized_prefix else key
                keys_to_fetch.append((key, rel))
    except ClientError as exc:  # noqa: PERF203
        dbg_log(
            hypothesis_id="H3",
            location="bootstrap.py:ensure_s3_model:list_objects",
            message="list_objects_v2 failed",
            data={"error": str(exc)},
        )
        raise RuntimeError(f"S3 model check failed while listing objects: {exc}")

    if not keys_to_fetch:
        dbg_log(
            hypothesis_id="H1",
            location="bootstrap.py:ensure_s3_model:no_objects",
            message="no objects found for prefix",
            data={"bucket": bucket, "prefix": normalized_prefix},
        )
        raise RuntimeError(
            f"S3 model check failed; no objects found at prefix '{normalized_prefix}' in bucket '{bucket}'."
        )

    dbg_log(
        hypothesis_id="H4",
        location="bootstrap.py:ensure_s3_model:objects_found",
        message="objects listed",
        data={"count": len(keys_to_fetch)},
    )

    for key, rel in keys_to_fetch:
        target = os.path.join(local_path, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if skip_existing and os.path.exists(target):
            continue
        print(f"[bootstrap] downloading {key} -> {target}", flush=True)
        s3.download_file(bucket, key, target)

    dbg_log(
        hypothesis_id="H4",
        location="bootstrap.py:ensure_s3_model:completed",
        message="s3 sync complete",
        data={"downloaded": len(keys_to_fetch)},
    )
    print(f"[bootstrap] S3 model sync complete -> {local_path}", flush=True)


def launch_models(endpoint: str) -> None:
    effective_endpoint = endpoint or os.getenv("XINFERENCE_ENDPOINT", f"http://127.0.0.1:{os.getenv('XINFERENCE_PORT', '9997')}")

    ensure_s3_model()
    wait_for_server(effective_endpoint)

    client = Client(effective_endpoint)
    launch_kwargs = build_launch_kwargs()
    fallback_model = os.getenv("VIDEO_FALLBACK_MODEL")

    def try_launch(kwargs: Dict[str, object]) -> str:
        return client.launch_model(**kwargs)

    try:
        uid = try_launch(launch_kwargs)
        print(f"[bootstrap] launched model '{launch_kwargs['model_name']}' as {uid}", flush=True)
        return
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "already launched" in message.lower() or "already exists" in message.lower():
            print(f"[bootstrap] model already running: {message}", flush=True)
            return
        if fallback_model and fallback_model != launch_kwargs["model_name"]:
            print(
                f"[bootstrap] primary model failed ({message}); attempting fallback '{fallback_model}'",
                flush=True,
            )
            launch_kwargs["model_name"] = fallback_model
            uid = try_launch(launch_kwargs)
            print(f"[bootstrap] launched fallback model '{fallback_model}' as {uid}", flush=True)
            return
        raise


def main() -> None:
    host = os.getenv("XINFERENCE_HOST", "0.0.0.0")
    port = os.getenv("XINFERENCE_PORT", "9997")
    log_level = os.getenv("XINFERENCE_LOG_LEVEL", "info")
    auto_launch = parse_bool(os.getenv("AUTO_LAUNCH_MODEL", "1"))
    endpoint = os.getenv("XINFERENCE_ENDPOINT", f"http://127.0.0.1:{port}")

    configure_xinference_home()
    server_proc = start_server(host, port, log_level)

    def _handle_signal(signum, frame):  # noqa: ANN001, D401
        """Terminate child server when the container receives a signal."""
        stop_server(server_proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        if auto_launch:
            launch_models(endpoint)
        server_proc.wait()
    except Exception:
        stop_server(server_proc)
        raise
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()
