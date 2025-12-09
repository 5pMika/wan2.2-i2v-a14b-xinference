import os
import time
from typing import Dict, List, Optional

import boto3
import requests
from botocore.exceptions import ClientError
from xinference.client import Client


def parse_bool(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "t", "yes", "y", "on"}


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
    local_path = os.getenv("MODEL_LOCAL_PATH", "/data/models/wan2.2-i2v-a14b")
    skip_existing = parse_bool(os.getenv("MODEL_SKIP_EXISTING", "1"))

    if not bucket or not endpoint:
        raise RuntimeError("S3 model sync is enabled but MODEL_S3_BUCKET or MODEL_S3_ENDPOINT is missing.")

    normalized_prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
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
            raise RuntimeError(f"S3 model check failed; missing keys: {missing}")

    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            found = True
            rel = key[len(normalized_prefix) :].lstrip("/") if normalized_prefix else key
            target = os.path.join(local_path, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if skip_existing and os.path.exists(target):
                continue
            print(f"[bootstrap] downloading {key} -> {target}", flush=True)
            s3.download_file(bucket, key, target)

    if not found:
        raise RuntimeError(f"S3 model check failed; no objects found at prefix '{normalized_prefix}' in bucket '{bucket}'.")

    print(f"[bootstrap] S3 model sync complete -> {local_path}", flush=True)


def main() -> None:
    port = os.getenv("XINFERENCE_PORT", "9997")
    endpoint = os.getenv("XINFERENCE_ENDPOINT", f"http://127.0.0.1:{port}")

    ensure_s3_model()
    wait_for_server(endpoint)

    client = Client(endpoint)
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


if __name__ == "__main__":
    main()
