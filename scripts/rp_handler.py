import os
import threading
from typing import Any, Dict, Optional

import httpx
import runpod

from bootstrap import (
    LAUNCHED_MODEL_UID,
    MODEL_READY,
    _decode_base64_image,
    _fetch_image_from_url,
    configure_xinference_home,
    dbg_log,
    launch_models,
    parse_bool,
    start_server,
    stop_server,
    wait_for_server,
)

# Protect initialization so only one worker thread starts Xinference + model.
_init_lock = threading.Lock()
_initialized = False
_server_proc = None
_xinference_endpoint: Optional[str] = None


def _init_runtime() -> None:
    """Start xinference-local and launch the model once per worker."""
    global _initialized, _server_proc, _xinference_endpoint
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        configure_xinference_home()

        host = os.getenv("XINFERENCE_HOST", "0.0.0.0")
        port = os.getenv("XINFERENCE_PORT", "9997")
        log_level = os.getenv("XINFERENCE_LOG_LEVEL", "info")
        endpoint = os.getenv("XINFERENCE_ENDPOINT", f"http://127.0.0.1:{port}")
        _xinference_endpoint = endpoint

    try:
        _server_proc = start_server(host, port, log_level)
        wait_for_server(endpoint, server_proc=_server_proc)
        auto_launch = parse_bool(os.getenv("AUTO_LAUNCH_MODEL", "1"))
        if auto_launch:
            launch_models(endpoint, _server_proc)
        _initialized = True
        dbg_log(
            hypothesis_id="INIT",
            location="rp_handler.py:_init_runtime",
            message="initialized runtime",
            data={"endpoint": endpoint, "model_ready": MODEL_READY},
        )
    except BaseException as exc:  # noqa: BLE001
        dbg_log(
            hypothesis_id="INIT_ERR",
            location="rp_handler.py:_init_runtime",
            message="runtime init failed",
            data={"error": str(exc)},
        )
        raise RuntimeError(f"handler init failed: {exc}") from exc


def _ensure_model_uid() -> str:
    uid = LAUNCHED_MODEL_UID or os.getenv("VIDEO_MODEL_UID")
    if not uid:
        raise ValueError("model not ready")
    return uid


def _call_xinference(uid: str, prompt: str, image_bytes: Optional[bytes], timeout: int) -> Dict[str, Any]:
    if not _xinference_endpoint:
        raise RuntimeError("xinference endpoint not initialized")

    if image_bytes:
        files = {"image": ("input", image_bytes, "application/octet-stream")}
        data = {"model": uid}
        if prompt:
            data["prompt"] = prompt
        resp = httpx.post(
            f"{_xinference_endpoint}/v1/video/generations/image",
            data=data,
            files=files,
            timeout=timeout,
        )
    else:
        payload = {"model": uid, "prompt": prompt or ""}
        resp = httpx.post(
            f"{_xinference_endpoint}/v1/video/generations",
            json=payload,
            timeout=timeout,
        )

    if resp.status_code >= 400:
        raise RuntimeError(f"xinference error {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler entrypoint.

    Expects job['input'] to contain:
      - prompt: Optional[str]
      - image_url: Optional[str]
      - image_base64: Optional[str]
    """
    try:
        _init_runtime()
    except Exception as exc:  # noqa: BLE001
        return {
            "error": "init_failed",
            "message": str(exc),
            "ready": MODEL_READY,
        }

    job_input = job.get("input") or {}
    prompt = job_input.get("prompt", "")
    image_url = job_input.get("image_url")
    image_b64 = job_input.get("image_base64")

    image_bytes: Optional[bytes] = None
    try:
        if image_url:
            image_bytes = _fetch_image_from_url(image_url)
        elif image_b64:
            image_bytes = _decode_base64_image(image_b64)
    except Exception as exc:  # noqa: BLE001
        dbg_log(
            hypothesis_id="INPUT_ERR",
            location="rp_handler.py:handler",
            message="failed to load input image",
            data={"error": str(exc)},
        )
        return {"error": "input_error", "message": str(exc), "ready": MODEL_READY}

    try:
        uid = _ensure_model_uid()
    except Exception as exc:  # noqa: BLE001
        dbg_log(
            hypothesis_id="MODEL_ERR",
            location="rp_handler.py:handler",
            message="model not ready",
            data={"error": str(exc)},
        )
        return {"error": "model_not_ready", "message": str(exc), "ready": MODEL_READY}

    api_timeout = int(os.getenv("API_HTTP_TIMEOUT", "600"))

    dbg_log(
        hypothesis_id="JOB",
        location="rp_handler.py:handler",
        message="processing job",
        data={"model_uid": uid, "has_image": bool(image_bytes), "prompt_len": len(prompt or "")},
    )

    try:
        result = _call_xinference(uid, prompt, image_bytes, api_timeout)
        return {"output": result, "model_uid": uid, "ready": MODEL_READY}
    except Exception as exc:  # noqa: BLE001
        dbg_log(
            hypothesis_id="INFER_ERR",
            location="rp_handler.py:handler",
            message="xinference call failed",
            data={"error": str(exc)},
        )
        return {"error": "inference_error", "message": str(exc), "ready": MODEL_READY}


def _shutdown() -> None:
    """Best-effort cleanup when the worker exits."""
    stop_server(_server_proc)


runpod.serverless.start({"handler": handler, "cleanup": _shutdown})
