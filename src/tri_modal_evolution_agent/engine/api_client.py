from __future__ import annotations

import time
from typing import Any

import requests


class LocalOmniApiClient:
    def __init__(self, base_url: str, timeout_sec: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=min(self.timeout_sec, 30.0))
        if not response.ok:
            raise RuntimeError(f"Health check failed for {self.base_url}: {response.status_code} {response.text}")
        return response.json()

    def wait_until_ready(self, timeout_sec: float = 900.0, poll_sec: float = 5.0) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                return self.health()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(poll_sec)
        raise RuntimeError(f"Server {self.base_url} did not become ready in time: {last_error}")

    def generate(self, *, messages: list[dict[str, Any]], generation: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "generation": generation or {},
        }
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.timeout_sec,
        )
        if not response.ok:
            raise RuntimeError(f"Generate request failed for {self.base_url}: {response.status_code} {response.text}")
        return response.json()
