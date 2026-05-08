from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, List

import requests
import yaml
from swift.rewards import ORM, orms


MIMO_CONFIG_ENV = "QWEN3OMNI_MIMO_CONFIG"
DEFAULT_MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_mimo_config() -> tuple[dict[str, Any], Path | None]:
    config_path_value = os.environ.get(MIMO_CONFIG_ENV, "").strip()
    if config_path_value:
        config_path = Path(config_path_value).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError(f"MIMO config must be a mapping: {config_path}")
        return config, config_path

    api_key_env = os.environ.get("QWEN3OMNI_MIMO_API_KEY_ENV", "XIAOMI_MIMO_API_KEY")
    model = os.environ.get("QWEN3OMNI_MIMO_MODEL", DEFAULT_MIMO_MODEL)
    return {
        "provider": "xiaomi_mimo",
        "model": {
            "key": model,
            "family": "xiaomi_mimo",
            "model_id": model,
        },
        "api": {
            "base_url": os.environ.get("QWEN3OMNI_MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL),
            "api_key_env": api_key_env,
            "model": model,
            "timeout_sec": 120,
            "max_retries": 2,
            "retry_sleep_sec": 2,
            "trust_env": True,
            "auth_header": os.environ.get("QWEN3OMNI_MIMO_AUTH_HEADER", "api-key"),
            "auth_scheme": os.environ.get("QWEN3OMNI_MIMO_AUTH_SCHEME", ""),
        },
        "request": {
            "endpoint": os.environ.get("QWEN3OMNI_MIMO_ENDPOINT", "/chat/completions"),
            "max_tokens_arg": os.environ.get("QWEN3OMNI_MIMO_MAX_TOKENS_ARG", "max_completion_tokens"),
            "extra_body": {"thinking": {"type": "disabled"}},
        },
        "generation_defaults": {
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }, None


def _normalize_answer(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return match.group(1)
    return re.sub(r"\s+", " ", text).strip().lower()


def _get_at(values: Any, index: int, default: Any = None) -> Any:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        return values[index] if index < len(values) else default
    return values


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            return "\n".join(part for part in parts if part)
    return ""


def _extract_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if not content and message.get("reasoning_content"):
        content = message.get("reasoning_content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                texts.append(str(part.get("text", "")))
            else:
                texts.append(str(part))
        return "\n".join(text for text in texts if text)
    return str(content or "")


def _parse_binary_score(text: str) -> float | None:
    normalized = (text or "").strip().lower()
    if not normalized:
        return None
    first_token = re.match(r"^[\s`'\"]*([01])(?:\b|[\s`'\".,:;})\]])", normalized)
    if first_token:
        return float(first_token.group(1))
    if "incorrect" in normalized or "wrong" in normalized or "not correct" in normalized:
        return 0.0
    if "correct" in normalized:
        return 1.0
    match = re.search(r'"score"\s*:\s*([01])', normalized)
    if match:
        return float(match.group(1))
    return None


class MimoAuthenticationError(RuntimeError):
    pass


class Qwen3OmniMimoJudge(ORM):
    """MIMO API based binary judge reward for Qwen3-Omni GRPO."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        config, self.config_path = _load_mimo_config()

        api_config = config.get("api", {}) or {}
        request_config = config.get("request", {}) or {}
        generation_defaults = config.get("generation_defaults", {}) or {}

        api_key_env = str(api_config.get("api_key_env") or "XIAOMI_MIMO_API_KEY")
        api_key = str(api_config.get("api_key") or os.environ.get(api_key_env, "")).strip()
        if not api_key:
            raise RuntimeError(
                f"Missing MIMO API key. Export {api_key_env} or set {MIMO_CONFIG_ENV} to a YAML config "
                "that names api.api_key_env. Do not commit real API keys."
            )

        self.provider = str(config.get("provider") or "xiaomi_mimo")
        self.model = str(api_config.get("model") or config.get("model", {}).get("model_id") or "mimo-v2.5")
        self.base_url = str(api_config.get("base_url") or DEFAULT_MIMO_BASE_URL).rstrip("/")
        self.endpoint = "/" + str(request_config.get("endpoint", "/chat/completions")).lstrip("/")
        self.url = f"{self.base_url}{self.endpoint}"
        self.timeout_sec = float(os.environ.get("QWEN3OMNI_MIMO_TIMEOUT_SEC", api_config.get("timeout_sec", 120.0)))
        self.max_retries = int(os.environ.get("QWEN3OMNI_MIMO_MAX_RETRIES", api_config.get("max_retries", 2)))
        self.retry_sleep_sec = float(os.environ.get(
            "QWEN3OMNI_MIMO_RETRY_SLEEP_SEC",
            api_config.get("retry_sleep_sec", 2.0),
        ))
        self.trust_env = bool(api_config.get("trust_env", True))
        self.max_tokens_arg = str(request_config.get("max_tokens_arg", "max_tokens"))
        self.extra_body = request_config.get("extra_body") or {}
        auth_header = str(api_config.get("auth_header", "Authorization"))
        auth_scheme = str(api_config.get("auth_scheme", "Bearer"))
        auth_value = api_key if not auth_scheme else f"{auth_scheme} {api_key}"
        self.headers = {auth_header: auth_value, "Content-Type": "application/json"}

        self.temperature = float(os.environ.get(
            "QWEN3OMNI_MIMO_TEMPERATURE",
            generation_defaults.get("temperature", 0.0),
        ))
        self.top_p = float(os.environ.get("QWEN3OMNI_MIMO_TOP_P", generation_defaults.get("top_p", 1.0)))
        self.max_tokens = int(os.environ.get("QWEN3OMNI_MIMO_MAX_TOKENS", "8"))
        self.allow_fallback = _env_bool("QWEN3OMNI_MIMO_ALLOW_FALLBACK", True)
        self.disable_after_auth_failure = _env_bool("QWEN3OMNI_MIMO_DISABLE_AFTER_AUTH_FAILURE", True)
        self.warn_limit = int(os.environ.get("QWEN3OMNI_MIMO_WARN_LIMIT", "4"))
        self._warn_count = 0
        self._cache: dict[str, float] = {}
        self._api_disabled = False
        self.api_success_count = 0
        self.api_failure_count = 0

    def config_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "config_path": str(self.config_path) if self.config_path else "<environment>",
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "allow_fallback": self.allow_fallback,
            "disable_after_auth_failure": self.disable_after_auth_failure,
        }

    def __call__(self, completions, solution=None, answer=None, messages=None, meta=None, **kwargs) -> List[float]:
        gold_values = solution if solution is not None else answer
        if gold_values is None:
            self._warn("Missing solution/answer column; returning zero rewards.")
            return [0.0 for _ in completions]

        rewards: list[float] = []
        for index, completion in enumerate(completions):
            gold = _get_at(gold_values, index, "")
            row_messages = _get_at(messages, index, [])
            row_meta = _get_at(meta, index, {}) or {}
            prompt_text = _last_user_text(row_messages)
            answer_text = row_meta.get("answer_text", "") if isinstance(row_meta, dict) else ""
            rewards.append(self._score_one(str(completion), str(gold), prompt_text, str(answer_text)))
        return rewards

    def _score_one(self, completion: str, gold: str, prompt_text: str, answer_text: str) -> float:
        cache_key = hashlib.sha256(
            "\n".join([prompt_text, completion, gold, answer_text]).encode("utf-8", errors="ignore")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self._api_disabled and self.allow_fallback:
            score = self._fallback_score(completion, gold)
            self._cache[cache_key] = score
            return score

        try:
            score = self._score_one_with_api(completion, gold, prompt_text, answer_text)
            self.api_success_count += 1
        except MimoAuthenticationError as exc:
            self.api_failure_count += 1
            if self.disable_after_auth_failure:
                self._api_disabled = True
            self._warn(f"MIMO authentication failed; using fallback={self.allow_fallback}. error={exc}")
            if not self.allow_fallback:
                raise
            score = self._fallback_score(completion, gold)
        except Exception as exc:  # noqa: BLE001
            self.api_failure_count += 1
            self._warn(f"MIMO request failed; using fallback={self.allow_fallback}. error={type(exc).__name__}: {exc}")
            if not self.allow_fallback:
                raise
            score = self._fallback_score(completion, gold)

        self._cache[cache_key] = score
        return score

    def _score_one_with_api(self, completion: str, gold: str, prompt_text: str, answer_text: str) -> float:
        payload = {
            "model": self.model,
            "messages": self._build_judge_messages(completion, gold, prompt_text, answer_text),
            "temperature": self.temperature,
            "top_p": self.top_p,
            self.max_tokens_arg: self.max_tokens,
        }
        payload.update(self.extra_body)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                session = requests.Session()
                session.trust_env = self.trust_env
                response = session.post(self.url, headers=self.headers, json=payload, timeout=self.timeout_sec)
                if not response.ok:
                    body = response.text[:500].replace("\n", " ")
                    if response.status_code in {401, 403}:
                        raise MimoAuthenticationError(f"HTTP {response.status_code}: {body}")
                    raise RuntimeError(f"HTTP {response.status_code}: {body}")
                text = _extract_chat_text(response.json())
                score = _parse_binary_score(text)
                if score is None:
                    raise RuntimeError(f"unparseable judge response: {text[:200]!r}")
                return score
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if isinstance(exc, MimoAuthenticationError):
                    raise
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_sec * (attempt + 1))
        raise RuntimeError(str(last_error))

    @staticmethod
    def _build_judge_messages(completion: str, gold: str, prompt_text: str, answer_text: str) -> list[dict[str, str]]:
        user_content = (
            "Judge whether the candidate answer should be accepted for the original multiple-choice task.\n"
            "Return exactly 1 for correct/acceptable and exactly 0 for incorrect. Do not explain.\n\n"
            f"Original prompt:\n{prompt_text}\n\n"
            f"Gold option letter: {gold}\n"
            f"Gold option text: {answer_text}\n\n"
            f"Candidate answer:\n{completion}"
        )
        return [
            {
                "role": "system",
                "content": "You are a strict binary evaluator for multiple-choice audio-visual QA answers.",
            },
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _fallback_score(completion: str, gold: str) -> float:
        prediction = _normalize_answer(completion)
        target = _normalize_answer(gold)
        return 1.0 if prediction and prediction == target else 0.0

    def _warn(self, message: str) -> None:
        if self._warn_count >= self.warn_limit:
            return
        self._warn_count += 1
        print(f"[mimo_reward][warning] {message}", flush=True)


orms["qwen3omni_mimo_judge"] = Qwen3OmniMimoJudge
