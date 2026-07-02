"""
Shared Groq caller. Used by both the Analyzer and the Recommend/Compare
explainer — kept in one place so retry/backoff/fallback logic isn't
duplicated per call site.

Two calls can happen per turn (Analyzer + explainer), so this is written to
fail fast rather than exhaust the request's 30s budget on one call: bounded
retries, short per-attempt timeout, and a smaller/faster fallback model if
the primary model's attempts are all exhausted.
"""

import os
import re
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]

if not GROQ_KEYS:
    raise ValueError("No Groq API keys found in environment variables")

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.1-8b-instant"

CLIENTS = [Groq(api_key=key) for key in GROQ_KEYS]
_client_index = 0
# One attempt per key, once through the rotation, then straight to the
# fallback model (or the deterministic fallback in agent.py). Under
# sustained load from an automated tester, retrying a key that's already
# rate-limited just burns wall-clock time without finding real capacity —
# better to fail fast and let the caller's deterministic fallback take over.
MAX_RETRIES_PER_KEY = 1
MAX_ATTEMPTS_PER_MODEL = len(CLIENTS) * MAX_RETRIES_PER_KEY


def _next_client() -> Groq:
    global _client_index
    client = CLIENTS[_client_index]
    _client_index = (_client_index + 1) % len(CLIENTS)
    return client


def _attempt_model(system: str, history: list[dict], model: str,
                    json_mode: bool, timeout: float) -> str | None:
    api_messages = [{"role": "system", "content": system}] + history

    for attempt in range(MAX_ATTEMPTS_PER_MODEL):
        client = _next_client()
        try:
            kwargs = dict(
                model=model,
                messages=api_messages,
                temperature=0.0,
                timeout=timeout,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            error_text = str(e).lower()

            if any(x in error_text for x in ["rate limit", "429", "too many requests"]):
                wait_time = min(2 ** attempt, 3)
                match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_text)
                if match:
                    mins = int(match.group(1)) if match.group(1) else 0
                    wait_time = int(mins * 60 + float(match.group(2))) + 1
                if wait_time > 3:
                    # Don't burn the request's time budget waiting on a key
                    # that's already rate-limited — rotate immediately.
                    continue
                time.sleep(wait_time)
                continue

            if any(x in error_text for x in ["connection", "timeout", "503"]):
                time.sleep(1)
                continue

            print(f"[llm] error on {model}: {e}")
            break

    return None


def call_llm(system: str, history: list[dict], json_mode: bool = True,
             timeout: float = 12.0) -> str:
    """
    history: list of {"role": "user"/"assistant", "content": str}
    Raises RuntimeError only if both the primary and fallback model are
    exhausted — callers should catch this and use a deterministic fallback,
    never let it propagate as an HTTP error.
    """
    result = _attempt_model(system, history, PRIMARY_MODEL, json_mode, timeout)
    if result is not None:
        return result

    print(f"[llm] {PRIMARY_MODEL} exhausted, falling back to {FALLBACK_MODEL}")
    result = _attempt_model(system, history, FALLBACK_MODEL, json_mode, timeout)
    if result is not None:
        return result

    raise RuntimeError("All Groq API attempts exhausted (primary + fallback)")