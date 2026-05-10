import json
import os
import re
import time

from groq import Groq
from dotenv import load_dotenv

from app.retrieval import (
    search,
    get_item_by_name,
    format_candidates,
    get_test_type,
)

from app.prompts import SYSTEM_PROMPT

from app.models import (
    ChatResponse,
    Recommendation,
    Message,
)

load_dotenv()

# ============================================================================
# MULTI-KEY CONFIG
# ============================================================================

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]

if not GROQ_KEYS:
    raise ValueError("No Groq API keys found in environment variables")

LLM_MODEL = "openai/gpt-oss-120b"
CLIENTS = [Groq(api_key=key) for key in GROQ_KEYS]
_client_index = 0
MAX_RETRIES_PER_KEY = 3
MAX_TOTAL_ATTEMPTS = len(CLIENTS) * MAX_RETRIES_PER_KEY


# ============================================================================
# CLIENT ROTATION
# ============================================================================

def get_next_client():
    global _client_index
    client = CLIENTS[_client_index]
    _client_index = (_client_index + 1) % len(CLIENTS)
    return client


# ============================================================================
# QUERY BUILDING
# ============================================================================

def build_query_from_history(messages: list[Message]) -> str:
    """
    Build retrieval query from conversation history.
    Recent messages weighted higher by repeating them —
    avoids diluting the query with stale early context.
    """
    user_msgs = [m.content for m in messages if m.role == "user"]

    if not user_msgs:
        return ""

    # Most recent message repeated for emphasis
    # Earlier messages provide context but less weight
    if len(user_msgs) == 1:
        return user_msgs[0]

    # Last message gets double weight via repetition
    recent = user_msgs[-1]
    context = " ".join(user_msgs[:-1])
    return f"{context} {recent} {recent}"


def extract_compare_names(messages: list[Message]) -> list[str]:
    """
    Detect compare intent and extract assessment names.
    Uses catalog name lookup rather than raw word extraction
    to avoid grabbing stop words and noise tokens.
    """
    if not messages:
        return []

    last = messages[-1].content
    lower = last.lower()

    if "difference between" not in lower and "compare" not in lower:
        return []

    # Look for quoted names first — most reliable
    quoted = re.findall(r'"([^"]+)"', last)
    if len(quoted) >= 2:
        return quoted

    # Fall back to finding catalog-like proper nouns
    # Match capitalized sequences including numbers and special chars
    candidates = re.findall(r'\b[A-Z][A-Za-z0-9+\-\.&\s]{2,40}', last)
    results = []
    seen = set()
    for c in candidates:
        c = c.strip()
        # Skip generic words
        if c.lower() in {"what", "the", "difference", "between", "and",
                         "compare", "which", "better", "should", "use"}:
            continue
        if c not in seen and len(c) > 2:
            results.append(c)
            seen.add(c)

    return results[:5]


def is_jd_paste(messages: list[Message]) -> bool:
    """
    Detect if the user has pasted a job description.
    Signals: JD keywords + substantial length.
    """
    if not messages:
        return False

    last = messages[-1].content.lower()
    jd_signals = [
        "job description", "jd:", "responsibilities",
        "requirements", "qualifications", "years of experience",
        "we are looking for", "the role", "you will",
        "must have", "nice to have", "about the role",
    ]
    signal_count = sum(1 for s in jd_signals if s in last)
    return signal_count >= 2 and len(last) > 200


# ============================================================================
# RETRIEVAL
# ============================================================================

def get_candidates(messages: list[Message]) -> list[dict]:
    """
    Build retrieval candidate pool.
    - Compare queries: fetch specific items by name + semantic context
    - JD paste: use full JD text as query for rich retrieval
    - Everything else: hybrid search on full conversation history
    """
    compare_names = extract_compare_names(messages)

    if compare_names:
        items = [get_item_by_name(n) for n in compare_names]
        items = [i for i in items if i is not None]

        if items:
            query = build_query_from_history(messages)
            semantic = search(query, top_k=15)
            seen = {i["entity_id"] for i in items}
            for s in semantic:
                if s["entity_id"] not in seen:
                    items.append(s)
                    seen.add(s["entity_id"])
            return items[:20]

    query = build_query_from_history(messages)
    return search(query, top_k=30)


# ============================================================================
# LLM CALLER
# ============================================================================

def call_llm(system: str, messages: list[Message]) -> str:
    """
    Groq caller with:
    - Multi-key rotation
    - Rate limit backoff with precise wait time extraction
    - Network error retry
    """
    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({
            "role": "user" if m.role == "user" else "assistant",
            "content": m.content,
        })

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        client = get_next_client()
        try:
            print(f"[llm] Attempt {attempt + 1} using {LLM_MODEL}...")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=api_messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        except Exception as e:
            error_text = str(e).lower()

            # Rate limit — extract precise wait time if available
            if any(x in error_text for x in ["rate limit", "429", "too many requests"]):
                wait_time = min(2 ** attempt, 30)
                match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_text)
                if match:
                    mins = int(match.group(1)) if match.group(1) else 0
                    secs = float(match.group(2))
                    wait_time = int(mins * 60 + secs) + 1

                if wait_time > 300:
                    print(f"[llm] Wait too long ({wait_time}s), rotating key...")
                    continue

                print(f"[llm] Rate limit — sleeping {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Network / transient errors
            if any(x in error_text for x in ["connection", "timeout", "503"]):
                print("[llm] Network issue, retrying...")
                time.sleep(2)
                continue

            print(f"[llm] Error: {e}")
            if attempt == MAX_TOTAL_ATTEMPTS - 1:
                break

    raise RuntimeError("All Groq API attempts exhausted")


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_llm_response(raw: str) -> ChatResponse:
    """Parse and validate LLM JSON response."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {raw[:200]}")

    data = json.loads(match.group())

    recommendations = []
    for r in data.get("recommendations", []):
        if isinstance(r, dict) and "name" in r and "url" in r:
            recommendations.append(Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r.get("test_type", "K"),
            ))

    # Spec says empty array, never null
    return ChatResponse(
        reply=data.get("reply", "Sorry, I could not process that."),
        recommendations=recommendations[:10],
        end_of_conversation=bool(data.get("end_of_conversation", False)),
    )


# ============================================================================
# FALLBACK
# ============================================================================

def safe_fallback(error_msg: str) -> ChatResponse:
    print(f"[agent] Fallback triggered: {error_msg}")
    return ChatResponse(
        reply="I'm having trouble processing that. Could you rephrase your request?",
        recommendations=[],
        end_of_conversation=False,
    )


# ============================================================================
# MAIN AGENT
# ============================================================================

def run_agent(messages: list[Message]) -> ChatResponse:
    """
    Main agent entrypoint.
    1. Retrieve candidates (hybrid search)
    2. Build system prompt with candidates injected
    3. Call LLM
    4. Parse + validate response
    5. URL safety filter — strip hallucinated URLs
    """
    try:
        # 1. Retrieve
        candidates = get_candidates(messages)
        candidates_text = format_candidates(candidates)

        # 2. Build prompt
        system = SYSTEM_PROMPT.replace("{candidates}", candidates_text)

        # 3. LLM call
        raw = call_llm(system, messages)

        # 4. Parse
        response = parse_llm_response(raw)

        # 5. Validate — URL safety filter + name canonicalization
        def clean_url(u: str) -> str:
            return u.lower().split("://")[-1].strip("/")

        valid_urls_map = {
            clean_url(item.get("link", "")): item.get("name")
            for item in candidates
            if item.get("link")
        }
        name_to_canonical = {
            c["name"].lower(): c["name"] for c in candidates
        }

        final_recs = []
        seen_names = set()

        for r in response.recommendations:
            canonical_name = None
            c_url = clean_url(r.url)

            # URL match first — most reliable
            if c_url in valid_urls_map:
                canonical_name = valid_urls_map[c_url]
            # Name match fallback
            elif r.name.lower() in name_to_canonical:
                canonical_name = name_to_canonical[r.name.lower()]

            # Find the actual catalog item for this canonical name
            catalog_item = next((c for c in candidates if c["name"] == canonical_name), None)

            if catalog_item and canonical_name not in seen_names:
                r.name = canonical_name
                r.url = catalog_item.get("link", r.url)
                r.test_type = get_test_type(catalog_item.get("keys", []))
                final_recs.append(r)
                seen_names.add(canonical_name)

        response.recommendations = final_recs
        return response

    except json.JSONDecodeError as e:
        print(f"[agent] JSON parse error: {e}")
        return safe_fallback(str(e))
    except Exception as e:
        print(f"[agent] Unexpected error: {e}")
        return safe_fallback(str(e))