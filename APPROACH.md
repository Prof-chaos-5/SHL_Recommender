# Agent Architecture

```text
                           POST /chat
                                │
                                ▼
                   ┌─────────────────────────┐
                   │  Turn-1 Heuristic Fast  │
                   │  Path (regex, no LLM)   │
                   └────────────┬────────────┘
                   Clear?       │ Ambiguous?
                   ▼            │            ▼
           Direct State         │   Conversation Analyzer (LLM)
                   │            │            │
                   └────────────┴────────────┘
                                │
                                ▼
                        Conversation State
                                │
                                ▼
                      Agent Controller (Python)
                                │
      ┌──────────────┬──────────┴──────────┬───────────────┐
      │              │                     │               │
      ▼              ▼                     ▼               ▼
  Slot Filling   RECOMMEND             COMPARE          Refusal
  (deterministic) │                     │            (deterministic)
                  ▼                     ▼
          ┌───────────────────────────────────┐
          │         Candidate Builder         │
          │  1. BM25 retrieval (KEYWORD_BOOST)│
          │  2. Business Rules (force-inject) │
          │  3. Pin prior-turn IDs            │
          │  4. Pin explicitly-named items    │
          └────────────────┬──────────────────┘
                           │
                           ▼ (top-15 slice)
                     Prompt Builder
                           │
                           ▼
                      Explainer LLM
                     (explain only)
                           │
                           ▼
                 Grounding Validator (Python)
                 – validate ids against pool
                 – force-include mandatory items
                 – backfill if LLM hallucinated
                           │
                           ▼
                  Schema Validator (Python)
                 – dedupe by entity_id
                 – cap at 10
                 – valid test_type
                           │
                           ▼
                 Stamp reply with entity IDs
                 (zero-width Unicode, invisible)
                           │
                           ▼
                      ChatResponse
```

---

# Philosophy

**The LLM should not decide what to do.**

The LLM has exactly two jobs:
1. **Understand** — extract structured state from raw conversation history (Analyzer).
2. **Explain** — write a natural-language description of a pre-selected candidate pool (Explainer).

Every other decision — routing, retrieval, ranking, forced-inclusion, validation, repair — is deterministic Python. This means:
- No hallucinated assessments can survive to the response.
- Business rules apply on every single turn regardless of what the LLM writes.
- The system degrades gracefully: if the Explainer LLM fails or times out, a deterministic fallback produces a valid response.

---

# Components

## 1. Turn-1 Heuristic Fast Path

**File:** `app/agents/heuristics.py`

**No LLM call.** Runs only on the first user message.

On turn 1 there is no prior state to merge across turns — which is the only thing the LLM Analyzer is strictly needed for. A fast regex pass can therefore resolve the majority of opening messages in zero LLM calls:

| Signal | Action |
|---|---|
| Clear job title present | `RECOMMEND` directly (1 LLM call total) |
| JD-shaped text pasted | `RECOMMEND` directly |
| Obviously vague ("need an assessment") | `SLOT_FILL`, 0 LLM calls |
| Injection / off-topic keywords | `REFUSE`, 0 LLM calls |
| "What's the difference between X and Y" | `COMPARE` directly |
| Anything else | Fall through to full Analyzer |

The role pattern recognizes plural nouns, lowercase titles, and a broad set of trigger verbs (`hiring`, `screening`, `need`, `recruiting`) including article-free forms ("need plant operators"). A leading-filler strip removes verbs/prepositions that the lazy regex anchor can drag into the captured role phrase.

---

## 2. Conversation Analyzer

**File:** `app/agents/analyzer.py`  
**Mechanism:** LLM call (only when the heuristic fast path doesn't resolve the turn)

Converts the full raw message history into a typed `ConversationState` object. The LLM never recommends assessments here — extraction only.

**Output (`ConversationState`):**

```json
{
  "intent": "recommend | compare | close",
  "role": "Senior Backend Engineer",
  "seniority": "senior",
  "skills": ["Java", "Spring", "SQL"],
  "constraints": {
    "duration_less_than_mins": null,
    "online_only": null,
    "adaptive": null
  },
  "required_traits": ["safety-conscious", "reliable"],
  "technical_required": true,
  "personality_required": false,
  "personality_excluded": false,
  "excluded_items": ["OPQ32r"],
  "comparison_targets": [],
  "explicitly_named_items": ["OPQ MQ Sales Report"],
  "is_off_topic": false,
  "is_injection": false,
  "missing_fields": []
}
```

Key fields added beyond a basic extraction:

- **`excluded_items`** — specific assessment names the user asked to remove this conversation. Used to prevent business rules from re-forcing them back in on subsequent turns.
- **`explicitly_named_items`** — any assessment name mentioned anywhere in the conversation (including inside an earlier assistant compare reply) that the user showed interest in adding or keeping. Feeds the `_pin_explicit_names` step in the candidate builder so names that only appeared in a compare turn still make it into the pool when the user later says "add MQ."

The Analyzer reads the **entire** conversation history, not just the latest message, and must carry forward all fields mentioned in earlier turns. It is stateless — the API is stateless — so it fully reconstructs the conversation on each call.

---

## 3. Agent Controller

**File:** `app/agents/controller.py`  
**Mechanism:** Pure Python, deterministic

Routes to one of four actions based on `ConversationState`:

```
Injection detected?           → REFUSE_INJECTION
Off-topic?                    → REFUSE_OFF_TOPIC
intent == "close"             → CLOSE
intent == "compare"           → COMPARE
missing_fields not empty?     → SLOT_FILL
                              → RECOMMEND
```

Never generates language. Never calls the LLM.

---

## 4. Slot Filling

**File:** `app/prompts.py` → `build_slot_fill_reply()`  
**Mechanism:** Deterministic template

Asks exactly **one** clarifying question when `role` is missing. Never stacks multiple questions.

The only required slot is `role`. Seniority and skills are optional — the BM25 retriever and business rules can work productively without them.

---

## 5. Candidate Builder

**File:** `app/agent.py` → `_get_candidates()`

Assembles the final candidate pool fed to the Explainer LLM. Runs in four deterministic stages:

### Stage 1 — BM25 Retrieval

**File:** `app/retrieval.py`

Pure Python BM25 over an enriched catalog. No sentence-transformers, no ONNX, no FAISS. Memory footprint ≈ 210 MB total (well within a 512 MB free tier).

Two enrichment layers compensate for BM25's zero semantic understanding:

**`EMBED_ENRICHMENT`** — per-item synonym injection. Adds domain keywords (e.g. "safety, reliability, procedure compliance") to assessments with opaque product names so BM25 can match them against natural hiring-manager language.

**`KEYWORD_BOOST_MAP`** — query-time 1.5× score multiplier for items whose names contain domain-matching fragments. Triggered by query keywords such as "contact center", "leadership", "java", "devops".

**`QUERY_ALWAYS_INCLUDE`** — keyword-triggered force-injection at score 997. Items that BM25 reliably misses but that are clearly correct for a context (e.g. "Global Skills Assessment" for "talent audit" queries).

**`get_item_by_name`** uses a four-stage resolution chain so hand-written catalog names in business rules never silently fail:
1. Exact case-insensitive match
2. Normalized exact match (strips `(New)`, dash/space variants, `&` → `and`)
3. Normalized substring match, either direction
4. Fuzzy difflib fallback (cutoff 0.72) with a loud log on miss

### Stage 2 — Business Rules

**File:** `app/agents/business_rules.py`

Deterministic force-injection of domain-specific assessments. Items injected at score 998, tagged `_retrieval="business_rule"`. The LLM still sees the full pool and chooses the final shortlist — these are candidates, not mandates.

Rules:
- **Personality (OPQ32r)** — included for all professional/non-junior roles unless `personality_excluded=True` or the item is in `excluded_items`.
- **Leadership recipe** (OPQ32r + OPQ Universal Competency Report + OPQ Leadership Report) — for senior/director/CXO seniority markers.
- **Contact center** — SVAR Spoken English, Contact Center Call Simulation, Entry Level Customer Service, Customer Service Phone Simulation.
- **Healthcare** — DSI. Healthcare + admin role additionally forces Medical Terminology and MS Word Essentials.
- **Manufacturing/industrial/chemical** — Manufac. & Indust. Safety & Dependability 8.0 and Workplace Health and Safety (New).

All rules respect `excluded_items` — an item explicitly removed by the user is never re-forced back in by a rule trigger.

### Stage 3 — Pin Prior-Turn IDs

**File:** `app/agent.py` → `_pin_prior()` / `_stamp_recommendation_ids()`

Each RECOMMEND response stamps the recommended entity IDs into the reply text as invisible zero-width Unicode characters. On the next turn, `_prior_recommendation_ids()` decodes these from the last assistant message and re-pins those items at score 998 (`_retrieval="previous_turn"`).

This is the **turn memory** mechanism — it prevents the candidate pool from drifting on later turns (e.g. when the user asks a comparison question and the pool temporarily narrows to only the two compared items). Items the user has since excluded are filtered out before pinning.

### Stage 4 — Pin Explicitly-Named Items

**File:** `app/agent.py` → `_pin_explicit_names()`

Resolves `ConversationState.explicitly_named_items` (assessment names the Analyzer found in the conversation) to catalog entries and injects them at score 999.5 — outranking even business-rule items. This handles the case where a user says "add MQ" after a compare turn where only the assistant mentioned the item's name, but BM25 can't reconstruct that from a short confirmation message alone.

---

## 6. Explainer LLM

**File:** `app/prompts.py` → `RECOMMEND_SYSTEM_TEMPLATE` / `COMPARE_SYSTEM_TEMPLATE`

Receives a trimmed slice of the top-15 candidates (prompt token budget control) and writes a natural-language reply selecting from them by entity ID.

**The LLM cannot:**
- Recommend anything not in the candidate list
- Invent or modify IDs
- Add items from its training data

**Selection priority guidance in the prompt:**
1. `business_rule` / `always_include` candidates
2. `explicit_name` (user named it)
3. `query_always_include`
4. Higher BM25-ranked candidates
5. Newer assessment versions over older ones

The running shortlist rule ("maintain the shortlist across turns — include previously agreed assessments unless the user explicitly removes them") is stated in the prompt as a first-class selection rule, providing a second layer of turn-memory on top of the pinning mechanism.

A deterministic fallback (`_deterministic_fallback`) is used if the LLM call fails or the 25-second request deadline is exceeded.

---

## 7. Validators

**File:** `app/agents/validators.py`

Two-pass guarantee. No LLM re-calls — all repair is pure Python.

### Grounding Validator (`ground_and_repair`)

1. Resolves LLM-selected IDs back against the actual retrieved pool — hallucinated IDs are dropped.
2. Backfills from the ranked pool if the LLM's entire selection was invalid.
3. Force-includes items tagged `business_rule`, `always_include`, or `previous_turn` unless they appear in `excluded_names`. Forced items are appended after the cap loop so they can replace the lowest-ranked non-forced item rather than being squeezed out.

Exclusion is tracked via explicit `ConversationState.excluded_items` state (extracted by the Analyzer), not by scanning the LLM's reply prose for keywords — the prose approach was unreliable in both directions.

### Schema Validator (`schema_check`)

- Deduplicates by `entity_id` (falls back to name)
- Enforces `test_type` ∈ `{K, A, P, B, C, D, S, E}`
- Hard cap at 10 recommendations

---

## 8. Refusal Agent

**Mechanism:** Deterministic templates in `app/prompts.py`

Two refusal types:
- **`REFUSE_INJECTION`** — triggered by known jailbreak keyword patterns (pre-LLM screen in `heuristics.py`) or when the Analyzer sets `is_injection=True`.
- **`REFUSE_OFF_TOPIC`** — for employment law, compensation, medical diagnosis questions that have nothing to do with assessment selection.

HIPAA-compliance questions are explicitly **not** off-topic — "do we need HIPAA compliance testing" is an assessment selection question.

---

## 9. LLM Client

**File:** `app/llm_client.py`

Shared Groq caller used by both the Analyzer and the Explainer. Designed for the free tier's tight rate limits:

- Up to 3 API keys rotated in round-robin (`GROQ_API_KEY_1/2/3`)
- Parses the `retry-after` header from 429 responses — rotates immediately if wait > 3s, otherwise sleeps
- Fallback model (`llama-3.1-8b-instant`) if the primary model exhausts all key attempts
- Raises `RuntimeError` only if both models are exhausted; caller uses deterministic fallback

---

## 10. Catalog Validator

**File:** `app/validate_catalog_names.py`

Offline diagnostic tool. Resolves every hardcoded assessment name in `business_rules.py`, `ALWAYS_INCLUDE`, `QUERY_ALWAYS_INCLUDE`, `EMBED_ENRICHMENT`, and `LEGACY_UPGRADES` against the live catalog and reports:

- **OK** — exact catalog match
- **FALLBACK** — only matched via fuzzy/substring (worth manual review)
- **MISSING** — no match; item is silently dropped from its forced pool

Non-zero exit code on any MISSING, suitable for CI.

---

# State Object

```python
class ConversationState:
    intent: str                    # "recommend" | "compare" | "close"
    role: str | None
    seniority: str | None
    skills: list[str]
    constraints: dict              # duration, online_only, adaptive
    required_traits: list[str]
    technical_required: bool
    personality_required: bool
    personality_excluded: bool     # user explicitly refused all personality tests
    excluded_items: list[str]      # specific assessment names user asked to remove
    comparison_targets: list[str]  # names in a "what's the diff" question
    explicitly_named_items: list[str]  # named anywhere; user showed interest
    is_off_topic: bool
    is_injection: bool
    missing_fields: list[str]      # only ever contains "role"
```

No downstream component reads raw messages directly. Everything operates on this object, which is reconstructed fresh from the full history on every call (stateless API).

---

# Retrieval Score Tiers

| Score | Source | Meaning |
|---|---|---|
| 999.5 | `explicit_name` | User named this item in conversation |
| 998.0 | `business_rule` / `previous_turn` | Business rule or stamped prior recommendation |
| 997.0 | `query_always_include` | Keyword-triggered always-include |
| < 50 | BM25 | Normal retrieval score |

---

# Why this architecture?

**LLM → Understand**: The Analyzer excels at reading intent and extracting structure from messy natural language. Use it for exactly that.

**Python → Decide**: All routing, forced-inclusion, and validation is deterministic. Business rules cannot be forgotten mid-conversation. Hallucinated items cannot survive to the response.

**Retrieval Pipeline**: BM25 + keyword boost + always-include tiers gives domain-specific precision without the memory footprint or cold-start latency of dense embeddings. EMBED_ENRICHMENT adds semantic coverage for opaque product names.

**Multi-tier Turn Memory**: Zero-width ID stamping (invisible to the user, readable by the next turn's `_pin_prior`) ensures the candidate pool never loses previously agreed items when the conversation topic temporarily narrows. The Analyzer's `explicitly_named_items` field catches the remaining case where short confirmation messages don't carry enough BM25 signal on their own.

**LLM → Explain**: The Explainer only writes sentences — it cannot introduce an item not already in the pool. Its entire creative freedom is bounded by a validated candidate list.

**Validators → Guarantee**: Two deterministic Python passes catch anything the Explainer misses: hallucinated IDs are dropped, forced items are re-inserted, schema is enforced. No second LLM call for repair.