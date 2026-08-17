# SHL Assessment Recommender Agent

A production-oriented conversational AI agent that helps recruiters and hiring managers find the right SHL assessments for their roles. The agent handles multi-turn conversations, comparison questions, slot-filling clarifications, and domain-specific business rules, with deterministic grounding and validation to prevent assessments outside the catalog from reaching the final recommendation set.

### 🔗 Try it live

**[Live API Playground](https://shl-recommender-k7yn.onrender.com/docs)**

Explore the `/chat` endpoint directly through the interactive Swagger UI and test multi-turn recommendation, comparison, and slot-filling flows.

Read the full technical deep dive in the **[Architecture & Approach Document](APPROACH.md)**.

---

## Architecture at a Glance

```
POST /chat
    │
    ├─ Turn-1 Heuristic Fast Path (regex, 0 LLM calls for clear first messages)
    │
    ├─ Conversation Analyzer LLM  → ConversationState
    │
    ├─ Agent Controller (Python, deterministic routing)
    │   ├─ SLOT_FILL  (deterministic template)
    │   ├─ REFUSE     (deterministic template)
    │   └─ RECOMMEND / COMPARE
    │       ├─ BM25 Retrieval + Keyword Boost
    │       ├─ Business Rules (domain-specific force-injection)
    │       ├─ Pin Prior-Turn IDs (turn memory via zero-width Unicode stamps)
    │       ├─ Pin Explicitly-Named Items
    │       ├─ Explainer LLM (explain only, never invent)
    │       ├─ Grounding Validator (drop hallucinations, re-add forced items)
    │       └─ Schema Validator (dedupe, cap, enforce test_type)
    │
    └─ ChatResponse
```
Execution model: Rather than using an open-ended ReAct loop, the agent follows a constrained, state-driven execution loop: the LLM handles language understanding and explanation, while Python controls routing, retrieval, memory, business rules, and correctness-critical validation.

**The LLM never decides what assessments exist — only how to explain the ones Python already selected.**

---

## Key Features

### Agentic Orchestration

- **State-Driven Agent Controller** — Python deterministically routes each turn through SLOT_FILL, REFUSE, RECOMMEND, or COMPARE states. The LLM is responsible for structured conversation understanding and explanation rather than correctness-critical decisions.
- **Multi-turn State & Memory** — Zero-width Unicode stamps in assistant replies carry entity IDs across turns. Items agreed on in turn 1 survive a comparison detour in turn 3 and reappear correctly in turn 4.
- **Explicit-Name Pinning** — Assessment names mentioned during a comparison turn are preserved through a dedicated `explicitly_named_items` state channel. Short confirmations like `"add MQ"` therefore don't depend on BM25 rediscovering the item.

### Retrieval & Grounding

- **BM25 Lite Retrieval** — Pure Python BM25 with per-item synonym enrichment (`EMBED_ENRICHMENT`) and query-time keyword boost. No sentence-transformers, no ONNX, no FAISS. Memory footprint ≈ 210 MB, well within a 512 MB free-tier cloud instance.
- **Deterministic Business Rules** — Domain-specific forced inclusion for leadership, contact center, healthcare, and manufacturing/industrial roles. Rules fire every turn in Python, independent of LLM output.
- **Hardened Grounding** — Two-pass Python validation drops hallucinated IDs, backfills invalid selections, re-inserts mandatory items, deduplicates by `entity_id`, and enforces the recommendation schema.
- **Fuzzy Catalog Resolution** — `get_item_by_name` uses a four-stage chain (exact → normalized → substring → difflib fuzzy) so hand-written catalog names in business rules never silently fail on dash/spacing/`(New)` variants.

### Reliability & Deployment

- **Rate-Limit Resilient LLM Client** — Three API keys rotate in round-robin order, `retry-after` is parsed from 429 responses, and a fast fallback model is used when required. Deterministic fallback behavior handles model failures.
- **Resource-Constrained Deployment** — Replaced the original FAISS + embedding pipeline with pure-Python BM25 to reduce memory usage from >900 MB to ≈210 MB and support constrained cloud deployment.

### Evaluation

- **Smart Evaluator** — LLM-based user simulation using Groq Qwen drives multi-turn conversations against live traces, computes Recall@10, and logs per-trace matches/misses. Supports `--all` or selective trace execution.
---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (stateless) |
| Orchestration | Custom Python controller |
| Primary Retrieval | Pure Python BM25 |
| LLM (Analyzer + Explainer) | Groq — `openai/gpt-oss-120b` (primary), `llama-3.1-8b-instant` (fallback) |
| User Simulator (eval) | Groq — `qwen/qwen3-32b` |
| Validation | Python — no LLM re-calls |
| Deployment | Uvicorn, compatible with Render Free Tier |
---

## Installation & Setup

1. **Clone and install:**
   ```powershell
   git clone <repo-url>
   cd shl-recommender
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Copy `.env.example` to `.env` and fill in your keys:
   ```powershell
   cp .env.example .env
   ```
   Required variables:
   ```
   GROQ_API_KEY=gsk_...
   GROQ_API_KEY_1=gsk_...   # optional — enables key rotation
   GROQ_API_KEY_2=gsk_...
   GROQ_API_KEY_3=gsk_...
   CATALOG_URL=https://...  # optional — falls back to data/catalog.json
   ```

3. **Run the server:**
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **API:**
   ```
   POST /chat
   Content-Type: application/json

   {
     "messages": [
       {"role": "user", "content": "We need assessments for a senior Java engineer."}
     ]
   }
   ```
   Response:
   ```json
   {
     "reply": "For a senior Java engineer I recommend...",
     "recommendations": [
       {"name": "Core Java (Advanced Level) (New)", "url": "...", "test_type": "K", "entity_id": "1234"},
       ...
     ],
     "end_of_conversation": false
   }
   ```

---

## Evaluation

Run the smart evaluator against all traces:
```powershell
cd eval
python smart_eval.py --all
```

Run against specific traces:
```powershell
python smart_eval.py --traces C1 C3 C9
```

The evaluator:
- Loads Markdown trace files from `data/traces/`
- Extracts the persona, opening message, and canonical user turns
- Drives the live agent through a simulated multi-turn conversation (Qwen simulates the user)
- Computes `recall@10` per trace and `mean_recall@10` overall
- Saves full conversation logs + scores to `eval/smart_results.json`

**Validate hardcoded catalog names** (catches silent mismatches before running eval):
```powershell
python -m app.validate_catalog_names
```

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── agent.py                   # Orchestrator — turn memory, candidate builder
│   ├── llm_client.py              # Groq client with key rotation + fallback
│   ├── models.py                  # Pydantic models (ConversationState, ChatResponse)
│   ├── prompts.py                 # Centralized prompt templates
│   ├── retrieval.py               # BM25 engine, enrichment, keyword boost, catalog lookup
│   ├── validate_catalog_names.py  # Offline catalog name validator (CI-friendly)
│   └── agents/
│       ├── analyzer.py            # Conversation → ConversationState (LLM)
│       ├── controller.py          # Deterministic router
│       ├── heuristics.py          # Turn-1 fast path (regex, no LLM)
│       ├── business_rules.py      # Domain-specific force-inclusion rules
│       └── validators.py          # Grounding + schema validation
├── data/
│   ├── catalog.json               # SHL assessment catalog
│   └── traces/                    # Evaluation trace files (C1.md … C10.md)
├── eval/
│   ├── smart_eval.py              # LLM-based multi-turn evaluator
│   ├── smart_results.json         # Latest evaluation output
│   └── metrics.py                 # recall@k, mean_recall@k
├── .env.example
├── requirements.txt
├── APPROACH.md                    # Architecture deep dive
└── README.md
```

---

## Evaluation Results

| Run | Model | Mean Recall@10 | Notes |
|---|---|---|---|
| Baseline | — | 0.320 | Direct BM25, no business rules |
| + Business rules | gpt-oss-120b | 0.735 | Force-inclusion, enrichment |
| + Turn memory + pinning | gpt-oss-120b | 0.786 | Zero-width stamps, explicit-name pin |
| Latest (qwen3-32b eval) | qwen3-32b sim | ~0.75 | After all architectural fixes |

---

## License

© 2026. Built for the SHL Labs AI Intern Assignment.
