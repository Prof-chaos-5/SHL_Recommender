# SHL Assessment Recommender Agent

A production-grade conversational AI agent that helps recruiters and hiring managers find the right SHL assessments for their roles. The agent handles multi-turn conversations, comparison questions, slot-filling clarifications, and domain-specific business rules — all without hallucinating assessments not in the catalog.

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

**The LLM never decides what assessments exist — only how to explain the ones Python already selected.**

---

## Key Features

- **BM25 Lite Retrieval** — Pure Python BM25 with per-item synonym enrichment (`EMBED_ENRICHMENT`) and query-time keyword boost. No sentence-transformers, no ONNX, no FAISS. Memory footprint ≈ 210 MB, well within a 512 MB free-tier cloud instance.
- **Multi-tier Turn Memory** — Zero-width Unicode stamps in assistant replies carry entity IDs across turns. Items agreed on in turn 1 survive a comparison detour in turn 3 and reappear correctly in turn 4.
- **Deterministic Business Rules** — Domain-specific forced-inclusion for leadership, contact center, healthcare, and manufacturing/industrial roles. Rules fire every turn in Python — the LLM cannot forget them.
- **Explicit-Name Pinning** — Assessment names mentioned during a comparison turn (and later requested by the user) are resolved back into the candidate pool via a dedicated `explicitly_named_items` channel. Short confirmations like "add MQ" don't need BM25 to find the item.
- **Hardened Groundedness** — Two-pass Python validator: hallucinated IDs are dropped and backfilled; forced items are re-inserted at the end; deduplication is by entity_id, not name.
- **Fuzzy Catalog Resolution** — `get_item_by_name` uses a four-stage chain (exact → normalized → substring → difflib fuzzy) so hand-written catalog names in business rules never silently fail on dash/spacing/`(New)` variants.
- **Rate-Limit Resilient LLM Client** — 3 API keys rotated in round-robin, parsed retry-after from 429 responses, and a fast fallback model. Deterministic fallback if both models time out.
- **Smart Evaluator** — LLM-based user simulation (Groq Qwen) drives multi-turn conversations against live traces, computes Recall@10, and logs per-trace matches/misses. Supports `--all` or `--traces C1 C3` selective modes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (stateless) |
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