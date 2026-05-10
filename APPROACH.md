# SHL Assessment Recommender Agent - Architecture & Approach

## 1. Design Philosophy: Defensible Systems over "Vibe-Coding"
The assignment explicitly warns against "vibe-coding" and utilizing frameworks without understanding their underlying mechanics. My core philosophy for this build was **Stateless Simplicity and Evaluation-Driven Iteration**. 

Instead of hiding behind bloated orchestration frameworks like LangChain or LangGraph, I built a lean, highly debuggable pipeline using raw `FastAPI`, `faiss-cpu`, and direct API SDKs. Every architectural decision was driven by measuring **Recall@10** against the provided evaluation traces, which improved from a baseline of **0.32** to a final score of **0.735** through systematic bottleneck resolution, successfully exceeding initial targets.

### The Problem-Solution Matrix
| Bottleneck Identified | Engineering Solution | Impact |
|-----------------------|----------------------|--------|
| **Embedding Domain Shift** (MiniLM failing on SHL acronyms) | Hybrid RRF (FAISS + BM25) + Semantic Enrichment | Addressed low recall on specific tech/admin tests |
| **Broad Test Blindspots** (OPQ32r missing from candidates) | "Always-Include" Pool bypassing vector search | Restored missing foundation tests across traces |
| **Trace-Agent Divergence** (Static evals derailing) | LLM-powered "Smart Evaluator" with Persona Extraction | Enabled accurate multi-turn conversation grading |
| **Malformed Catalog Data** (JSON control character errors) | Custom strict-bypass parser + local disk caching | 100% startup reliability & zero network-fail crashes |

## 2. Advanced Retrieval (Solving the RAG "Domain Shift")
Pure vector search (FAISS + `all-MiniLM-L6-v2`) was insufficient because general embedding models lack SHL's specific HR domain context (e.g., matching "OPQ32r" to "Java Developer"). I implemented a three-layered hybrid retrieval engine:
1. **Semantic Enrichment:** I manually mapped opaque test names and acronyms (e.g., SVAR, DSI) to rich descriptive strings prior to indexing. This translates implicit human HR knowledge into explicit vector signals the embedding model can understand.
2. **Hybrid RRF Search:** By merging semantic FAISS search with a custom TF-IDF/BM25 index via Reciprocal Rank Fusion (RRF), I ensured that high-signal acronyms and exact matches are prioritized even if their semantic cosine similarity is mathematically distant.
3. **Always-Include Pool:** Tests with universal applicability across all roles (like *OPQ32r* and *Verify G+*) routinely failed semantic matching. I hardcoded these into a guaranteed candidate pool, shifting the responsibility from the retrieval engine (which lacks context) to the LLM (which decides relevance based on the conversation).

## 3. Agentic Logic & Behavioral Guardrails
Powered by **OpenAI gptoss120b**, the agent's logic is tightly constrained by a system prompt and backend heuristics designed to pass strict behavioral probes:
- **Turn 1 Strict Scope Check:** The system analyzes initial queries. If vague (e.g., "I need an assessment"), it restricts the LLM to asking exactly *one* clarifying question. If specific (role + domain provided), it skips clarification and recommends immediately.
- **JD Paste Detection Heuristics:** Implemented a string-matching heuristic to detect pasted Job Descriptions. If triggered, the agent is forced to extract role requirements and recommend instantly, bypassing standard multi-turn clarification.
- **Hallucination Safety Filter:** An absolute backend URL guardrail cross-references all agent-generated URLs against the FAISS candidate set. If the LLM hallucinates a URL, it is physically stripped from the `recommendations` array before the client receives it.
- **Out-of-Scope Rejection:** Explicit prompt triggers ensure the agent returns an empty array `[]` and refuses answering legal/compliance questions (e.g., HIPAA requirements) or prompt-injection attempts.

## 4. Evaluation Rigor: The "Smart Evaluator"
Replaying static markdown traces verbatim fails because a dynamic LLM agent will inevitably ask different clarifying questions than the historical agent did. To accurately measure the `Recall@10` of **0.735**, I built a custom **Smart Evaluator** (`smart_eval.py`):
- **Persona Extraction:** Pre-processes the markdown trace to extract known facts and historical Q&A pairs (e.g., what the user answered when asked about seniority).
- **LLM User Simulation:** Uses a secondary LLM with a low temperature (0.3) to roleplay the hiring manager. It responds naturally to the Agent's dynamic questions while strictly adhering to the extracted persona constraints.
- This allowed me to rigorously test the 8-turn cap limit and measure how well the agent handles mid-conversation refinement (e.g., "Actually, add a cognitive test").

## 5. Resilient Engineering & Environment Hardening
Built with production readiness in mind, the system handles edge cases gracefully:
- **Data Ingestion Fault Tolerance:** The SHL catalog contained invalid JSON control characters (literal `\n` in strings). I implemented a byte-level decoding phase with regex stripping and strict-mode overrides, backed by a local `data/catalog.json` disk cache to survive network timeouts.
- **Environment Stability (Windows Development):** Standardized process cleanup to prevent Windows port-locking on `8000/8001`, and mapped evaluator logging to ASCII-safe status markers (`[OK]`, `[MISS]`) to prevent character encoding crashes in standard terminals.

## 6. Reflection & Trade-offs
I actively chose a **Stateless API** over a stateful framework. While this requires passing the full conversation history payload on every POST request, it fundamentally aligns with scalable microservice design. It eliminates memory leak risks, simplifies horizontal scaling, makes isolated unit-testing trivial, and ultimately results in a significantly more robust, auditable, and debuggable service.

---
*Note: This architecture was developed iteratively, leveraging AI assistance for pair-programming, regex construction, metric evaluator writing, and environmental debugging, while all architectural design, prompt engineering, and RAG pipeline decisions remain entirely my own.*