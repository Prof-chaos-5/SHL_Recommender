# SHL Assessment Recommender Agent - Architecture & Approach

## 1. Design Philosophy: Lean Systems Over Bloated Frameworks
The core philosophy for this build was **Stateless Simplicity and Hardware-Aware Engineering**. While the system initially utilized a **Hybrid RRF (FAISS + BM25)** pipeline, I made a strategic pivot to a custom **BM25 Lite Retrieval Engine** to guarantee stability on constrained hardware.

Every architectural decision was driven by measuring **Recall@10** against evaluation traces. This metrics-first approach allowed for a major pivot when hardware constraints (Render's 512MB RAM) threatened stability, resulting in a final system that is both accurate (**Mean Recall@10: 0.716**) and extremely stable.

### The Problem-Solution Matrix
| Bottleneck Identified | Engineering Solution | Impact |
|-----------------------|----------------------|--------|
| **Memory Exhaustion** (512MB limit) | **BM25 Lite Engine** (Removed ONNX/FAISS) | RAM usage dropped from 600MB+ to ~200MB |
| **Embedding Domain Shift** | Manual **Domain-Specific Enrichment** | Improved precision on opaque test names (SVAR, OPQ, DSI) |
| **Environment Build Errors** | Python 3.11.9 Pinning + Pre-built Wheels | Bypassed Rust/Cargo compilation failures on Render |
| **Trace-Agent Divergence** | LLM-powered **Smart Evaluator** | Accurate multi-turn conversation grading (3-8 turns) |

## 2. Hardware-Aware Retrieval (The BM25 Lite Engine)
Traditional RAG setups often default to vector embeddings (FAISS/Chroma). However, for a specialized catalog of 377 items, local embedding models (like `all-MiniLM-L6-v2`) proved too heavy for the 512MB RAM target.

1. **Lite Engine Strategy:** I replaced the entire ML-based retrieval layer with an optimized **BM25 keyword engine**. By removing the ONNX runtime and shared libraries, I cleared over 400MB of RAM while maintaining comparable search quality.
2. **Domain Enrichment:** To solve the lack of "semantic" understanding in BM25, I manually enriched catalog items with industry keywords. For example, "OPQ32r" was enriched with "senior leadership, executive, benchmark," ensuring it surfaces for high-level management queries.
3. **Weighted Keyword Boosting:** Implemented a `KEYWORD_BOOST_MAP` that applies a 1.5x multiplier to the BM25 scores of specific assessment fragments when clear intent (e.g., "sales", "rust", "contact center") is detected in the query.

## 3. Agentic Logic & Behavioral Guardrails
Powered by **Llama 3.3 70B (Groq)**, the agent's logic is tightly constrained by a system prompt designed to pass strict behavioral probes:
- **Clarification Loop:** The system analyzes initial queries. If vague, it asks exactly one clarifying question about Level, Role, or Experience.
- **Groundedness Filter:** A backend validation layer ensures the agent only recommends assessments that physically exist in the source catalog, effectively eliminating hallucinations.
- **Out-of-Scope Rejection:** Explicit prompt triggers ensure the agent refuses to answer legal, medical, or compliance questions (e.g., HIPAA requirements), returning an empty `[]` array for recommendations.

## 4. Evaluation Rigor: The "Smart Evaluator"
Replaying static traces verbatim is brittle because a dynamic LLM will ask different questions than a historical agent. To accurately measure the **Mean Recall@10 of 0.716**, I used a custom **Smart Evaluator**:
- **Persona Roleplay:** A secondary LLM (the "User") acts as the hiring manager, responding to the agent's dynamic questions based on extracted facts from the original traces.
- **Robustness Testing:** This process verified that the agent stays on track over 3-8 turns and successfully incorporates mid-conversation feedback (e.g., "Add a cognitive test" or "Remove OPQ").

## 5. Deployment & Stability
Built for production readiness on the Render Free Tier:
- **Python 3.11.9 Pinning**: To avoid build-time errors caused by Rust compilation (common with Pydantic-core in newer Python versions), the environment is pinned to ensure pre-built binary wheels are used.
- **Data Resiliency**: The system features byte-level decoding for the SHL catalog and local disk caching to ensure 100% startup reliability regardless of network availability.

## 6. Reflection & Trade-offs
I actively chose a **Stateless API** over a stateful framework. This simplifies horizontal scaling, makes isolated testing trivial, and results in a significantly more robust service. By prioritizing **BM25 + Manual Enrichment** over "fancy" but memory-heavy vector embeddings, I achieved a "Production-Correct" solution that is stable, fast, and highly accurate.