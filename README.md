# SHL Assessment Recommender Agent

A production-grade conversational AI agent designed to help recruiters and hiring managers find the perfect SHL assessments for their roles. While I initially implemented a **Hybrid RRF (FAISS + BM25)** approach, I pivoted to a specialized **BM25 Lite Engine** to achieve superior accuracy within tight memory constraints.

Read the full technical deep dive in the **[Approach & Architecture Document](APPROACH.md)**.

## Key Features

- **BM25 Lite Retrieval (Mean Recall@10: 0.716)**: An optimized keyword engine that outperforms heavy ML models on this specific catalog by using domain-specific enrichment and weighted boosts.
- **Extreme Memory Optimization**: Runs on ~200MB RAM, making it perfectly stable on standard cloud free tiers (e.g., Render Free Tier).
- **Hardened Guardrails**: 
  - **Groundedness Filter**: Strict validation layer that ensures every recommended assessment exists in the source catalog.
  - **Clarification Loop**: Intelligently identifies vague queries and asks for missing context (Level, Role, Experience).
- **Domain-Specific Intelligence**:
  - **Mandatory Assessment Pillar**: Enforces SHL's foundational testing standards (OPQ32r for professional roles, G+ for cognitive screening).
  - **Intelligent Proxying**: Gracefully handles tech stacks not in the catalog (e.g., Rust -> Linux/Networking) using specialized semantic enrichment.
- **Windows-Optimized Infrastructure**: 
  - Automated "Zombie Process" cleanup and Port 8001 standardization for deployment stability.
  - ASCII-safe logging for full compatibility with Windows CMD/PowerShell environments.

## Tech Stack

- **Framework**: FastAPI (Stateless API)
- **AI Models**: OpenAI (gptoss120b)
- **Vector DB**: FAISS (In-memory for ultra-low latency)
- **Embeddings**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Evaluation**: Custom-built Smart Evaluator (LLM-based user simulation)

## Installation & Setup

1. **Clone and Install**:
   ```powershell
   git clone <repo-url>
   cd shl-recommender
   pip install -r requirements.txt
   ```

2. **Environment**:
   Copy `.env.example` to `.env` and add your OpenAI API key and the Catalog URL.
   ```powershell
   cp .env.example .env
  
   ```

3. **Run**:
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

## Evaluation Rigor

This project moved beyond "vibe-coding" by implementing a formal evaluation loop:
- **Baseline**: 0.320 Recall@10
- **Optimized**: 0.735 Recall@10
- **Improvements**: Switch to Hybrid RRF (+20% gain), Semantic Enrichment (+15% gain), and Name Canonicalization (+12.9% gain).

## License
© 2026. Built for the SHL Labs AI Intern Assignment.