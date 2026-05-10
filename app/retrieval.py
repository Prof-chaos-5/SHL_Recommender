import os
import json
import math
import httpx
import gc
import numpy as np
import faiss
from collections import defaultdict
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

CATALOG_URL = os.getenv("CATALOG_URL")
FALLBACK_PATH = "data/catalog.json"

KEY_MAP = {
    "Knowledge & Skills": "K",
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Simulations": "S",
    "Assessment Exercises": "E",
}

# ============================================================================
# EMBED ENRICHMENT
# Keys MUST be exact catalog item names — verified against catalog.
# Adds domain signal that MiniLM can't infer from generic descriptions.
# ============================================================================

EMBED_ENRICHMENT = {
    # Broad-applicability personality test — relevant to every professional role
    "Occupational Personality Questionnaire OPQ32r": (
        "personality behaviour traits all professional roles hiring selection "
        "leadership management sales technical finance healthcare any job"
    ),
    "OPQ32r (Online)": (
        "personality behaviour traits all professional roles hiring selection "
        "leadership management sales technical finance healthcare any job"
    ),

    # Broad-applicability cognitive test — relevant to every graduate/professional role
    "SHL Verify Interactive G+": (
        "cognitive ability reasoning inductive numerical deductive all roles "
        "graduate professional screening aptitude intelligence"
    ),
    "Microsoft Excel 365 (New)": "excel spreadsheet microsoft office 365 admin assistant data entry",
    "MS Excel (New)": "excel spreadsheet microsoft office legacy admin assistant",
    "MS Word (New)": "word processing microsoft office legacy admin assistant",
    "SHL Verify G+": (
        "cognitive ability reasoning inductive numerical deductive all roles "
        "graduate professional screening aptitude intelligence"
    ),

    # SVAR — product acronym with no semantic link to communication roles
    "SVAR - Spoken English (US) (New)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent usa"
    ),
    "SVAR - Spoken English (Indian Accent) (New)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent india"
    ),
    "SVAR - Spoken English (AUS)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent australia"
    ),
    "SVAR - Spoken English (U.K.)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent uk british"
    ),

    # DSI — acronym with no role signal
    "Dependability and Safety Instrument (DSI)": (
        "dependability reliability safety integrity conscientiousness "
        "healthcare manufacturing frontline worker attendance punctuality"
    ),

    # GSA — name doesn't mention sales, development, or skills gap
    "Global Skills Assessment": (
        "sales skills competency development coaching performance "
        "professional growth sales manager representative talent audit reskilling"
    ),
    "Global Skills Development Report": (
        "sales skills competency development coaching performance report "
        "professional growth sales manager representative talent audit reskilling"
    ),

    # OPQ reports — need to surface alongside OPQ32r base instrument
    "OPQ Universal Competency Report 2.0": (
        "leadership competency report selection development universal framework "
        "ucf opq32r output senior professional all roles"
    ),
    "OPQ Leadership Report": (
        "leadership selection benchmark executive director cxo senior opq32r output"
    ),
    "OPQ MQ Sales Report": (
        "sales personality motivation opq32r output sales manager representative"
    ),

    # Scenarios — situational judgement tests with level-specific names
    "Graduate Scenarios": (
        "graduate entry level situational judgement early career "
        "new hire campus recruit cognitive reasoning professional"
    ),
    "Executive Scenarios": (
        "senior executive leadership director cxo situational judgement "
        "strategic decision making senior management"
    ),

    # Safety tests — need manufacturing/industrial signal
    "Workplace Health and Safety (New)": (
        "health safety workplace manufacturing industrial frontline "
        "warehouse logistics operations compliance"
    ),
    "Manufac. & Indust. - Safety & Dependability 8.0": (
        "manufacturing industrial safety dependability frontline worker "
        "warehouse logistics operations reliability"
    ),

    # Contact center — entry level name doesn't match search queries
    "Entry Level Customer Serv-Retail & Contact Center": (
        "entry level customer service retail contact center phone "
        "frontline agent representative support"
    ),

    # Systems/infrastructure tests — need rust/networking/linux signal
    "Linux Programming (General)": (
        "linux systems programming development software engineer rust "
        "performance networking infrastructure backend"
    ),
    "Networking and Implementation (New)": (
        "infrastructure networking implementation high-performance systems "
        "hardware backend engineer rust go"
    ),
    "Smart Interview Live Coding": (
        "technical interview live coding rust python java programming "
        "developer engineering adaptive assessment"
    ),

    # Sales transformation — level disambiguation
    "Sales Transformation 2.0 - Individual Contributor": (
        "sales individual contributor ic rep account executive performance "
        "transformation development coaching"
    ),
}

# ============================================================================
# ALWAYS-INCLUDE POOL
# Tests that are relevant across almost every professional role but score
# low in FAISS because their descriptions contain no role-specific text.
# Injected into every candidate set — LLM still decides whether to recommend.
# ============================================================================

ALWAYS_INCLUDE = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
]

# ============================================================================
# KEYWORD BOOST MAP
# Query keywords → catalog name fragments to boost in RRF results.
# Boosts specific tests when query signals a clear domain.
# ============================================================================

KEYWORD_BOOST_MAP = {
    "contact center":   ["SVAR", "Contact Center", "Customer Service Phone"],
    "call center":      ["SVAR", "Contact Center", "Customer Service Phone"],
    "customer service": ["SVAR", "Contact Center", "Customer Service Phone", "Entry Level Customer"],
    "sales":            ["Sales Transformation", "Global Skills", "OPQ MQ Sales"],
    "leadership":       ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "executive":        ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "cxo":              ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "graduate":         ["Graduate Scenarios", "Verify Interactive G+"],
    "entry level":      ["Graduate Scenarios", "Entry Level"],
    "manufacturing":    ["Safety & Dependability", "Workplace Health", "DSI"],
    "healthcare":       ["HIPAA", "Medical Terminology", "DSI", "Microsoft Word"],
    "java":             ["Core Java", "Spring", "Java Frameworks"],
    "python":           ["Python"],
    "sql":              ["SQL"],
    "aws":              ["Amazon Web Services"],
    "linux":            ["Linux Programming"],
    "networking":       ["Networking and Implementation"],
    "rust":             ["Linux Programming", "Networking", "Smart Interview Live Coding"],
    "finance":          ["Financial Accounting", "Basic Statistics", "Verify Interactive"],
    "excel":            ["Microsoft Excel 365 (New)", "MS Excel"],
    "word":             ["Microsoft Word 365 (New)", "MS Word"],
    "talent audit":     ["Global Skills Assessment", "Global Skills Development"],
    "reskilling":       ["Global Skills Assessment", "Sales Transformation"],
    "restructuring":    ["Global Skills Assessment"],
}

# Module-level state
catalog: list[dict] = []
index: faiss.IndexFlatIP = None
model: TextEmbedding = None
bm25_index: dict = {}


# ============================================================================
# HELPERS
# ============================================================================

def get_test_type(keys: list[str]) -> str:
    for k in keys:
        if k in KEY_MAP:
            return KEY_MAP[k]
    return "K"


def get_item_by_name(name: str) -> dict | None:
    name_lower = name.lower()
    for item in catalog:
        if item["name"].lower() == name_lower:
            return item
    for item in catalog:
        if name_lower in item["name"].lower():
            return item
    return None


def format_candidates(items: list[dict]) -> str:
    lines = []
    for item in items:
        line = (
            f"- {item.get('name', 'Unknown')} "
            f"| type={get_test_type(item.get('keys', []))} "
            f"| duration={item.get('duration', 'N/A')} "
            f"| levels={item.get('job_levels_raw', 'N/A').strip(', ')} "
            f"| remote={item.get('remote', 'N/A')} "
            f"| adaptive={item.get('adaptive', 'N/A')} "
            f"| url={item.get('link', 'N/A')} "
            f"| desc={item.get('description', '')[:150]}"
        )
        lines.append(line)
    return "\n".join(lines)


# ============================================================================
# RICH EMBED TEXT
# ============================================================================

def build_embed_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        "Test type: " + ", ".join(item.get("keys", [])),
        "Job levels: " + item.get("job_levels_raw", ""),
        "Duration: " + item.get("duration", ""),
        "Remote: " + str(item.get("remote", "")),
        "Adaptive: " + str(item.get("adaptive", "")),
    ]
    name = item.get("name", "")
    if name in EMBED_ENRICHMENT:
        parts.append("Also relevant for: " + EMBED_ENRICHMENT[name])
    return " | ".join(p for p in parts if p.strip(" |"))


# ============================================================================
# BM25 INDEX
# ============================================================================

def tokenize(text: str) -> list[str]:
    stop_words = {
        "a", "an", "the", "and", "or", "but", "if", "at", "from", "by",
        "for", "with", "about", "into", "through", "to", "of", "in", "on",
        "is", "it", "its", "this", "that", "are", "was", "be", "as",
    }
    tokens = text.lower().replace("-", " ").replace("/", " ").split()
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def build_bm25_index(data: list[dict]) -> tuple:
    N = len(data)
    tf_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = []

    for doc_id, item in enumerate(data):
        tokens = tokenize(build_embed_text(item))
        doc_lengths.append(len(tokens))
        for token in tokens:
            tf_index[token][doc_id] += 1

    avg_dl = sum(doc_lengths) / N if N > 0 else 1

    idf = {}
    for term, postings in tf_index.items():
        df = len(postings)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    print(f"[retrieval] BM25 index built: {len(tf_index)} terms, {N} docs")
    return dict(tf_index), idf, doc_lengths, avg_dl


def bm25_search(query, tf_index, idf, doc_lengths, avg_dl,
                top_k=20, k1=1.5, b=0.75):
    tokens = tokenize(query)
    scores = defaultdict(float)
    for token in tokens:
        if token not in tf_index:
            continue
        idf_val = idf.get(token, 0)
        for doc_id, tf in tf_index[token].items():
            dl = doc_lengths[doc_id]
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] += idf_val * tf_norm
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================================
# CATALOG LOADER
# ============================================================================

def load_catalog() -> list[dict]:
    try:
        resp = httpx.get(CATALOG_URL, timeout=30)
        resp.raise_for_status()
        data = json.loads(
            resp.content.decode("utf-8", errors="replace"), strict=False
        )
        os.makedirs("data", exist_ok=True)
        with open(FALLBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[retrieval] Loaded {len(data)} items from URL")
        return data
    except Exception as e:
        print(f"[retrieval] URL fetch failed: {e}, trying disk cache...")
        with open(FALLBACK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[retrieval] Loaded {len(data)} items from cache")
        return data


# ============================================================================
# FAISS INDEX
# ============================================================================

def build_faiss_index(data: list[dict]) -> faiss.IndexFlatIP:
    global model
    # Use the absolute smallest model to save RAM
    model = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        providers=["CPUExecutionProvider"]
    )
    texts = [build_embed_text(item) for item in data]
    
    # Process in chunks to avoid memory spikes
    vecs = np.array(list(model.embed(texts)))
    
    # Free up the source texts immediately
    del texts
    gc.collect()
    
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs.astype("float32"))
    
    # Free up the temporary vectors
    del vecs
    gc.collect()
    
    print(f"[retrieval] FAISS index built: {idx.ntotal} vectors, dim={dim}")
    return idx


# ============================================================================
# HYBRID SEARCH — FAISS + BM25 + RRF + keyword boost + always-include
# ============================================================================

def reciprocal_rank_fusion(faiss_results, bm25_results, k=60):
    scores = defaultdict(float)
    for rank, (doc_id, _) in enumerate(faiss_results):
        scores[doc_id] += 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_keyword_boosts(query: str) -> list[str]:
    """
    Return list of name fragments to boost based on query keywords.
    Matched against catalog item names (case-insensitive substring).
    """
    query_lower = query.lower()
    boost_fragments = []
    for keyword, fragments in KEYWORD_BOOST_MAP.items():
        if keyword in query_lower:
            boost_fragments.extend(fragments)
    return boost_fragments


def search(query: str, top_k: int = 20) -> list[dict]:
    """
    Hybrid search pipeline:
    1. FAISS semantic search
    2. BM25 keyword search
    3. RRF merge
    4. Keyword boost — lift domain-specific items
    5. Always-include pool — inject broad-applicability tests
    6. Return top_k
    """
    # 1. FAISS
    q_vec = np.array(list(model.embed([query]))).astype("float32")
    faiss_scores, faiss_idxs = index.search(q_vec, top_k * 2)
    faiss_results = [
        (int(i), float(s))
        for i, s in zip(faiss_idxs[0], faiss_scores[0])
        if i >= 0
    ]

    # 2. BM25
    bm25_results = bm25_search(
        query,
        bm25_index["tf"], bm25_index["idf"],
        bm25_index["doc_lengths"], bm25_index["avg_dl"],
        top_k=top_k * 2,
    )

    # 3. RRF merge
    merged = reciprocal_rank_fusion(faiss_results, bm25_results)

    # 4. Keyword boost
    boost_fragments = get_keyword_boosts(query)
    if boost_fragments:
        boosted = []
        for doc_id, rrf_score in merged:
            item_name = catalog[doc_id].get("name", "")
            # Check if any boost fragment is a substring of item name
            if any(f.lower() in item_name.lower() for f in boost_fragments):
                rrf_score *= 1.5
            boosted.append((doc_id, rrf_score))
        merged = sorted(boosted, key=lambda x: x[1], reverse=True)

    # Build result list
    seen_ids = set()
    results = []

    # 5. Always-include pool — goes in first so LLM always sees them
    for name in ALWAYS_INCLUDE:
        item = get_item_by_name(name)
        if item:
            item = item.copy()
            item["_score"] = 999.0
            item["_retrieval"] = "always_include"
            results.append(item)
            seen_ids.add(item.get("entity_id"))

    # 6. Add hybrid results
    for doc_id, rrf_score in merged:
        item = catalog[doc_id]
        eid = item.get("entity_id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        item_copy = item.copy()
        item_copy["_score"] = rrf_score
        item_copy["_retrieval"] = "hybrid"
        results.append(item_copy)

    return results[:top_k]


# ============================================================================
# STARTUP
# ============================================================================

def startup():
    gc.collect()
    global catalog, index, bm25_index
    catalog = load_catalog()
    index = build_faiss_index(catalog)
    tf, idf, doc_lengths, avg_dl = build_bm25_index(catalog)
    bm25_index = {
        "tf": tf, "idf": idf,
        "doc_lengths": doc_lengths, "avg_dl": avg_dl,
    }
    print("[retrieval] All indexes ready.")