"""
retrieval.py — BM25 Lite Mode
No sentence-transformers, no ONNX runtime, no FAISS.
Pure Python BM25 over enriched catalog text.

Memory profile:
  BM25 index (377 docs)  ~5MB
  Catalog JSON           ~3MB
  FastAPI + uvicorn      ~100MB
  OS overhead            ~100MB
  Total                  ~210MB  (well under 512MB free tier)
"""

import os
import json
import math
import httpx
import re
import difflib
from collections import defaultdict
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
# Adds domain signal for tests with opaque names.
# Critical for BM25 since it has zero semantic understanding —
# it can only match tokens that actually appear in the text.
# ============================================================================

EMBED_ENRICHMENT = {
    "Occupational Personality Questionnaire OPQ32r": (
        "personality behaviour traits all professional roles hiring selection "
        "leadership management sales technical finance healthcare any job opq"
    ),
    "OPQ32r (Online)": (
        "personality behaviour traits all professional roles hiring selection "
        "leadership management sales technical finance healthcare any job opq"
    ),
    "SHL Verify Interactive G+": (
        "cognitive ability reasoning inductive numerical deductive all roles "
        "graduate professional screening aptitude intelligence verify g+"
    ),
    "SHL Verify G+": (
        "cognitive ability reasoning inductive numerical deductive all roles "
        "graduate professional screening aptitude intelligence verify g+"
    ),
    "SVAR - Spoken English (US) (New)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent usa svar"
    ),
    "SVAR - Spoken English (Indian Accent) (New)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent india svar"
    ),
    "SVAR - Spoken English (AUS)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent australia svar"
    ),
    "SVAR - Spoken English (U.K.)": (
        "spoken english communication verbal fluency contact center customer "
        "service call center phone agent voice accent uk british svar"
    ),
    "Dependability and Safety Instrument (DSI)": (
        "dependability reliability safety integrity conscientiousness "
        "healthcare manufacturing frontline worker attendance punctuality dsi"
    ),
    "Global Skills Assessment": (
        "sales skills competency development coaching performance "
        "professional growth sales manager representative talent audit reskilling gsa"
    ),
    "Global Skills Development Report": (
        "sales skills competency development coaching performance report "
        "professional growth sales manager representative gsa"
    ),
    "OPQ Universal Competency Report 2.0": (
        "leadership competency report selection development universal framework "
        "ucf opq32r output senior professional all roles ucr"
    ),
    "OPQ Leadership Report": (
        "leadership selection benchmark executive director cxo senior opq32r output"
    ),
    "OPQ MQ Sales Report": (
        "sales personality motivation opq32r output sales manager representative mq"
    ),
    "Graduate Scenarios": (
        "graduate entry level situational judgement early career "
        "new hire campus recruit cognitive reasoning professional sjt"
    ),
    "Executive Scenarios": (
        "senior executive leadership director cxo situational judgement "
        "strategic decision making senior management sjt"
    ),
    "Workplace Health and Safety (New)": (
        "health safety workplace manufacturing industrial frontline "
        "warehouse logistics operations compliance"
    ),
    "Docker (New)": (
        "docker container containers kubernetes cloud native microservices devops software engineer"
    ),
    "Amazon Web Services (AWS) Development (New)": (
        "aws amazon web services cloud deployment cloud native infrastructure software engineer"
    ),
    "Manufac. & Indust. - Safety & Dependability 8.0": (
        "manufacturing industrial safety dependability frontline worker "
        "warehouse logistics operations reliability chemical plant operator"
    ),
    "Entry Level Customer Serv-Retail & Contact Center": (
        "entry level customer service retail contact center phone "
        "frontline agent representative support"
    ),
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
    "Sales Transformation 2.0 - Individual Contributor": (
        "sales individual contributor ic rep account executive performance "
        "transformation development coaching"
    ),
    "Microsoft Excel 365 (New)": (
        "excel spreadsheet microsoft office 365 admin assistant data entry"
    ),
    "Microsoft Word 365 (New)": (
        "word processing document microsoft office 365 admin assistant"
    ),
    "MS Excel (New)": (
        "excel spreadsheet microsoft office legacy admin assistant data entry"
    ),
    "MS Word (New)": (
        "word processing document microsoft office legacy admin assistant"
    ),
    "SHL Verify Interactive – Numerical Reasoning": (
    "numerical reasoning logic aptitude inference interpretation graduate professional"
),
"SHL Verify Interactive Numerical Calculation": (
    "numerical calculation arithmetic computation basic maths speed accuracy"
),
}

# ============================================================================
# LEGACY UPGRADES
# Maps legacy test names to their newer equivalents to automatically replace them.
# ============================================================================

LEGACY_UPGRADES = {
    "Verify - G+": "SHL Verify Interactive G+",
    "Verify - Numerical Ability": "SHL Verify Interactive – Numerical Reasoning",
}

# ============================================================================
# ALWAYS-INCLUDE POOL
# Injected into every candidate set regardless of query, tagged
# _retrieval="always_include" -- which validators.FORCED_SOURCES treats as
# mandatory, same tier as a business_rules.py force-include.
#
# FIX (C3): OPQ32r used to live here unconditionally. business_rules.py
# separately decides OPQ32r inclusion with real gating (suppressed for
# junior/frontline roles, respects personality_excluded and per-item
# excluded_items) -- but this list injected it into the candidate pool on
# every single call *before* business_rules.apply() ever ran, tagged as a
# forced source. So an entry-level/junior conversation where
# required_item_names() correctly omitted OPQ32r still got it back anyway,
# because this pool and validators.py both treated "always_include" as
# mandatory regardless of what business_rules decided. business_rules.py's
# gating was decorative as a result.
#
# Fix is to have exactly one owner for this decision: business_rules.py.
# Nothing belongs in this list unless it should genuinely be force-included
# with NO gating at all, for every conversation, forever. If a future item
# needs conditional inclusion, it belongs in business_rules.py's
# required_item_names(), not here.
# ============================================================================

ALWAYS_INCLUDE: list[str] = []

# ============================================================================
# QUERY-TRIGGERED ALWAYS-INCLUDE
# Injected only when a high-signal keyword appears in the retrieval query.
# Sits between ALWAYS_INCLUDE and BM25 results in the pipeline so the LLM
# always sees these items when the use-case matches — without polluting every
# other result set.
# ============================================================================

QUERY_ALWAYS_INCLUDE = {
    # Talent audit / reskilling — Global Skills suite rarely wins BM25
    # against sales-specific items but is often the correct answer.
    "talent audit":   ["Global Skills Assessment", "Global Skills Development Report"],
    "reskilling":     ["Global Skills Assessment", "Global Skills Development Report"],
    "re-skill":       ["Global Skills Assessment", "Global Skills Development Report"],
    "restructuring":  ["Global Skills Assessment"],
    # Medical-record / admin — Medical Terminology tends to score low on BM25
    # because the item description is terse; force-inject when context is clear.
    "patient record": ["Medical Terminology (New)", "HIPAA (Security)"],
    "medical record": ["Medical Terminology (New)", "HIPAA (Security)"],
    # High-performance networking/systems (e.g. Rust, C++) — explicitly include
    # Networking and Live Coding since BM25 might miss them.
    "rust":           ["Smart Interview Live Coding", "Networking and Implementation (New)", "Linux Programming (General)"],
    # Cognitive ability — inject Verify G+ when the role clearly needs it.
    "engineer":       ["SHL Verify Interactive G+"],
    "developer":      ["SHL Verify Interactive G+"],
    "analyst":        ["SHL Verify Interactive G+"],
    "graduate":       ["SHL Verify Interactive G+"],
    "scientist":      ["SHL Verify Interactive G+"],
    "cognitive":      ["SHL Verify Interactive G+"],
    "ability test":   ["SHL Verify Interactive G+"],
}

# ============================================================================
# KEYWORD BOOST MAP
# Query keywords → catalog name fragments to boost in BM25 results.
# Essential for BM25 since it has no semantic understanding —
# "contact center" won't naturally find "SVAR" without this.
# ============================================================================

KEYWORD_BOOST_MAP = {
    "contact center":   ["SVAR", "Contact Center", "Customer Service Phone"],
    "contact centre":   ["SVAR", "Contact Center", "Customer Service Phone"],
    "call center":      ["SVAR", "Contact Center", "Customer Service Phone"],
    "call centre":      ["SVAR", "Contact Center", "Customer Service Phone"],
    "customer service": ["SVAR", "Contact Center", "Customer Service Phone", "Entry Level Customer"],
    "inbound":          ["SVAR", "Contact Center", "Customer Service Phone"],
    "sales":            ["Sales Transformation", "Global Skills", "OPQ MQ Sales Report"],
    "reskilling":       ["Global Skills Assessment", "Sales Transformation"],
    "talent audit":     ["Global Skills Assessment", "Global Skills Development"],
    "restructuring":    ["Global Skills Assessment"],
    "leadership":       ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "executive":        ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "cxo":              ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "director":         ["OPQ Leadership", "Executive Scenarios", "OPQ Universal"],
    "graduate":         ["Graduate Scenarios", "Verify Interactive G+"],
    "entry level":      ["Graduate Scenarios", "Entry Level"],
    "manufacturing":    ["Safety & Dependability", "Workplace Health", "DSI"],
    "chemical":         ["Safety & Dependability", "Workplace Health", "DSI"],
    "plant operator":   ["Safety & Dependability", "Workplace Health", "DSI"],
    "healthcare":       ["HIPAA", "Medical Terminology", "DSI", "Microsoft Word"],
    "medical":          ["HIPAA", "Medical Terminology", "DSI"],
    "hipaa":            ["HIPAA"],
    "java":             ["Core Java", "Spring", "Java Frameworks"],
    "python":           ["Python"],
    "sql":              ["SQL"],
    "aws":              ["Amazon Web Services"],
    "linux":            ["Linux Programming"],
    "networking":       ["Networking and Implementation"],
    "rust":             ["Linux Programming", "Networking", "Smart Interview Live Coding"],
    "devops":           ["Docker", "Amazon Web Services", "Linux Programming"],
    "finance":          ["Financial Accounting", "Basic Statistics", "Verify Interactive"],
    "analyst":          ["Financial Accounting", "Basic Statistics", "Verify Interactive"],
    "excel":            ["Microsoft Excel 365 (New)", "Microsoft Excel 365 - Essentials", "MS Excel"],
    "word":             ["Microsoft Word 365 (New)", "Microsoft Word 365 - Essentials", "MS Word"],
    "admin":            ["Microsoft Excel", "Microsoft Word", "MS Excel", "MS Word"],
    "bilingual":        ["SVAR"],
    "spanish":          ["SVAR - Spoken Spanish"],
    "spoken":           ["SVAR"],
}

# Module-level state
catalog: list[dict] = []
bm25_state: dict = {}


# ============================================================================
# HELPERS
# ============================================================================

def get_test_type(keys: list[str]) -> str:
    for k in keys:
        if k in KEY_MAP:
            return KEY_MAP[k]
    return "K"


_NAME_INDEX_CACHE: dict[str, dict] = {}
_NAME_INDEX_CATALOG_ID: int = -1


def _normalize_name(name: str) -> str:
    """
    Collapses exactly the cosmetic variance that broke the old exact/
    substring-only matcher: dash vs space, "&" vs "and", a missing/extra
    "(New)" suffix, punctuation, and whitespace runs.
    """
    n = (name or "").lower()
    n = n.replace("&", "and")
    n = re.sub(r"\(new\)", "", n)
    n = re.sub(r"[\-\u2013\u2014/,]", " ", n)   # hyphen, en-dash, em-dash, slash, comma
    n = re.sub(r"[^\w\s\+\.]", " ", n)          # strip remaining punctuation, keep + and .
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _build_name_index() -> dict[str, dict]:
    """Cached normalized-name -> catalog item map. Rebuilt whenever the
    catalog object identity changes (i.e. after startup()/reload)."""
    global _NAME_INDEX_CACHE, _NAME_INDEX_CATALOG_ID
    if _NAME_INDEX_CATALOG_ID == id(catalog) and _NAME_INDEX_CACHE:
        return _NAME_INDEX_CACHE
    idx: dict[str, dict] = {}
    for item in catalog:
        norm = _normalize_name(item.get("name", ""))
        idx.setdefault(norm, item)
    _NAME_INDEX_CACHE = idx
    _NAME_INDEX_CATALOG_ID = id(catalog)
    return idx


def get_item_by_name(name: str) -> dict | None:
    """
    Resolves a hand-written catalog name (from business_rules.py,
    EMBED_ENRICHMENT, QUERY_ALWAYS_INCLUDE, LEGACY_UPGRADES, or an
    LLM-extracted comparison/explicit-name target) to the real catalog
    entry.

    FIX (C1): the previous implementation only tried (a) an exact
    case-insensitive match, then (b) "query is a substring of the catalog
    name." (b) only helps when the caller passes a short fragment (e.g.
    "OPQ32r"); every caller here actually writes out the *full* catalog-
    shaped name by hand, so a single differing token — a dash variant, a
    missing/extra "(New)", "&" vs "and", "Serv-Retail" spelled differently
    — made the lookup return None. And it failed *silently*: callers like
    business_rules.apply() just got back a shorter forced list with no
    signal anything was dropped (see the SVAR / Contact Center 50/50
    force-include split in the C1/C3 trace).

    New resolution order:
      1. exact case-insensitive match (unchanged, cheapest).
      2. normalized exact match — strips the cosmetic variance above.
      3. normalized substring match, either direction (so both "query is a
         fragment of the catalog name" and "catalog name is a fragment of
         the query" now work, not just the former).
      4. fuzzy best-match fallback (difflib) above a similarity floor, to
         catch the remaining single-token differences that neither exact
         nor substring matching can.
    A genuine miss is now logged loudly instead of disappearing quietly,
    so a dropped forced/named item is visible in the logs.
    """
    if not name:
        return None

    name_lower = name.lower().strip()

    # 1. Exact case-insensitive match.
    for item in catalog:
        if item.get("name", "").lower() == name_lower:
            return item

    idx = _build_name_index()
    name_norm = _normalize_name(name)

    # 2. Normalized exact match.
    if name_norm in idx:
        return idx[name_norm]

    # 3. Normalized substring match, either direction.
    for norm, item in idx.items():
        if norm and (name_norm in norm or norm in name_norm):
            return item

    # 4. Fuzzy fallback for single differing tokens that substring
    # matching can't catch because neither string contains the other.
    close = difflib.get_close_matches(name_norm, list(idx.keys()), n=1, cutoff=0.72)
    if close:
        matched = idx[close[0]]
        print(f"[retrieval] get_item_by_name: fuzzy-matched '{name}' -> "
              f"'{matched.get('name')}' (no exact/substring hit)")
        return matched

    print(f"[retrieval] get_item_by_name: NO MATCH for '{name}' — "
          f"forced/named item silently dropped from candidate pool")
    return None

def get_item_by_id(entity_id) -> dict | None:
    eid = str(entity_id)
    for item in catalog:
        if str(item.get("entity_id")) == eid:
            return item
    return None

def format_candidates(items: list[dict]) -> str:
    """
    Includes entity_id explicitly so the explainer LLM can select candidates
    by id (selected_ids) rather than retyping names — this is what lets the
    Grounding Validator do an exact id lookup instead of fuzzy name matching,
    closing off most hallucination paths before validation even runs.
    """
    lines = []
    for item in items:
        source = item.get("_retrieval", "search")
        line = (
            f"- id={item.get('entity_id', 'N/A')} "
            f"| name={item.get('name', 'Unknown')} "
            f"| type={get_test_type(item.get('keys', []))} "
            f"| duration={item.get('duration', 'N/A')} "
            f"| levels={item.get('job_levels_raw', 'N/A').strip(', ')} "
            f"| remote={item.get('remote', 'N/A')} "
            f"| adaptive={item.get('adaptive', 'N/A')} "
            f"| url={item.get('link', 'N/A')} "
            f"| source={source} "
            f"| desc={item.get('description', '')[:150]}"
        )
        lines.append(line)
    return "\n".join(lines)


# ============================================================================
# RICH TEXT FOR BM25
# More important here than with FAISS — BM25 can ONLY match tokens
# that literally appear in the document text. Every keyword matters.
# ============================================================================

def build_doc_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        "keys: " + " ".join(item.get("keys", [])),
        "levels: " + item.get("job_levels_raw", ""),
        "duration: " + item.get("duration", ""),
        "remote: " + str(item.get("remote", "")),
        "adaptive: " + str(item.get("adaptive", "")),
        "languages: " + " ".join(item.get("languages", [])),
    ]
    name = item.get("name", "")
    if name in EMBED_ENRICHMENT:
        parts.append(EMBED_ENRICHMENT[name])
    return " ".join(p for p in parts if p.strip())


# ============================================================================
# BM25
# ============================================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize for BM25. Keep stop words minimal —
    removing too many hurts BM25 recall on short catalog descriptions.
    """
    stop_words = {
        "a", "an", "the", "and", "or", "but", "if",
        "at", "from", "by", "for", "with", "into",
        "to", "of", "in", "on", "is", "it", "as",
    }
    # Preserve acronyms and product codes (OPQ32r, G+, AWS, etc.)
    text = text.lower()
    text = re.sub(r"[^\w\s\+\-\.]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def build_bm25(data: list[dict]) -> dict:
    N = len(data)
    tf_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = []

    for doc_id, item in enumerate(data):
        tokens = tokenize(build_doc_text(item))
        doc_lengths.append(len(tokens))
        for token in tokens:
            tf_index[token][doc_id] += 1

    avg_dl = sum(doc_lengths) / N if N > 0 else 1

    idf = {}
    for term, postings in tf_index.items():
        df = len(postings)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    print(f"[retrieval] BM25 index: {len(tf_index)} terms, {N} docs")
    return {
        "tf": dict(tf_index),
        "idf": idf,
        "doc_lengths": doc_lengths,
        "avg_dl": avg_dl,
    }


def bm25_score(query: str, top_k: int = 20,
               k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    tokens = tokenize(query)
    scores = defaultdict(float)
    tf = bm25_state["tf"]
    idf = bm25_state["idf"]
    doc_lengths = bm25_state["doc_lengths"]
    avg_dl = bm25_state["avg_dl"]

    for token in tokens:
        if token not in tf:
            continue
        idf_val = idf.get(token, 0)
        for doc_id, freq in tf[token].items():
            dl = doc_lengths[doc_id]
            tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] += idf_val * tf_norm

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================================
# KEYWORD BOOST
# ============================================================================

def get_boost_fragments(query: str) -> list[str]:
    query_lower = query.lower()
    fragments = []
    for keyword, frags in KEYWORD_BOOST_MAP.items():
        if keyword in query_lower:
            fragments.extend(frags)
    return fragments


# ============================================================================
# SEARCH
# ============================================================================

def search(query: str, top_k: int = 20) -> list[dict]:
    """
    BM25 search with keyword boost and always-include pool.

    Pipeline:
    1. BM25 score all docs
    2. Apply keyword boost for domain-specific terms
    3. Re-rank by boosted score
    4. Inject global always-include pool  (score 999)
    5. Inject query-triggered always-include (score 997)
    6. Fill from BM25 results
    7. Return top_k
    """
    # 1. BM25
    raw_results = bm25_score(query, top_k=top_k * 2)

    # 2. Keyword boost
    boost_fragments = get_boost_fragments(query)
    if boost_fragments:
        boosted = []
        for doc_id, score in raw_results:
            item_name = catalog[doc_id].get("name", "")
            if any(f.lower() in item_name.lower() for f in boost_fragments):
                score *= 1.5
            boosted.append((doc_id, score))
        raw_results = sorted(boosted, key=lambda x: x[1], reverse=True)

    # 3. Build result list
    seen_ids = set()
    results = []

    # 4. Global always-include pool
    for name in ALWAYS_INCLUDE:
        item = get_item_by_name(name)
        if item:
            item = item.copy()
            item["_score"] = 999.0
            item["_retrieval"] = "always_include"
            results.append(item)
            seen_ids.add(item.get("entity_id"))

    # 5. Query-triggered always-include — only fires when a high-signal
    # keyword appears in the query, so it doesn't pollute other result sets.
    query_lower = query.lower()
    for keyword, names in QUERY_ALWAYS_INCLUDE.items():
        if keyword in query_lower:
            for name in names:
                item = get_item_by_name(name)
                if item and item.get("entity_id") not in seen_ids:
                    item = item.copy()
                    item["_score"] = 997.0
                    item["_retrieval"] = "query_always_include"
                    results.append(item)
                    seen_ids.add(item.get("entity_id"))

    # 6. BM25 results
    for doc_id, score in raw_results:
        item = catalog[doc_id]
        eid = item.get("entity_id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        item_copy = item.copy()
        item_copy["_score"] = score
        item_copy["_retrieval"] = "bm25"
        results.append(item_copy)

    # 7. Upgrade legacy items
    upgraded = []
    final_seen = set()
    for c in results:
        name = c.get("name", "")
        if name in LEGACY_UPGRADES:
            new_item = get_item_by_name(LEGACY_UPGRADES[name])
            if new_item:
                new_item = new_item.copy()
                new_item["_score"] = c.get("_score", 0.0)
                new_item["_retrieval"] = "legacy_upgrade"
                c = new_item
        
        eid = c.get("entity_id")
        if eid not in final_seen:
            final_seen.add(eid)
            upgraded.append(c)

    return upgraded[:top_k]


# ============================================================================
# CATALOG LOADER
# ============================================================================

def load_catalog() -> list[dict]:
    try:
        resp = httpx.get(CATALOG_URL, timeout=30)
        resp.raise_for_status()
        data = json.loads(
            resp.content.decode("utf-8", errors="replace"),
            strict=False
        )
        # Clean up encoding replacement characters in catalog names
        for item in data:
            if "name" in item:
                name = item["name"]
                if "Verify Interactive" in name:
                    name = name.replace("\ufffd", "–")
                elif "360" in name:
                    name = name.replace("\ufffd", "°")
                elif name == "Microsoft Excel 365 - Essentials (New)":
                    name = "Microsoft Excel 365 (New)"
                item["name"] = name

        os.makedirs("data", exist_ok=True)
        with open(FALLBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[retrieval] Loaded {len(data)} items from URL")
        return data
    except Exception as e:
        print(f"[retrieval] URL fetch failed: {e}, trying cache...")
        with open(FALLBACK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Clean up encoding replacement characters in catalog names from cache
        for item in data:
            if "name" in item:
                name = item["name"]
                if "Verify Interactive" in name:
                    name = name.replace("\ufffd", "–")
                elif "360" in name:
                    name = name.replace("\ufffd", "°")
                elif name == "Microsoft Excel 365 - Essentials (New)":
                    name = "Microsoft Excel 365 (New)"
                item["name"] = name
        print(f"[retrieval] Loaded {len(data)} items from cache")
        return data


# ============================================================================
# STARTUP
# ============================================================================

def startup():
    global catalog, bm25_state
    catalog = load_catalog()
    bm25_state = build_bm25(catalog)
    print("[retrieval] BM25 lite mode ready. No ML engine loaded.")