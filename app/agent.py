"""
Agent orchestrator.

    POST /chat
        |
        v
    Analyzer            (LLM  -- understand)
        |
        v
    Controller          (Python -- decide)
        |
   +----+----+-----------+----------+
   |         |            |          |
 REFUSE  SLOT_FILL     RECOMMEND   COMPARE
   |         |            |          |
   |         |     Retriever + business_rules (Python)
   |         |            |          |
   |         |         Prompt Builder + LLM (explain only)
   |         |            |          |
   +---------+----- Validators (Python -- guarantee) ----+
                          |
                     ChatResponse

REFUSE and SLOT_FILL never touch the LLM a second time — the Analyzer's
output is already everything they need. Only RECOMMEND and COMPARE spend a
second LLM call, and only to explain a pre-selected, pre-validated candidate
pool — the LLM never has the freedom to introduce an item outside it.
"""

import json
import re
import time

from app.models import ChatResponse, Message
from app.agents.analyzer import analyze
from app.agents.controller import route
from app.agents import business_rules
from app.agents import heuristics
from app.retrieval import search, get_item_by_name, format_candidates
from app.prompts import (
    build_recommend_prompt,
    build_compare_prompt,
    build_slot_fill_reply,
    build_refusal_reply,
)
from app.agents.validators import ground_and_repair, schema_check
from app.llm_client import call_llm

# Evaluator allows 30s per call; we leave headroom so Python-side formatting
# and validation always has time to run after the LLM responds.
REQUEST_DEADLINE_SECONDS = 25


def _build_retrieval_query(state, messages: list[Message]) -> str:
    """
    Builds the text handed to BM25 + the keyword-boost/always-include maps
    in retrieval.py.

    FIX: previously this was role/seniority/skills + only the *latest* user
    message. That silently dropped query-triggered context that was only
    ever stated once, early in the conversation, and never resolves into a
    role/seniority/skill field on ConversationState -- things like "talent
    audit", "reskilling", "restructuring". A user who opens with "we need
    to re-skill our Sales org as part of our talent audit" and later just
    confirms selections ("we'll use OPQ and add MQ") would, by turn 3, have
    a retrieval query containing none of the words that made
    QUERY_ALWAYS_INCLUDE / KEYWORD_BOOST_MAP fire on turn 1 -- so Global
    Skills Assessment/Report and the sales-specific OPQ MQ Sales Report
    would silently drop out of the candidate pool on later turns even
    though they were clearly still the point of the conversation.

    Folding in the first user message (not just the latest) is a cheap,
    robust fix: opening context is rarely invalidated by later turns, and
    it costs nothing extra in LLM calls since this is pure string
    concatenation feeding a local BM25 index, not a prompt.
    """
    first_user = next((m.content for m in messages if m.role == "user"), "")
    latest_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    parts = [
        state.role or "",
        state.seniority or "",
        " ".join(state.skills),
        first_user,
    ]
    if latest_user != first_user:
        parts.append(latest_user)
    return " ".join(p for p in parts if p)

# Cleaner: dedicated separator outside the digit set
_ZW_DIGITS = ["\u200b", "\u200c", "\u200d", "\u2060", "\u2064",
              "\u206a", "\u206b", "\u206c", "\u206d", "\u206e"]
_ZW_ITEM_SEP = "\u206f"

def _encode_ids(ids: list[str]) -> str:
    out = []
    for i, eid in enumerate(ids):
        if i:
            out.append(_ZW_ITEM_SEP)
        out.append("".join(_ZW_DIGITS[int(d)] for d in str(eid) if d.isdigit()))
    return "".join(out)

def _decode_ids(blob: str) -> list[str]:
    rev = {c: str(i) for i, c in enumerate(_ZW_DIGITS)}
    ids, cur = [], []
    for ch in blob:
        if ch == _ZW_ITEM_SEP:
            if cur:
                ids.append("".join(cur)); cur = []
        elif ch in rev:
            cur.append(rev[ch])
    if cur:
        ids.append("".join(cur))
    return ids

def _stamp_recommendation_ids(reply: str, recommendations: list) -> str:
    ids = [r.entity_id for r in recommendations if r.entity_id]
    return f"{reply}{_encode_ids(ids)}" if ids else reply

def _prior_recommendation_ids(messages: list[Message]) -> list[str]:
    for m in reversed(messages):
        if m.role == "assistant":
            ids = _decode_ids(m.content)
            if ids:
                return ids
    return []

def _pin_prior(prior_ids: list[str], candidates: list[dict], excluded_names: set[str] | None = None) -> list[dict]:
    """
    FIX (C10): a previously-stamped id (e.g. OPQ32r recommended two turns
    ago) used to get re-pinned as a forced candidate every subsequent turn
    regardless of what the user said in between. If the user has since
    explicitly asked to remove that item, it must not come back in through
    this path either — excluded_names is checked here in addition to
    validators.ground_and_repair's own check, so an excluded item never
    even enters the candidate pool as "previous_turn"-sourced.
    """
    from app.retrieval import get_item_by_id
    excluded_lower = {n.strip().lower() for n in (excluded_names or set())}
    seen = {c.get("entity_id") for c in candidates}
    pinned = []
    for pid in prior_ids:
        item = get_item_by_id(pid)
        if not item or (item.get("name") or "").lower() in excluded_lower:
            continue
        if pid in seen:
            for c in candidates:
                if c.get("entity_id") == pid:
                    c["_score"] = max(c.get("_score", 0), 998.0)
                    if c.get("_retrieval") not in ("business_rule", "explicit_name"):
                        c["_retrieval"] = "previous_turn"
                    break
        else:
            item = item.copy()
            item["_score"] = 998.0
            item["_retrieval"] = "previous_turn"
            pinned.append(item)
            seen.add(pid)
    return pinned + candidates    


def _pin_explicit_names(names: list[str], candidates: list[dict]) -> list[dict]:
    """
    Resolves ConversationState.explicitly_named_items (anything the Analyzer
    saw named in the conversation -- including names that only ever appeared
    inside an earlier COMPARE reply, never in the retrieval-query blob) into
    real catalog items and pins them to the front of the candidate pool.

    FIX: without this, an item named only during a compare turn (e.g. "OPQ
    MQ Sales Report" surfaced while explaining a diff) and later asked for
    via "add MQ" has no path into the candidate pool -- _build_retrieval_query
    only concatenates the first + latest user message, so a short
    confirmation like "add MQ" is too weak for BM25 to hit on its own, and
    business_rules.py only force-includes rule-triggerable items, not
    specific names a user asked for by reference.
    """
    from app.retrieval import get_item_by_name
    seen = {c.get("entity_id") for c in candidates}
    pinned = []
    for name in names:
        item = get_item_by_name(name)
        if not item:
            continue
        eid = item.get("entity_id")
        if eid in seen:
            for c in candidates:
                if c.get("entity_id") == eid:
                    c["_score"] = max(c.get("_score", 0), 999.5)
                    c["_retrieval"] = "explicit_name"
                    break
        else:
            item = item.copy()
            item["_score"] = 999.5  # outranks business_rule/previous_turn (998) --
                                      # an item the user explicitly named this
                                      # conversation should never lose a slot to
                                      # a generic rule-triggered one.
            item["_retrieval"] = "explicit_name"
            pinned.append(item)
            seen.add(eid)
    return pinned + candidates

def _get_candidates(state, messages: list[Message]) -> list[dict]:
    if state.intent == "compare" and state.comparison_targets:
        items = [get_item_by_name(n) for n in state.comparison_targets]
        items = [i for i in items if i is not None]
        if items:
            return items[:20]

    query = _build_retrieval_query(state, messages)
    candidates = search(query, top_k=30)
    candidates = business_rules.apply(state, candidates)
    if state.intent == "recommend":
        excluded = set(getattr(state, "excluded_items", None) or [])
        candidates = _pin_prior(_prior_recommendation_ids(messages), candidates, excluded_names=excluded)
        candidates = _pin_explicit_names(state.explicitly_named_items, candidates)
    return candidates


def _parse_llm_json(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {raw[:200]}")
    return json.loads(match.group())


def _deterministic_fallback(candidates: list[dict], decision: str, excluded_names: set[str] | None = None) -> ChatResponse:
    """
    Used when the explainer LLM call fails or the time budget is exhausted.
    Bypasses the LLM entirely and takes the top-ranked candidates straight
    from retrieval/business_rules, so the response is always non-empty and
    schema-valid even under LLM failure.
    """
    ranked_ids = [
        c.get("entity_id")
        for c in sorted(candidates, key=lambda c: c.get("_score", 0), reverse=True)
    ]
    recommendations = schema_check(ground_and_repair(ranked_ids, candidates, excluded_names=excluded_names))
    reply = (
        "Here is a comparison based on our catalog data."
        if decision == "COMPARE"
        else "Here are assessments that match your criteria."
    )
    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=False)


def run_agent(messages: list[Message]) -> ChatResponse:
    start = time.monotonic()
    turn_count = len(messages)

    # 1. Understand — LLM extraction only, never decides what happens next.
    # On the first user message, try a free heuristic first: most turn-1
    # queries are either clearly complete ("hiring a Java developer") or
    # clearly vague ("I need an assessment"), and regex resolves both
    # without spending a Groq call. Only genuinely ambiguous turn-1 messages,
    # and every turn from turn 2 onward (which need real state-merging),
    # fall through to the Analyzer LLM.
    state = None
    if len(messages) == 1 and messages[0].role == "user":
        state = heuristics.try_fast_path(messages[0].content)
    if state is None:
        state = analyze(messages)

    # 2. Decide — pure Python, no LLM.
    prior_assistant_turns = sum(1 for m in messages if m.role == "assistant")
    decision = route(state, turn_count, prior_assistant_turns)

    # 3a. Refusals — deterministic template, no LLM call.
    if decision in ("REFUSE_INJECTION", "REFUSE_OFF_TOPIC"):
        return ChatResponse(
            reply=build_refusal_reply(decision),
            recommendations=[],
            end_of_conversation=False,
        )

    # 3b. Slot filling — deterministic template, no LLM call.
    if decision == "SLOT_FILL":
        return ChatResponse(
            reply=build_slot_fill_reply(state.missing_fields),
            recommendations=[],
            end_of_conversation=False,
        )

    # 3c. Closing — deterministic, no LLM call.
    if decision == "CLOSE":
        return ChatResponse(
            reply="Glad that works! Good luck with the hiring process.",
            recommendations=[],
            end_of_conversation=True,
        )

    # 3d/3e. RECOMMEND / COMPARE — retrieval + business rules (Python), then
    # a single LLM call to explain the pre-selected candidate pool.
    candidates = _get_candidates(state, messages)
    # Full pool stays available to the Grounding Validator for backfill;
    # only a trimmed slice goes into the prompt itself, since prompt tokens
    # are the main lever against Groq's free-tier TPM ceiling.
    prompt_candidates = sorted(candidates, key=lambda c: c.get("_score", 0), reverse=True)[:15]
    candidates_text = format_candidates(prompt_candidates)

    if time.monotonic() - start > REQUEST_DEADLINE_SECONDS:
        return _deterministic_fallback(candidates, decision, excluded_names=set(getattr(state, "excluded_items", None) or []))

    system = (
        build_compare_prompt(candidates_text)
        if decision == "COMPARE"
        else build_recommend_prompt(candidates_text)
    )
    history = [
        {"role": "user" if m.role == "user" else "assistant", "content": m.content}
        for m in messages
    ]

    try:
        raw = call_llm(system, history, json_mode=True)
        data = _parse_llm_json(raw)
    except Exception as e:
        print(f"[agent] explainer LLM failed, using deterministic fallback: {e}")
        return _deterministic_fallback(candidates, decision, excluded_names=set(getattr(state, "excluded_items", None) or []))

    # Grounding + schema validation (Python -- guarantee). For RECOMMEND
    # turns, ground_and_repair also force-includes any business_rule /
    # always_include candidate the LLM's reply doesn't explicitly
    # exclude -- see fix #2 in validators.py.
    recommendations = schema_check(
        ground_and_repair(
            data.get("selected_ids", []),
            candidates,
            excluded_names=set(getattr(state, "excluded_items", None) or []),
            force_include=(decision == "RECOMMEND"),
        )
    )

    reply = data.get("reply", "Here are some assessments that fit.")
    if decision == "RECOMMEND":
        reply = _stamp_recommendation_ids(reply, recommendations)

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=False,
    )