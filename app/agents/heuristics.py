"""
Turn-1 heuristic fast path.

The Analyzer LLM call is only strictly necessary when state needs to be
*merged* across turns (refine, multi-turn compare) — something regex can't
reliably do. On the first user message there's no prior state to merge, so
a fast, free heuristic can resolve most conversations without touching an
LLM at all:

    clear role stated       -> skip Analyzer, go straight to RECOMMEND (1 LLM call total)
    JD paste                -> same
    obviously vague          -> SLOT_FILL, zero LLM calls
    injection / off-topic    -> REFUSE, zero LLM calls
    quoted compare request   -> skip Analyzer, go straight to COMPARE (1 LLM call total)
    anything ambiguous       -> fall through to the full Analyzer (unchanged behavior)

This exists purely to cut Groq call volume — under an automated 8-turn
tester hammering the endpoint with no pauses, halving LLM calls on the
highest-volume turn (turn 1) is the single biggest lever available before
needing more API capacity.
"""

import re

from app.models import ConversationState

# FIX: the previous pattern required the captured role phrase to start with
# a capital letter and only recognized a fixed singular suffix list missing
# common titles (assistant, agent, coordinator, operator, nurse, clerk).
# Real turn-1 messages are frequently lowercase, plural ("admin assistants",
# "plant operators", "graduate financial analysts"), and phrased with verbs
# other than "hiring a"/"need a" ("screen admin assistants", "need plant
# operators" with no article). The old pattern silently failed to extract a
# role that was clearly stated, forcing an unnecessary SLOT_FILL turn and
# burning the ask-at-most-once clarification budget (e.g. C8 in eval).
#
# This version:
#   - is case-insensitive on the captured group, not just the anchor phrase
#   - accepts plural role nouns via an optional trailing "s"
#   - accepts "need <plural noun>" without a preceding article
#   - broadens the trigger verbs and the suffix list
# A separate filler-word strip (see _extract_role) trims leading words like
# "to", "quickly", "screen" that the lazy anchor match can otherwise pull
# into the captured role text.
ROLE_SIGNAL_PATTERN = re.compile(
    r"\b(?:hiring|looking for|need(?:s)?(?:\s+a)?|recruiting|screen|screening)\b.{0,60}?"
    r"\b([a-zA-Z][a-zA-Z\-]*(?:\s+[a-zA-Z][a-zA-Z\-]*){0,2}\s*"
    r"(?:developer|engineer|analyst|manager|administrator|representative|"
    r"scientist|designer|specialist|technician|consultant|associate|lead|"
    r"assistant|agent|coordinator|operator|nurse|clerk)s?)\b",
    re.IGNORECASE,
)

# Words the lazy anchor match can drag into the front of the captured role
# phrase (verbs/fillers between the trigger and the actual title). Stripped
# after extraction, same idea as the existing "a/an/the" strip below.
_LEADING_FILLERS = {
    "a", "an", "the", "to", "quickly", "screen", "screening", "please",
    "urgently", "immediately", "some", "our", "these", "those", "new", "my",
}

JD_SIGNALS = [
    "job description", "jd:", "responsibilities", "requirements",
    "qualifications", "years of experience", "we are looking for",
    "the role", "you will", "must have", "nice to have", "about the role",
]

CLOSING_PHRASES = {"thanks", "thank you", "perfect", "that works", "great",
                    "looks good", "all set", "sounds good", "awesome"}

OFF_TOPIC_KEYWORDS = [
    # Legal liability / employment law questions (not assessment selection)
    "legally required", "sue us", "lawsuit", "are we liable",
    # Compensation / pay questions
    "salary", "compensation range", "how much should we pay",
    # Actual medical / clinical questions (not admin/healthcare hiring)
    "medical diagnosis", "prescri",
    # Clearly non-hiring topics
    "who should we vote", "political",
    # NOTE: "hipaa" is intentionally absent — "HIPAA compliance" in a
    # hiring context almost always means the HIPAA (Security) SHL test,
    # not a legal question. The LLM analyzer handles genuine edge cases.
]

INJECTION_KEYWORDS = [
    "ignore previous instructions", "ignore all previous", "disregard your instructions",
    "disregard all prior", "you are now", "new instructions:", "reveal your system prompt",
    "reveal your instructions", "developer mode", "jailbreak", "act as if you have no",
]


def _is_jd_paste(text: str) -> bool:
    low = text.lower()
    signal_count = sum(1 for s in JD_SIGNALS if s in low)
    return signal_count >= 2 and len(text) > 200


def _extract_role(text: str) -> str | None:
    match = ROLE_SIGNAL_PATTERN.search(text)
    if not match:
        return None
    words = match.group(1).strip().split()
    while words and words[0].lower() in _LEADING_FILLERS:
        words.pop(0)
    if not words:
        return None
    return " ".join(words)


def _extract_compare_names(text: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 2:
        return quoted
    return []


def try_fast_path(latest_user_msg: str) -> ConversationState | None:
    """Returns a ConversationState if the heuristics are confident, else None
    (meaning: fall through to the full Analyzer LLM call)."""
    low = latest_user_msg.lower().strip()

    if any(k in low for k in INJECTION_KEYWORDS):
        return ConversationState(intent="recommend", is_injection=True)

    if any(k in low for k in OFF_TOPIC_KEYWORDS):
        return ConversationState(intent="recommend", is_off_topic=True)

    if len(latest_user_msg.strip()) < 40 and any(p in low for p in CLOSING_PHRASES):
        return ConversationState(intent="close")

    compare_names = _extract_compare_names(latest_user_msg)
    if compare_names:
        return ConversationState(intent="compare", comparison_targets=compare_names)

    if _is_jd_paste(latest_user_msg):
        # Role extraction from a full JD isn't reliable via regex — hand the
        # raw text through as "role" context; the explainer prompt still
        # gets the real message via conversation history regardless.
        return ConversationState(intent="recommend", role=latest_user_msg[:120], missing_fields=[])

    role = _extract_role(latest_user_msg)
    if role:
        return ConversationState(intent="recommend", role=role, missing_fields=[])

    # Genuinely vague first message ("I need an assessment", "help me hire someone")
    # — short, no role signal, no JD signal. Confident enough to slot-fill
    # without spending an LLM call.
    if len(latest_user_msg.strip()) < 80:
        return ConversationState(intent="recommend", missing_fields=["role"])

    # Longer message with no clear signal — ambiguous, let the real Analyzer
    # handle it rather than guessing.
    return None