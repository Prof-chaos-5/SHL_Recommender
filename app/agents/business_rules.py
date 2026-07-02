"""
Deterministic business rules.

These used to live as prose instructions buried inside the single God
Prompt (e.g. "Always include OPQ32r for senior roles", "use the Verify G+
bundle not individual components"). Encoding them here means they apply
every single turn, regardless of what the explainer LLM decides to write —
no risk of the model forgetting or inconsistently applying a rule.

The LLM still has final say on whether a force-included item actually
belongs in the shortlist it presents — this only guarantees the item is
*available* as a candidate, it doesn't force it into the final response.
"""

from app.retrieval import get_item_by_name

JUNIOR_MARKERS = {"entry", "entry-level", "junior", "frontline", "intern"}
SENIOR_MARKERS = {"senior", "manager", "director", "vp", "executive", "cxo", "c-suite", "head", "lead"}

LEADERSHIP_RECIPE = [
    "Occupational Personality Questionnaire OPQ32r",
    "OPQ Universal Competency Report 2.0",
    "OPQ Leadership Report",
]

CONTACT_CENTER_KEYWORDS = {
    "contact center", "contact centre", "call center", "call centre",
    "inbound", "customer service", "customer support"
}
HEALTHCARE_KEYWORDS = {"healthcare", "medical", "clinical", "nurse", "hospital"}
ADMIN_KEYWORDS = {"admin", "administrator", "administrative", "clerical", "coordinator",
                  "receptionist", "secretary", "patient record", "patient records"}

# FIX (C6): required_item_names() previously had no manufacturing/plant-
# safety branch at all -- only HEALTHCARE_KEYWORDS and
# CONTACT_CENTER_KEYWORDS triggered forced items. A chemical/plant-operator/
# safety conversation had no guaranteed force-include pairing, so
# "Manufac. & Indust. - Safety & Dependability 8.0" and "Workplace Health
# and Safety (New)" were only ever reachable via BM25 + KEYWORD_BOOST_MAP --
# one ranking away from being cut when the LLM caps the shortlist, with
# nothing in this module guaranteeing they survive. Additive fix, matching
# the existing HEALTHCARE_KEYWORDS pattern.
MANUFACTURING_KEYWORDS = {
    "manufacturing", "chemical", "plant operator", "plant operators",
    "industrial", "warehouse", "factory", "production line",
}


def _is_junior(state) -> bool:
    blob = f"{state.role or ''} {state.seniority or ''}".lower()
    return any(m in blob for m in JUNIOR_MARKERS)


def _is_senior(state) -> bool:
    blob = f"{state.role or ''} {state.seniority or ''}".lower()
    return any(m in blob for m in SENIOR_MARKERS)


def required_item_names(state) -> list[str]:
    """Catalog item names that must be force-included as candidates this turn."""
    names: set[str] = set()
    blob = f"{state.role or ''} {state.seniority or ''} {' '.join(state.skills)}".lower()

    # FIX (C10): items the user has explicitly asked to remove/replace must
    # never be re-forced back in on a later turn just because a rule
    # trigger (seniority, role keywords, etc.) still matches. Previously
    # only `personality_excluded` (a blanket "no personality testing at
    # all" flag) gated OPQ32r, so a user who removed OPQ32r specifically
    # ("drop the OPQ, replace with something shorter") but still wanted
    # *a* personality assessment had OPQ32r silently re-added every turn,
    # because personality_excluded was correctly False. excluded_items is
    # the specific, per-item signal that actually matches user intent here.
    excluded = {n.strip().lower() for n in (getattr(state, "excluded_items", None) or [])}

    def _add(name: str) -> None:
        if name.lower() not in excluded:
            names.add(name)

    # Personality — default-include for professional/senior roles unless the
    # user explicitly excluded it or the role reads as junior/frontline.
    if not state.personality_excluded and not _is_junior(state):
        _add("Occupational Personality Questionnaire OPQ32r")

    if _is_senior(state):
        for item in LEADERSHIP_RECIPE:
            _add(item)

    if any(k in blob for k in CONTACT_CENTER_KEYWORDS):
        _add("SVAR - Spoken English (US) (New)")
        _add("Entry Level Customer Serv-Retail & Contact Center")
        _add("Contact Center Call Simulation (New)")
        _add("Customer Service Phone Simulation")

    if any(k in blob for k in HEALTHCARE_KEYWORDS):
        _add("Dependability and Safety Instrument (DSI)")
        # Healthcare admin roles additionally need terminology and doc skills
        if any(k in blob for k in ADMIN_KEYWORDS):
            _add("Medical Terminology (New)")
            _add("Microsoft Word 365 - Essentials (New)")

    if any(k in blob for k in MANUFACTURING_KEYWORDS):
        _add("Manufac. & Indust. - Safety & Dependability 8.0")
        _add("Workplace Health and Safety (New)")

    return list(names)


def apply(state, candidates: list[dict]) -> list[dict]:
    """Merges required items into the retrieved candidate pool, boosted to
    the front. The LLM still decides whether each one belongs in the final
    shortlist for this specific query."""
    seen = {c.get("entity_id") for c in candidates}
    forced = []
    for name in required_item_names(state):
        item = get_item_by_name(name)
        if item and item.get("entity_id") not in seen:
            item = item.copy()
            item["_score"] = 998.0
            item["_retrieval"] = "business_rule"
            forced.append(item)
            seen.add(item.get("entity_id"))
    return forced + candidates