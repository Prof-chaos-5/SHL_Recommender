from app.models import Recommendation
from app.retrieval import get_test_type

VALID_TEST_TYPES = {"K", "A", "P", "B", "C", "D", "S", "E"}

FORCED_SOURCES = {"business_rule", "always_include", "previous_turn"}


def ground_and_repair(
    selected_ids: list[str],
    candidates: list[dict],
    max_items: int = 10,
    excluded_names: set[str] | None = None,
    force_include: bool = True,
) -> list[Recommendation]:
    by_id = {c.get("entity_id"): c for c in candidates}
    ranked_pool = sorted(
        candidates,
        key=lambda c: c.get("_score", 0),
        reverse=True,
    )

    final: list[Recommendation] = []
    selected_candidate_ids: set[str] = set()

    # Dedupe primarily by entity_id, falling back to name if needed.
    seen: set[str] = set()

    def _key(item: dict) -> str:
        return str(item.get("entity_id")) if item.get("entity_id") is not None else item.get("name", "")

    def _try_add(item: dict) -> bool:
        key = _key(item)
        if key in seen:
            return False

        name = item.get("name")
        link = item.get("link")

        if not name or not link:
            return False

        final.append(
            Recommendation(
                name=name,
                url=link,
                test_type=get_test_type(item.get("keys", [])),
                entity_id=str(item.get("entity_id"))
                if item.get("entity_id") is not None
                else None,
            )
        )

        seen.add(key)

        if item.get("entity_id") is not None:
            selected_candidate_ids.add(str(item["entity_id"]))

        return True

    #
    # Add valid LLM selections
    #
    for sid in selected_ids:
        item = by_id.get(sid)
        if item:
            _try_add(item)
        if len(final) >= max_items:
            break

    #
    # Fallback if every selected id hallucinated / invalid
    #
    if not final:
        for item in ranked_pool:
            if _try_add(item) and len(final) >= max_items:
                break

    #
    # Guarantee forced candidates
    #
    if force_include:
        # FIX (C10): forced-inclusion used to be gated by scanning the LLM's
        # generated reply text for keywords like "excluded"/"skip"/"without".
        # That's unreliable in both directions — a correct reply like
        # "Verify G+ and Graduate Scenarios" (OPQ32r deliberately omitted)
        # contains none of those words, so the exclusion was never detected
        # and OPQ32r got force-reinserted right after the LLM correctly left
        # it out. Exclusion is now tracked as explicit conversation state
        # (ConversationState.excluded_items, extracted by the Analyzer) and
        # passed in directly, instead of being inferred from prose.
        excluded_lower = {n.strip().lower() for n in (excluded_names or set())}

        forced_items = [
            item
            for item in ranked_pool
            if item.get("_retrieval") in FORCED_SOURCES
            and (item.get("name") or "").lower() not in excluded_lower
        ]

        for forced in forced_items:
            if _key(forced) in seen:
                continue

            if len(final) < max_items:
                _try_add(forced)
                continue

            #
            # Replace the lowest-ranked non-forced recommendation.
            #
            replacement_idx = None

            for i in range(len(final) - 1, -1, -1):
                rec = final[i]
                candidate = by_id.get(rec.entity_id)

                if (
                    candidate
                    and candidate.get("_retrieval")
                    not in FORCED_SOURCES
                ):
                    replacement_idx = i
                    break

            if replacement_idx is None:
                continue

            removed = final.pop(replacement_idx)

            removed_key = (
                removed.entity_id
                if removed.entity_id is not None
                else removed.name
            )
            seen.discard(removed_key)

            _try_add(forced)

    return final


def schema_check(
    recommendations: list[Recommendation],
) -> list[Recommendation]:
    seen: set[str] = set()
    cleaned: list[Recommendation] = []

    for r in recommendations:
        key = r.entity_id or r.name

        if key in seen:
            continue

        if r.test_type not in VALID_TEST_TYPES:
            r.test_type = "K"

        cleaned.append(r)
        seen.add(key)

    return cleaned[:10]