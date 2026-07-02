"""
Agent Controller.

100% deterministic. No LLM. Decides which workflow handles this turn based
solely on the ConversationState produced by the Analyzer. Never generates
language — that happens in the prompts/templates the chosen workflow uses.

Decision tree:

    injection or off-topic?                    -> REFUSE
    intent == "compare"?                       -> COMPARE
    intent == "close"?                         -> CLOSE
    missing role AND haven't asked yet?        -> SLOT_FILL
    otherwise                                  -> RECOMMEND
"""

from app.models import ConversationState

# Evaluator caps conversations at 8 turns total. Kept as a hard backstop,
# but the real limit on clarification is prior_assistant_turns below --
# turn_count alone previously let the agent re-ask the same question every
# turn as long as missing_fields kept coming back non-empty (e.g. the user
# answers with a seniority level like "CXO and director-level" rather than
# a literal job title, so the Analyzer never resolves "role"). That produced
# an infinite clarification loop instead of the "ask exactly ONE question"
# behavior the architecture promises.
SOFT_TURN_LIMIT = 6


def route(state: ConversationState, turn_count: int, prior_assistant_turns: int) -> str:
    if state.is_injection:
        return "REFUSE_INJECTION"

    if state.is_off_topic:
        return "REFUSE_OFF_TOPIC"

    if state.intent == "compare":
        return "COMPARE"

    if state.intent == "close":
        return "CLOSE"

    # Ask at most ONE clarifying question, ever. If we've already produced
    # at least one prior assistant turn and the user still hasn't resolved
    # to a clean role, proceed with whatever's been given rather than
    # asking again -- this is what actually prevents the repeat loop, not
    # the turn cap (which only kicks in after several wasted rounds).
    already_asked = prior_assistant_turns > 0
    if state.missing_fields and not already_asked and turn_count < SOFT_TURN_LIMIT:
        return "SLOT_FILL"

    return "RECOMMEND"