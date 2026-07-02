from attr import field
from pydantic import BaseModel, field_validator
from typing import Optional, Literal


# ============================================================================
# API SCHEMA — unchanged from the original spec, this is non-negotiable
# ============================================================================

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # K, A, P, B, C, D, S, E
    entity_id: Optional[str] = None

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]  # always a list, empty when clarifying/refusing
    end_of_conversation: bool


# ============================================================================
# CONVERSATION STATE — the object every downstream component reads from.
# No component below the Analyzer inspects raw messages directly.
# ============================================================================

class Constraints(BaseModel):
    duration_less_than_mins: Optional[int] = None
    online_only: Optional[bool] = None
    adaptive: Optional[bool] = None


class ConversationState(BaseModel):
    intent: Literal["recommend", "compare", "close"] = "recommend"

    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: list[str] = []
    constraints: Constraints = Constraints()
    required_traits: list[str] = []

    technical_required: bool = False
    personality_required: bool = False
    personality_excluded: bool = False  # user explicitly said "no personality tests"
    excluded_items: list[str] = []
    comparison_targets: list[str] = []
    # Any assessment named anywhere in the conversation (by the user, or by
    # the assistant inside an earlier COMPARE reply) that the user has shown
    # interest in adding or keeping. Populated by the Analyzer and consumed
    # by agent.py's _pin_explicit_names() so names that only ever surfaced
    # in a compare turn -- and are too weak a signal for BM25 to catch on a
    # short later confirmation like "add MQ" -- still make it into the
    # candidate pool on subsequent RECOMMEND turns.
    explicitly_named_items: list[str] = []

    is_off_topic: bool = False
    is_injection: bool = False

    missing_fields: list[str] = []