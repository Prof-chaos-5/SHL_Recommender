from pydantic import BaseModel, field_validator
from typing import Optional

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # K, A, P, B, C, D, S, E

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]   # always a list, empty when clarifying
    end_of_conversation: bool
