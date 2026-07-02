# Agent Architecture

```text
                           POST /chat
                                │
                                ▼
                    Conversation Analyzer (LLM)
                                │
                                ▼
                       Conversation State
                                │
                                ▼
                      Agent Controller (Python)
                                │
     ┌──────────────┬───────────────┬───────────────┐
     │              │               │               │
     ▼              ▼               ▼               ▼
 Slot Filling   Recommendation   Comparison      Refusal
     │              │               │
     │              ▼               ▼
     │          Retriever       Retriever
     │              │               │
     │              ▼               ▼
     │            Ranker         Ranker
     │              │               │
     │              ▼               ▼
     │         Prompt Builder  Prompt Builder
     │              │               │
     │              ▼               ▼
     │             LLM             LLM
     │              │               │
     └──────────────┴───────┬───────┘
                            ▼
                    Response Formatter
                            ▼
                    Schema Validator
                            ▼
                  Grounding Validator
                            ▼
                     JSON Response
```

---

# Philosophy

The LLM should **not** decide what to do.

The LLM should only generate natural language and extract information.

Everything else is deterministic.

---

# Components

## 1. Conversation Analyzer

Implemented using an LLM.

### Responsibility

Convert the raw message history into structured state. LLMs are excellent at this information extraction task.

Input

```json
{
    "messages":[
        ...
    ]
}
```

Output

```json
{
    "intent": "recommend",
    "role": "Backend Engineer",
    "seniority": "Mid",
    "skills": ["Python", "SQL"],
    "constraints": {
        "duration_less_than_mins": 30,
        "online_only": true,
        "adaptive": true
    },
    "required_traits": [],
    "technical_required": true,
    "personality_required": true,
    "comparison_targets": [],
    "clarification_history": [],
    "conversation_complete": false,
    "missing_fields": []
}
```

The analyzer reconstructs the conversation because the API is stateless.

It never recommends assessments.

It only extracts information to build the structured state. Notice that it extracts constraints directly instead of burying them inside history, and clearly specifies the `intent` (e.g., `recommend`, `compare`, `refuse`, etc.).

---

## 2. Agent Controller

Implemented entirely in Python.

No LLM. 100% deterministic.

### Responsibility

Decide which workflow to execute based on the structured state provided by the Conversation Analyzer. (Renamed from "Planner" to avoid confusion with orchestration frameworks like LangGraph, AutoGen, CrewAI).

Decision tree

```text
Off-topic?
↓
REFUSE
--------------------
Comparison request? (intent == "compare")
↓
COMPARE
--------------------
Enough information? (missing_fields empty)
↓
RECOMMEND
--------------------
Otherwise
↓
SLOT FILLING
```

The controller never generates language. It only routes requests deterministically.

---

## 3. Slot Filling Agent

Goal

Ask exactly one high-value question to fill missing slots (e.g., Role, Level, Industry).

Example

User
```text
Need an assessment.
```

Missing
```text
Role
```

Response
```text
What role are you hiring for?
```

NOT
```text
Experience?
Industry?
Location?
Company size?
```

Stay below the assignment's turn limit. This acts as a classic NLP slot filling mechanism rather than a generic "clarification" agent.

---

## 4. Centralized Components

### Prompt Builder

Prompts are centralized in a single `PromptBuilder` component. 

```text
PromptBuilder
↓
recommend()
↓
compare()
↓
clarify()
```

Individual agents do not build their own prompts. This makes the system highly maintainable.

### Response Formatter

Converts the raw LLM output into the final response schema.
This prevents prompt tweaks from breaking the API integration.

```text
LLM
↓
Formatter
↓
Validator
↓
JSON
```

---

## 5. Recommendation Agent

Pipeline

```text
Recommendation Agent
↓
Retriever
↓
Ranker
↓
Prompt Builder
↓
LLM
↓
Structured Response
```

Responsibilities
- Explains recommendations.
- Recommends only retrieved assessments.
- The retrieval logic (Retriever, Ranker) is entirely decoupled from the LLM.

---

## 6. Comparison Agent

Pipeline

```text
Assessment A & Assessment B
↓
Retriever
↓
Ranker
↓
Prompt Builder
↓
LLM
↓
Structured Response
```

Never compare from model memory. Always use retrieved catalog context.

---

## 7. Refusal Agent

Handles
- Prompt injection
- Hiring advice
- Legal questions
- Medical questions
- Politics
- Out-of-domain requests

Example

```text
Ignore previous instructions.
```

↓

Refuse.

---

## 8. Dual Validators

Runs before every response. Split into two clear responsibilities:

### Schema Validator
- Ensures valid JSON structure according to API requirements.
- Max recommendations <= 10.
- No duplicate recommendations.

### Grounding Validator
- Ensures recommended assessment physically exists in the catalog.
- Ensures the SHL URL exists.
- Never allows hallucinated assessments through.

If validation fails → repair → return.

---

# State Object

Everything downstream receives one object.

```python
ConversationState

intent
role
seniority
skills
constraints
required_traits
technical_required
personality_required
comparison_targets
clarification_history
conversation_complete
missing_fields
history
```

No component should inspect raw messages directly. Everything operates on the structured state. This allows for trivial refinement (e.g., User says "Actually make it senior", only `seniority` changes, nothing else).

---

# Why this architecture?

This provides a very clean split of responsibilities resembling a professional software-engineered backend rather than a simple RAG wrapper:
- **LLM → Understand**: Analyzer understands the conversation and extracts state/constraints.
- **Python → Decide**: Agent Controller makes deterministic decisions.
- **Retrieval Pipeline**: Retriever and Ranker find relevant evidence independently.
- **LLM → Explain**: Explains the evidence naturally based on centralized Prompts.
- **Formatter & Validators → Guarantee Correctness**: Ensures the output is valid and grounded.

This architecture explicitly prepares for robust **evaluation** (Recall@10, clarification success, refusal accuracy, and latency metrics) by isolating each step of the pipeline.