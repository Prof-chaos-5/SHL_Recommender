"""
Smart Evaluator for SHL Assessment Recommender.

Instead of replaying trace messages verbatim, this evaluator uses
Groq Llama to simulate a realistic user who:
  - Knows ALL facts from the full trace conversation
  - Responds naturally to whatever the agent actually says
  - Answers agent questions truthfully from the persona
  - Says "no preference" when asked something outside the persona facts
  - Ends the conversation when satisfied with recommendations

Usage:
    python smart_eval.py
"""

import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root regardless of where script is run from
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx
from groq import Groq
from metrics import mean_recall_at_k, recall_at_k


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRACES_PATH = BASE_DIR / "data" / "traces"
RESULTS_PATH = BASE_DIR / "eval" / "smart_results.json"
API_URL = "http://localhost:8000/chat"

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 8
REQUEST_DELAY = 2       # seconds between agent calls
GROQ_DELAY = 0.5        # seconds between Groq calls

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================================
# TABLE EXTRACTION
# ============================================================================

def extract_last_table_names(text: str) -> list[str]:
    """Extract assessment names from the last markdown table in text."""
    table_blocks = []
    current = []

    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            if current:
                table_blocks.append("\n".join(current))
                current = []
    if current:
        table_blocks.append("\n".join(current))

    if not table_blocks:
        return []

    names = []
    for line in table_blocks[-1].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) < 2 or cols[1].lower() == "name":
            continue
        if not cols[0].isdigit():
            continue
        if cols[1] not in names:
            names.append(cols[1])

    return names


# ============================================================================
# PERSONA EXTRACTOR
# Builds a structured persona from ALL turns in the trace —
# not just the first message. This gives the simulated user
# full context so it can answer agent questions naturally.
# ============================================================================

def extract_persona_from_trace(content: str) -> str:
    """
    Build a structured persona from the full trace conversation.

    Reads ALL user + agent turns to extract:
    - What role is being hired for
    - What facts the user revealed across all turns
    - What the agent clarified and user confirmed

    This gives the simulated user full context so it can answer
    agent questions naturally, not just replay scripted messages.
    """

    # 1. Try explicit ## Persona or ## Facts section first
    persona_match = re.search(
        r"## (?:Persona|Facts|Context|Background)\s*(.+?)(?=\n## |\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if persona_match:
        return persona_match.group(1).strip()

    # 2. Extract ALL turns (user + agent) to build context
    turns = []
    turn_blocks = re.split(r"(?=### Turn \d+)", content)

    for block in turn_blocks:
        if "### Turn" not in block:
            continue

        user_match = re.search(
            r"\*\*User\*\*\s*>\s*(.+?)(?=\*\*Agent\*\*|\Z)",
            block,
            re.DOTALL,
        )
        agent_match = re.search(
            r"\*\*Agent\*\*\s*\n(.+?)(?=### Turn|\Z)",
            block,
            re.DOTALL,
        )

        if user_match:
            user_msg = user_match.group(1).strip()
            turns.append(("user", user_msg))

        if agent_match:
            agent_msg = agent_match.group(1)
            # Strip markdown tables
            agent_msg = re.sub(r"\|.+\|", "", agent_msg)
            # Strip metadata lines like _No recommendations_
            agent_msg = re.sub(r"_.*?_", "", agent_msg)
            agent_msg = agent_msg.strip()
            if agent_msg:
                turns.append(("agent", agent_msg))

    if not turns:
        return "No persona available. Answer based on conversation context."

    # 3. Build structured persona
    # User messages = facts the hiring manager revealed
    # Agent Q + User A pairs = key clarifications already made

    user_facts = []
    qa_pairs = []

    for i, (role, msg) in enumerate(turns):
        if role == "user":
            user_facts.append(msg)
        elif role == "agent" and i + 1 < len(turns):
            next_role, next_msg = turns[i + 1]
            if next_role == "user" and "?" in msg:
                question_lines = [
                    line.strip()
                    for line in msg.splitlines()
                    if "?" in line and line.strip()
                ]
                if question_lines:
                    qa_pairs.append({
                        "question": question_lines[-1],
                        "answer": next_msg,
                    })

    # Build persona text
    persona_parts = [
        "You are a hiring manager or recruiter with the following known facts:",
        "",
        "## Facts about the role you are hiring for:",
    ]

    for fact in user_facts:
        clean = " ".join(fact.split())
        persona_parts.append(f"- {clean}")

    if qa_pairs:
        persona_parts.append("")
        persona_parts.append("## Additional details you have already confirmed:")
        for qa in qa_pairs:
            q = " ".join(qa["question"].split())
            a = " ".join(qa["answer"].split())
            persona_parts.append(f"- When asked '{q}', you said: '{a}'")

    persona_parts.extend([
        "",
        "## Behaviour rules:",
        "- Answer questions using ONLY the facts above",
        "- If asked something not in your facts, say 'I have no preference on that'",
        "- Do NOT invent new requirements or constraints",
        "- Keep responses short and natural (1-2 sentences max)",
        "- Do NOT mention you are an AI or simulating anything",
    ])

    return "\n".join(persona_parts)


# ============================================================================
# TRACE LOADER
# ============================================================================

def load_trace(path: Path) -> dict:
    """Load a markdown or JSON trace into structured format."""
    trace_id = path.stem

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    with open(path, encoding="utf-8") as f:
        content = f.read()

    expected = extract_last_table_names(content)
    persona = extract_persona_from_trace(content)

    # Get first user message to kick off the conversation
    first_user = re.search(
        r"### Turn 1.*?\*\*User\*\*\s*>\s*(.+?)(?=\*\*Agent\*\*|\n###)",
        content,
        re.DOTALL,
    )
    opening_message = first_user.group(1).strip() if first_user else ""

    return {
        "id": trace_id,
        "persona": persona,
        "opening_message": opening_message,
        "expected_assessments": expected,
    }


# ============================================================================
# SIMULATED USER (Groq)
# ============================================================================

USER_SYSTEM_PROMPT = """You are simulating a hiring manager in a conversation with an AI assessment recommender.

Your persona and known facts:
{persona}

Current conversation so far:
{history}

The agent just said:
{agent_message}

Your response (1-2 sentences, natural and conversational):"""


def simulate_user_response(
    persona: str,
    history: list[dict],
    agent_message: str,
) -> str:
    """
    Use Groq to generate a realistic user response
    given the full persona and current agent message.
    """
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in history[-6:]
    ) if history else "(conversation just started)"

    prompt = USER_SYSTEM_PROMPT.format(
        persona=persona,
        history=history_text,
        agent_message=agent_message,
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[groq] Error: {e}")
        return "Please go ahead with the recommendations."


def is_conversation_ending(user_response: str) -> bool:
    """Detect if simulated user is wrapping up."""
    endings = [
        "thank", "perfect", "great", "that works", "exactly",
        "looks good", "sounds good", "that's what", "appreciate",
        "no more", "that's all", "we're done", "all set",
    ]
    lower = user_response.lower()
    return any(e in lower for e in endings)


# ============================================================================
# AGENT CALLER
# ============================================================================

def call_agent(messages: list[dict], retries: int = 3) -> dict | None:
    """Call the FastAPI agent with retry + backoff."""
    for attempt in range(retries):
        try:
            resp = httpx.post(
                API_URL,
                json={"messages": messages},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"[agent] 429 rate limit — sleeping {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()

        except httpx.ConnectError:
            print("[agent] Cannot connect — is the server running?")
            time.sleep(5)
        except Exception as e:
            print(f"[agent] Request failed: {e}")
            time.sleep(3)

    return None


def extract_recommendations(data: dict) -> list[str]:
    """Extract recommendation names from agent response."""
    if data.get("recommendations"):
        return [r["name"] for r in data["recommendations"] if "name" in r]

    # Fallback: parse markdown table from reply text
    if data.get("reply"):
        names = []
        for line in data["reply"].splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if re.match(r"^\|[-| ]+\|$", line):
                continue
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) >= 2 and cols[1].lower() != "name" and cols[0].isdigit():
                names.append(cols[1])
        return names

    return []


# ============================================================================
# TRACE RUNNER
# ============================================================================

def run_trace(trace: dict) -> dict:
    """
    Run a full simulated conversation for one trace.

    Flow:
    1. Send opening message to agent
    2. Agent replies
    3. Groq simulates user response based on full persona + agent reply
    4. Repeat until end_of_conversation or turn cap
    """
    trace_id = trace.get("id", "unknown")
    persona = trace.get("persona", "")
    opening = trace.get("opening_message", "")
    expected = trace.get("expected_assessments", [])

    print(f"\n{'='*70}")
    print(f"Trace:    {trace_id}")
    print(f"Opening:  {opening}")
    print(f"Expected: {expected}")
    print(f"{'='*70}")

    if not opening:
        print(f"[eval] No opening message for {trace_id}, skipping")
        return {
            "trace_id": trace_id,
            "expected": expected,
            "recommended": [],
            "conversation": [],
            "turns_used": 0,
        }

    messages = []
    final_recommendations = []
    conversation_log = []
    turn = 0
    current_user_message = opening

    while turn < MAX_TURNS:
        turn += 1

        # Add user message
        messages.append({"role": "user", "content": current_user_message})
        conversation_log.append({"role": "user", "content": current_user_message})
        print(f"\n[T{turn}] USER:  {current_user_message}")

        # Call agent
        time.sleep(REQUEST_DELAY)
        agent_data = call_agent(messages)

        if agent_data is None:
            print(f"[eval] Agent call failed at turn {turn}")
            break

        agent_reply = agent_data.get("reply", "")
        end_flag = agent_data.get("end_of_conversation", False)

        # Track latest non-empty recommendations
        recs = extract_recommendations(agent_data)
        if recs:
            final_recommendations = recs

        messages.append({"role": "assistant", "content": agent_reply})
        conversation_log.append({"role": "assistant", "content": agent_reply})

        print(f"[T{turn}] AGENT: {agent_reply[:200]}{'...' if len(agent_reply) > 200 else ''}")
        if recs:
            print(f"[T{turn}] RECS:  {recs}")

        if end_flag:
            print(f"[eval] end_of_conversation=true at turn {turn}")
            break

        # Leave room for closing turn
        if turn >= MAX_TURNS - 1:
            print(f"[eval] Turn cap reached")
            break

        # Simulate next user response
        time.sleep(GROQ_DELAY)
        current_user_message = simulate_user_response(
            persona=persona,
            history=messages,
            agent_message=agent_reply,
        )
        print(f"[T{turn}] SIMULATED: {current_user_message}")

        # If user is closing and we have recommendations, wrap up
        if is_conversation_ending(current_user_message) and final_recommendations:
            messages.append({"role": "user", "content": current_user_message})
            conversation_log.append({"role": "user", "content": current_user_message})
            turn += 1

            time.sleep(REQUEST_DELAY)
            final_data = call_agent(messages)
            if final_data:
                final_recs = extract_recommendations(final_data)
                if final_recs:
                    final_recommendations = final_recs
                conversation_log.append({
                    "role": "assistant",
                    "content": final_data.get("reply", ""),
                })
            break

    return {
        "trace_id": trace_id,
        "expected": expected,
        "recommended": final_recommendations,
        "conversation": conversation_log,
        "turns_used": turn,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"[eval] Smart evaluator — {GROQ_MODEL}")
    print(f"[eval] Traces: {TRACES_PATH}")
    print(f"[eval] Agent:  {API_URL}\n")

    if not TRACES_PATH.exists():
        print("[error] Traces directory not found")
        return

    trace_files = sorted(
        f for f in TRACES_PATH.iterdir()
        if f.suffix in (".json", ".md")
    )
    print(f"[eval] Found {len(trace_files)} traces\n")

    results = []

    for path in trace_files:
        try:
            trace = load_trace(path)

            # Sanity check before running
            if not trace.get("opening_message"):
                print(f"[warn] {path.name}: no opening message, skipping")
                continue
            if not trace.get("expected_assessments"):
                print(f"[warn] {path.name}: no expected assessments, skipping")
                continue

            result = run_trace(trace)
            result["recall@10"] = recall_at_k(
                result["expected"],
                result["recommended"],
            )
            results.append(result)

            # Per-trace summary
            expected_set = set(result["expected"])
            recommended_set = set(result["recommended"])
            matches = expected_set & recommended_set
            missing = expected_set - recommended_set
            extra = recommended_set - expected_set

            print(f"\nRESULT: {result['trace_id']}")
            print(f"  Turns:    {result.get('turns_used', '?')}/{MAX_TURNS}")
            print(f"  Recall@10: {result['recall@10']:.2f}")
            print(f"  Matches:  {sorted(matches) or '(none)'}")
            print(f"  Missing:  {sorted(missing) or '(none)'}")
            print(f"  Extra:    {sorted(extra) or '(none)'}")

        except Exception as e:
            import traceback
            print(f"[error] Failed {path.name}: {e}")
            traceback.print_exc()

    if not results:
        print("[eval] No results to report")
        return

    mean = mean_recall_at_k(results)
    print(f"\n{'#'*70}")
    print(f"Mean Recall@10: {mean:.4f} across {len(results)} traces")
    print(f"{'#'*70}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": GROQ_MODEL,
                "mean_recall@10": mean,
                "traces": results,
            },
            f,
            indent=2,
        )
    print(f"\n[eval] Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()