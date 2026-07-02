"""
Smart Evaluator for SHL Assessment Recommender.

Instead of replaying trace messages verbatim, this evaluator uses
an LLM to simulate a realistic user who:
  - Knows ALL facts from the full trace conversation
  - Responds naturally to whatever the agent actually says
  - Answers agent questions truthfully from the persona
  - Says "no preference" when asked something outside the persona facts
  - Ends the conversation when satisfied with recommendations

Usage:
    python smart_eval.py
    python smart_eval.py --all
    python smart_eval.py --traces trace_1.md trace_2.json
"""

import argparse
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

# Toggle to run all traces in the TRACES_PATH, or only a specific array of traces
RUN_ALL_TRACES = True
SPECIFIC_TRACES = ["C1.md", "C10.md"
    # List specific trace names here (you can include or omit .md/.json extension)
    # "trace_01",
    # "trace_02.md",
]

BASE_DIR = Path(__file__).resolve().parent.parent
TRACES_PATH = BASE_DIR / "data" / "traces"
RESULTS_PATH = BASE_DIR / "eval" / "smart_results.json"
API_URL = "http://localhost:8000/chat"

GROQ_MODEL = "qwen/qwen3-32b"
MAX_TURNS = 8
REQUEST_DELAY = 2       # seconds between agent calls
GROQ_DELAY = 4          # seconds between Groq calls

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]
if not GROQ_KEYS:
    raise ValueError("No Groq API keys found in environment variables")

GROQ_CLIENTS = [Groq(api_key=key) for key in GROQ_KEYS]
_client_index = 0

def _next_client() -> Groq:
    global _client_index
    client = GROQ_CLIENTS[_client_index]
    _client_index = (_client_index + 1) % len(GROQ_CLIENTS)
    return client


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
# ============================================================================

def extract_persona_from_trace(content: str) -> str:
    """
    Build a structured persona from the full trace conversation.
    """
    persona_match = re.search(
        r"## (?:Persona|Facts|Context|Background)\s*(.+?)(?=\n## |\Z)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if persona_match:
        return persona_match.group(1).strip()

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
            agent_msg = re.sub(r"\|.+\|", "", agent_msg)
            agent_msg = re.sub(r"_.*?_", "", agent_msg)
            agent_msg = agent_msg.strip()
            if agent_msg:
                turns.append(("agent", agent_msg))

    if not turns:
        return "No persona available. Answer based on conversation context."

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

    user_turns = []
    for m in re.finditer(
        r"\*\*User\*\*\s*>\s*(.+?)(?=\*\*Agent\*\*|\n###|\Z)",
        content,
        re.DOTALL,
    ):
        user_turns.append(m.group(1).strip())

    opening_message = user_turns[0] if user_turns else ""

    return {
        "id": trace_id,
        "persona": persona,
        "opening_message": opening_message,
        "user_turns": user_turns,
        "expected_assessments": expected,
    }


# ============================================================================
# SIMULATED USER (Groq)
# ============================================================================

USER_SYSTEM_PROMPT = """You are simulating a hiring manager talking to an AI assessment recommender.

Your persona and known facts:
{persona}

Conversation so far:
{history}

The agent just said:
{agent_message}

Rules for your reply:
- Write 1-2 SHORT sentences max. Think terse executive, not chatty.
- ONLY use facts from your persona above. NEVER invent new requirements, constraints, report formats, or preferences.
- NEVER ask the agent a question — you are the one being asked.
- If the agent asks you something not covered by your facts, say exactly: "I have no preference on that."
- If the agent's recommendations look reasonable and you have no more facts to add, confirm briefly (e.g. "Sounds good.", "That works.", "Confirmed.").
- Match the tone of these examples:
  "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant."
  "English."
  "We're industrial. The 8.0 bundle is the right fit. Confirmed."
  "Good. Can you also add a situational judgement element?"

Your response:"""


def _clean_for_prompt(text: str) -> str:
    lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("|") and trimmed.endswith("|"):
            continue
        if trimmed.startswith("_") and trimmed.endswith("_"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def simulate_user_response(
    persona: str,
    history: list[dict],
    agent_message: str,
) -> str:
    clean_history = []
    for m in history[-6:]:
        clean_history.append({
            "role": m["role"],
            "content": _clean_for_prompt(m["content"])
        })

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in clean_history
    ) if clean_history else "(conversation just started)"

    clean_agent_message = _clean_for_prompt(agent_message)

    prompt = USER_SYSTEM_PROMPT.format(
        persona=persona,
        history=history_text,
        agent_message=clean_agent_message,
    )

    MAX_ATTEMPTS = len(GROQ_CLIENTS) * 2
    for attempt in range(MAX_ATTEMPTS):
        client = _next_client()
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            content = (response.choices[0].message.content or "").strip()
            content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip()
            
            if not content:
                return "Please go ahead with the recommendations."
            return content

        except Exception as e:
            error_text = str(e).lower()
            if any(x in error_text for x in ["rate limit", "429", "too many requests"]):
                wait_time = min(2 ** attempt, 3)
                match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_text)
                if match:
                    mins = int(match.group(1)) if match.group(1) else 0
                    wait_time = int(mins * 60 + float(match.group(2))) + 1
                if wait_time > 3:
                    print(f"[groq] Key rate limited, rotating to next key...")
                    continue
                time.sleep(wait_time)
                continue

            if any(x in error_text for x in ["connection", "timeout", "503"]):
                time.sleep(1)
                continue

            print(f"[groq] Error: {e}")
            break

    return "Please go ahead with the recommendations."


def is_conversation_ending(user_response: str) -> bool:
    lower = user_response.lower()

    closing_keywords = [
        "thank", "perfect", "that works", "exactly",
        "looks good", "sounds good", "that's what", "appreciate",
        "no more", "that's all", "we're done", "all set",
        "let's proceed", "let's go with", "confirmed",
        "that's great", "we'll use",
    ]
    if not any(kw in lower for kw in closing_keywords):
        return False

    continuation_signals = [
        "?", "can you", "could you", "also add", "also include",
        "replace", "swap", "remove", "instead", "what about",
        "how about", "one more", "but ", "however", "actually",
    ]
    if any(sig in lower for sig in continuation_signals):
        return False

    return True


# ============================================================================
# AGENT CALLER
# ============================================================================

def call_agent(messages: list[dict], retries: int = 3) -> dict | None:
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
    if data.get("recommendations"):
        return [r["name"] for r in data["recommendations"] if "name" in r]

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
    trace_id = trace.get("id", "unknown")
    persona = trace.get("persona", "")
    opening = trace.get("opening_message", "")
    expected = trace.get("expected_assessments", [])
    canonical_user_turns = trace.get("user_turns", [])

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

        messages.append({"role": "user", "content": current_user_message})
        conversation_log.append({"role": "user", "content": current_user_message})
        print(f"\n[T{turn}] USER:  {current_user_message}")

        time.sleep(REQUEST_DELAY)
        agent_data = call_agent(messages)

        if agent_data is None:
            print(f"[eval] Agent call failed at turn {turn}")
            break

        agent_reply = agent_data.get("reply", "")
        end_flag = agent_data.get("end_of_conversation", False)

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

        if turn >= MAX_TURNS - 1:
            print(f"[eval] Turn cap reached")
            break

        time.sleep(GROQ_DELAY)
        simulated_response = simulate_user_response(
            persona=persona,
            history=messages,
            agent_message=agent_reply,
        )

        canonical_turn = ""
        if turn < len(canonical_user_turns):
            canonical_turn = canonical_user_turns[turn].strip()

        if canonical_turn:
            current_user_message = canonical_turn
        else:
            current_user_message = simulated_response
        print(f"[T{turn}] SIMULATED: {current_user_message}")

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
    # Setup Argument Parser to allow overriding the config via CLI
    parser = argparse.ArgumentParser(description="Smart Evaluator")
    parser.add_argument("--all", action="store_true", help="Run all traces in the traces folder")
    parser.add_argument("--traces", nargs="+", help="Run specific traces (e.g. --traces trace1.md trace2.json)")
    args = parser.parse_args()

    # Determine execution mode from args or config
    run_all = RUN_ALL_TRACES
    specific_traces = SPECIFIC_TRACES

    if args.all:
        run_all = True
    if args.traces:
        run_all = False
        specific_traces = args.traces

    print(f"[eval] Smart evaluator — {GROQ_MODEL}")
    print(f"[eval] Traces: {TRACES_PATH}")
    print(f"[eval] Agent:  {API_URL}\n")

    if not TRACES_PATH.exists():
        print("[error] Traces directory not found")
        return

    # Select the files to process
    if run_all:
        print("[eval] Mode: RUN ALL TRACES")
        trace_files = sorted(
            f for f in TRACES_PATH.iterdir()
            if f.suffix in (".json", ".md")
        )
    else:
        print(f"[eval] Mode: SPECIFIC TRACES ({len(specific_traces)} requested)")
        trace_files = []
        for name in specific_traces:
            path = TRACES_PATH / name
            # Handle cases where the user included or didn't include file extensions
            if path.is_file():
                trace_files.append(path)
            elif (TRACES_PATH / f"{name}.md").is_file():
                trace_files.append(TRACES_PATH / f"{name}.md")
            elif (TRACES_PATH / f"{name}.json").is_file():
                trace_files.append(TRACES_PATH / f"{name}.json")
            else:
                print(f"[warn] Trace not found: {name}")

        # Deduplicate files while preserving order
        trace_files = list(dict.fromkeys(trace_files))

    print(f"[eval] Found {len(trace_files)} traces to run\n")

    results = []

    for path in trace_files:
        try:
            trace = load_trace(path)

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

    mean = sum(r["recall@10"] for r in results) / len(results)
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