"""
Run evaluation against conversation traces.

Supports:
- JSON traces
- Markdown transcript traces (SHL format)

Expected assessments are extracted from the LAST markdown table
in the trace (the final confirmed shortlist), since the traces
don't have a separate ## Expected Assessments section.
"""

import json
import re
import time
from pathlib import Path

import httpx

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import mean_recall_at_k, recall_at_k


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRACES_PATH = BASE_DIR / "data" / "traces"
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULTS_PATH = (
    BASE_DIR
    / "eval"
    / f"results_{TIMESTAMP}.json"
)

API_URL = "http://localhost:8001/chat"

REQUEST_DELAY_SECONDS = 2
MAX_RETRIES = 5
BACKOFF_BASE = 2


# ============================================================================
# TABLE EXTRACTION (shared by trace parser + reply parser)
# ============================================================================

def extract_names_from_table(text: str) -> list[str]:
    """
    Extract assessment names from markdown table Name column.

    Table format:
    | # | Name | Test Type | Keys | Duration | Languages | URL |
    |---|------|-----------|------|----------|-----------|-----|
    | 1 | Occupational Personality Questionnaire OPQ32r | P | ...

    Returns names in order, deduplicated.
    """
    names = []

    for line in text.splitlines():
        line = line.strip()

        if not line.startswith("|"):
            continue

        # Skip separator rows like |---|---|
        if re.match(r"^\|[-| ]+\|$", line):
            continue

        cols = [c.strip() for c in line.split("|") if c.strip()]

        # Need at least 2 columns: # and Name
        if len(cols) < 2:
            continue

        # Skip header row
        if cols[1].lower() == "name":
            continue

        # Skip if first col isn't a number (extra safety)
        if not cols[0].isdigit():
            continue

        name = cols[1]
        if name and name not in names:
            names.append(name)

    return names


def extract_last_table_names(text: str) -> list[str]:
    """
    Find all markdown tables in text and return names from the LAST one.
    The last table = the final confirmed shortlist in the trace.
    """
    # Split on table blocks: a table is contiguous lines starting with |
    table_blocks = []
    current_block = []

    for line in text.splitlines():
        if line.strip().startswith("|"):
            current_block.append(line)
        else:
            if current_block:
                table_blocks.append("\n".join(current_block))
                current_block = []

    if current_block:
        table_blocks.append("\n".join(current_block))

    if not table_blocks:
        return []

    # Use the last table — that's the final confirmed shortlist
    return extract_names_from_table(table_blocks[-1])


# ============================================================================
# MARKDOWN TRACE PARSER
# ============================================================================

def parse_markdown_trace(content: str, trace_id: str) -> dict:
    """
    Parse SHL markdown transcript traces into structured format.

    Turn structure:
        ### Turn N
        **User**
        > user message
        **Agent**
        agent reply (may include markdown table)

    Expected assessments = names from the LAST table in the entire trace.
    This is the final confirmed shortlist the agent committed to.
    """
    turns = []

    # Match each turn block
    turn_blocks = re.split(r"(?=### Turn \d+)", content)

    for block in turn_blocks:
        if not block.strip() or "### Turn" not in block:
            continue

        # Extract user message
        user_match = re.search(
            r"\*\*User\*\*\s*>\s*(.+?)(?=\*\*Agent\*\*|\Z)",
            block,
            re.DOTALL,
        )

        # Extract agent reply
        agent_match = re.search(
            r"\*\*Agent\*\*\s*\n(.+?)(?=### Turn|\Z)",
            block,
            re.DOTALL,
        )

        if user_match:
            turns.append({
                "role": "user",
                "content": user_match.group(1).strip(),
            })

        if agent_match:
            turns.append({
                "role": "assistant",
                "content": agent_match.group(1).strip(),
            })

    # Expected = names from the last table in the full trace
    # This is the ground truth shortlist we evaluate against
    expected_assessments = extract_last_table_names(content)

    if expected_assessments:
        print(
            f"[parser] {trace_id}: found {len(expected_assessments)} "
            f"expected assessments from last table"
        )
    else:
        # Fallback: check for explicit ## Expected Assessments section
        expected_section = re.search(
            r"## Expected Assessments\s*(.+?)(?=\n## |\Z)",
            content,
            re.DOTALL,
        )
        if expected_section:
            expected_assessments = re.findall(
                r"-\s+(.+)", expected_section.group(1)
            )
            print(
                f"[parser] {trace_id}: found {len(expected_assessments)} "
                f"expected assessments from explicit section"
            )
        else:
            print(f"[parser] {trace_id}: WARNING — no expected assessments found")

    return {
        "id": trace_id,
        "turns": turns,
        "expected_assessments": expected_assessments,
    }


# ============================================================================
# TRACE LOADER
# ============================================================================

def load_trace(path: Path) -> dict:
    trace_id = path.stem

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    elif path.suffix == ".md":
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return parse_markdown_trace(content, trace_id)

    else:
        raise ValueError(f"Unsupported trace format: {path.suffix}")


# ============================================================================
# TRACE EXECUTION
# ============================================================================

def run_trace(trace: dict) -> dict:
    """
    Replay a conversation trace against the live API.

    Recommendation extraction priority:
    1. structured recommendations[] field in response JSON
    2. markdown table in reply text (fallback)
    """
    messages = []
    recommendations = []

    user_turns = [t for t in trace["turns"] if t["role"] == "user"]

    for turn in user_turns:
        messages.append({
            "role": "user",
            "content": turn["content"],
        })

        success = False
        data = {}

        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"[eval] trace={trace.get('id')} | "
                    f"turn={len(messages)} | "
                    f"attempt={attempt + 1}"
                )

                response = httpx.post(
                    API_URL,
                    json={"messages": messages},
                    timeout=60,
                )

                if response.status_code == 429:
                    wait = min(BACKOFF_BASE ** attempt, 30)
                    print(f"[rate-limit] 429 — sleeping {wait}s")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                success = True
                break

            except httpx.ConnectError:
                print("[error] Cannot connect to API")
                time.sleep(5)

            except Exception as e:
                print(f"[error] {e}")
                time.sleep(3)

        if not success:
            print(f"[error] Giving up on trace {trace.get('id')}")
            break

        # Maintain conversation history
        messages.append({
            "role": "assistant",
            "content": data.get("reply", ""),
        })

        # 1. Structured recommendations (primary)
        if data.get("recommendations"):
            recommendations = [
                r["name"] for r in data["recommendations"] if "name" in r
            ]

        # 2. Markdown table fallback
        elif data.get("reply"):
            extracted = extract_names_from_table(data["reply"])
            if extracted:
                recommendations = extracted

        if data.get("end_of_conversation"):
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    return {
        "trace_id": trace.get("id", "unknown"),
        "expected": trace.get("expected_assessments", []),
        "recommended": recommendations,
        "chat_history": messages,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n[eval] Traces path: {TRACES_PATH}")

    if not TRACES_PATH.exists():
        print("[error] Trace directory not found")
        return

    files = sorted(TRACES_PATH.iterdir())
    trace_files = [f for f in files if f.suffix in (".json", ".md")]
    print(f"[eval] Found {len(trace_files)} trace files\n")

    results = []

    for path in trace_files:
        try:
            trace = load_trace(path)
            result = run_trace(trace)
            result["recall@10"] = recall_at_k(
                result["expected"], result["recommended"]
            )
            results.append(result)

            # Pretty print comparison
            expected = set(result["expected"])
            recommended = set(result["recommended"])
            matches = expected & recommended
            missing = expected - recommended
            extra = recommended - expected

            print("\n" + "=" * 70)
            print(f"Trace: {result['trace_id']}")

            print("\nEXPECTED:")
            for item in sorted(expected) or ["(none)"]:
                print(f"  - {item}")

            print("\nRECOMMENDED:")
            for item in sorted(recommended) or ["(none)"]:
                print(f"  - {item}")
 
            print("\nMATCHES:")
            for item in sorted(matches) or ["(none)"]:
                print(f"  [OK] {item}")

            print("\nMISSING (expected but not recommended):")
            for item in sorted(missing) or ["(none)"]:
                print(f"  [MISS] {item}")

            print("\nEXTRA (recommended but not expected):")
            for item in sorted(extra) or ["(none)"]:
                print(f"  [EXTRA] {item}")

            print(f"\nRECALL@10: {result['recall@10']:.2f}")
            print("=" * 70)

        except Exception as e:
            print(f"[error] Failed {path.name}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("[eval] No results to report")
        return

    mean = mean_recall_at_k(results)
    print(f"\n{'#' * 70}")
    print(f"Mean Recall@10: {mean:.4f} across {len(results)} traces")
    print(f"{'#' * 70}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"mean_recall@10": mean, "traces": results}, f, indent=2)

    print(f"\n[eval] Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()