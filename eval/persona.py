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
            # Clean up the user message
            user_msg = user_match.group(1).strip()
            turns.append(("user", user_msg))

        if agent_match:
            # Strip markdown table and metadata lines from agent reply
            agent_msg = agent_match.group(1)
            # Remove table lines
            agent_msg = re.sub(r"\|.+\|", "", agent_msg)
            # Remove _No recommendations_, _end_of_conversation_ lines
            agent_msg = re.sub(r"_.*?_", "", agent_msg)
            agent_msg = agent_msg.strip()
            if agent_msg:
                turns.append(("agent", agent_msg))

    if not turns:
        return "No persona available. Answer based on conversation context."

    # 3. Build structured persona from extracted turns
    # User messages = facts the hiring manager revealed
    # Agent questions = what clarifications were needed
    # User answers to agent questions = key facts

    user_facts = []
    qa_pairs = []

    for i, (role, msg) in enumerate(turns):
        if role == "user":
            user_facts.append(msg)
        elif role == "agent" and i + 1 < len(turns):
            # If agent asked a question and next turn is user answer
            next_role, next_msg = turns[i + 1]
            if next_role == "user" and "?" in msg:
                # Extract the question from agent message
                question_lines = [
                    line.strip() for line in msg.splitlines()
                    if "?" in line and line.strip()
                ]
                if question_lines:
                    qa_pairs.append({
                        "question": question_lines[-1],
                        "answer": next_msg,
                    })

    # Build the persona text
    persona_parts = [
        "You are a hiring manager or recruiter with the following known facts:",
        "",
        "## Facts about the role you are hiring for:",
    ]

    for fact in user_facts:
        # Clean up multiline facts
        clean = " ".join(fact.split())
        persona_parts.append(f"- {clean}")

    if qa_pairs:
        persona_parts.append("")
        persona_parts.append("## Additional details you have confirmed:")
        for qa in qa_pairs:
            q = " ".join(qa["question"].split())
            a = " ".join(qa["answer"].split())
            persona_parts.append(f"- When asked '{q}', you said: '{a}'")

    persona_parts.extend([
        "",
        "## Instructions:",
        "- Answer questions using ONLY the facts above",
        "- If asked something not in your facts, say 'I have no preference on that'",
        "- Do NOT invent new requirements",
        "- Keep responses short and natural (1-2 sentences)",
    ])

    return "\n".join(persona_parts)