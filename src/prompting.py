"""One prompt format shared by training and every evaluation condition."""
from __future__ import annotations

SYSTEM = (
    "You are an expert SQL assistant. Given a database schema with sample rows and a question, "
    "write one SQLite query that answers the question. Reply with the SQL only."
)


def user_turn(context: str, question: str) -> str:
    return f"Schema and data:\n{context}\n\nQuestion: {question}"


def messages(context: str, question: str, shots: list[dict] | None = None) -> list[dict]:
    """Chat messages for one question, optionally preceded by worked examples (few-shot)."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots or []:
        msgs.append({"role": "user", "content": user_turn(s["context"], s["question"])})
        msgs.append({"role": "assistant", "content": s["sql"]})
    msgs.append({"role": "user", "content": user_turn(context, question)})
    return msgs


def training_example(row: dict) -> dict:
    """Prompt + target answer in the chat format the trainer expects."""
    return {"messages": messages(row["context"], row["question"]) + [{"role": "assistant", "content": row["sql"]}]}
