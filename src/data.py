"""Build reproducible train / test splits from gretelai/synthetic_text_to_sql.

Kept examples must be checkable by execution:
  * the reference SQL is a SELECT (or WITH) query,
  * the schema + rows in `sql_context` load in SQLite,
  * the reference query runs and returns at least one row.
The official test split is used only for testing, so no test question is ever trained on.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from src.sqlcheck import run

DATASET = "gretelai/synthetic_text_to_sql"


def checkable(row: dict) -> bool:
    sql = row["sql"].strip().lower()
    if not (sql.startswith("select") or sql.startswith("with")):
        return False
    try:
        return len(run(row["sql_context"], row["sql"])) > 0
    except Exception:
        return False


def slim(row: dict) -> dict:
    return {
        "id": row["id"],
        "question": row["sql_prompt"],
        "context": row["sql_context"],
        "sql": row["sql"],
        "complexity": row["sql_complexity"],
        "domain": row["domain"],
    }


def sample(split, n: int, seed: int, max_context_chars: int) -> list[dict]:
    idx = list(range(len(split)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        row = split[i]
        if len(row["sql_context"]) > max_context_chars or not checkable(row):
            continue
        out.append(slim(row))
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-context-chars", type=int, default=2000)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    ds = load_dataset(DATASET)
    train = sample(ds["train"], a.n_train, a.seed, a.max_context_chars)
    test = sample(ds["test"], a.n_test, a.seed, a.max_context_chars)

    # Guard against duplicates across splits (same question text).
    test_q = {r["question"].strip().lower() for r in test}
    train = [r for r in train if r["question"].strip().lower() not in test_q]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"train={len(train)} test={len(test)} -> {out}/")


if __name__ == "__main__":
    main()
