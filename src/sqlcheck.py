"""Run SQL against a throwaway SQLite database and compare results.

Execution accuracy asks "does the predicted query return the same rows as the
reference query?", which is fairer than string matching: two differently
written queries can be equally correct.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model answer (handles ``` fences and extra prose)."""
    m = FENCE.search(text)
    sql = m.group(1) if m else text
    sql = sql.strip()
    # Keep only the first statement; models sometimes add a second one or an explanation.
    first = sql.split(";")[0].strip()
    return first + ";" if first else ""


def normalize(sql: str) -> str:
    """Lower-case and collapse whitespace for exact-match comparison."""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def run(context: str, sql: str, timeout_ops: int = 200_000) -> list[tuple]:
    """Build the schema + rows from `context` in memory, run `sql`, return all rows.

    Raises sqlite3.Error (or ValueError) if the context or the query fails.
    A progress handler stops runaway queries.
    """
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(context)
        steps = {"n": 0}

        def guard() -> int:
            steps["n"] += 1
            return 1 if steps["n"] > timeout_ops else 0  # non-zero aborts the query

        con.set_progress_handler(guard, 1000)
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _round(row: tuple) -> tuple:
    return tuple(round(v, 4) if isinstance(v, float) else v for v in row)


def results_match(gold: list[tuple], pred: list[tuple], ordered: bool) -> bool:
    """Compare result sets. Order matters only if the reference query uses ORDER BY."""
    g = [_round(r) for r in gold]
    p = [_round(r) for r in pred]
    return g == p if ordered else Counter(g) == Counter(p)


def execution_match(context: str, gold_sql: str, pred_sql: str) -> bool:
    try:
        gold = run(context, gold_sql)
    except Exception:
        return False
    try:
        pred = run(context, pred_sql)
    except Exception:
        return False
    return results_match(gold, pred, ordered="order by" in gold_sql.lower())
