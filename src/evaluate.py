"""Benchmark: base model (zero-shot, 3-shot) vs the QLoRA fine-tuned model.

All conditions use the same 4-bit base weights, the same prompt and greedy decoding,
so the only difference is the adapter (or the worked examples).
Metrics: execution accuracy (same rows as the reference query), exact match,
valid SQL rate, and single-request latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompting import messages
from src.sqlcheck import execution_match, extract_sql, normalize, run


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def pick_shots(train: list[dict]) -> list[dict]:
    """Three fixed worked examples of different difficulty, chosen from training data only."""
    want, shots = ["basic SQL", "aggregation", "single join"], []
    for w in want:
        shots.append(next(r for r in train if r["complexity"] == w and len(r["context"]) < 600))
    return shots


def load_model(base: str, adapter: str | None):
    compute = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0}, torch_dtype=compute)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"  # left padding for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def generate(model, tok, convs: list[list[dict]], batch: int, max_new: int) -> list[str]:
    outs = []
    for i in range(0, len(convs), batch):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs[i : i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return outs


@torch.no_grad()
def latency_ms(model, tok, convs: list[list[dict]], max_new: int) -> dict:
    times = []
    for c in convs:
        enc = tok(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": round(statistics.median(times)),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))]),
        "requests": len(times),
    }


def valid_sql(context: str, sql: str) -> bool:
    try:
        run(context, sql)
        return True
    except Exception:
        return False


def score(rows: list[dict], raw: list[str]) -> tuple[dict, list[dict]]:
    preds, by_cx = [], defaultdict(lambda: [0, 0])
    em = ex = ok = 0
    for r, text in zip(rows, raw):
        sql = extract_sql(text)
        e = execution_match(r["context"], r["sql"], sql)
        m = normalize(sql) == normalize(r["sql"])
        v = valid_sql(r["context"], sql)
        em += m
        ex += e
        ok += v
        by_cx[r["complexity"]][0] += e
        by_cx[r["complexity"]][1] += 1
        preds.append({"id": r["id"], "question": r["question"], "gold": r["sql"], "pred": sql, "exec_match": e, "exact": m, "valid": v})
    n = len(rows)
    metrics = {
        "n": n,
        "execution_accuracy": round(100 * ex / n, 1),
        "exact_match": round(100 * em / n, 1),
        "valid_sql_rate": round(100 * ok / n, 1),
        "execution_accuracy_by_complexity": {k: {"acc": round(100 * c / t, 1), "n": t} for k, (c, t) in sorted(by_cx.items())},
    }
    return metrics, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapter")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--latency-n", type=int, default=50)
    a = ap.parse_args()

    test = load_rows(a.test)
    shots = pick_shots(load_rows(a.train))
    zero = [messages(r["context"], r["question"]) for r in test]
    few = [messages(r["context"], r["question"], shots) for r in test]

    Path("results").mkdir(exist_ok=True)
    report = {"test_examples": len(test), "decoding": "greedy", "max_new_tokens": a.max_new, "conditions": {}}

    conditions = [("base_zero_shot", None, zero), ("base_3_shot", None, few), ("qlora_zero_shot", a.adapter, zero)]
    loaded = {}
    for name, adapter, convs in conditions:
        key = adapter or "base"
        if key not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            loaded[key] = load_model(a.model, adapter)
        model, tok = loaded[key]
        t0 = time.time()
        raw = generate(model, tok, convs, a.batch, a.max_new)
        metrics, preds = score(test, raw)
        metrics["eval_minutes"] = round((time.time() - t0) / 60, 1)
        metrics["latency_single_request"] = latency_ms(model, tok, convs[: a.latency_n], a.max_new)
        report["conditions"][name] = metrics
        with open(f"results/predictions_{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(name, json.dumps({k: v for k, v in metrics.items() if k != "execution_accuracy_by_complexity"}))

    report["gpu"] = torch.cuda.get_device_name(0)
    json.dump(report, open("results/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
