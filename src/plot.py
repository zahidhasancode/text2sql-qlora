"""Draw the training/validation loss curve and the benchmark bar chart from results/."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def loss_curve(log_path: str = "results/train_log.json", out: str = "results/loss.png") -> None:
    hist = json.load(open(log_path))["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*tr), label="train loss")
    if ev:
        plt.plot(*zip(*ev), marker="o", label="validation loss")
    plt.xlabel("optimizer step")
    plt.ylabel("loss")
    plt.title("QLoRA fine-tuning: loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)


def bars(metrics_path: str = "results/metrics.json", out: str = "results/benchmark.png") -> None:
    c = json.load(open(metrics_path))["conditions"]
    names = list(c)
    vals = [c[n]["execution_accuracy"] for n in names]
    plt.figure(figsize=(7, 4))
    b = plt.bar([n.replace("_", " ") for n in names], vals)
    for rect, v in zip(b, vals):
        plt.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
    plt.ylabel("execution accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Text-to-SQL on 500 held-out questions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)


if __name__ == "__main__":
    loss_curve()
    bars()
