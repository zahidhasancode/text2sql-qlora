# Text-to-SQL with QLoRA (Qwen2.5-1.5B-Instruct)

Fine-tuning a small open model to turn questions into SQLite queries, measured by **execution accuracy**
(does the predicted query return the same rows as the reference?) on 500 held-out questions.

**Status:** training and benchmark run in progress. Results, loss curves and the adapter will be added here.

- Data: [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache-2.0); only questions whose reference query runs in SQLite and returns rows are kept (`src/data.py`).
- Training: QLoRA (4-bit NF4, LoRA r=16 on all linear layers, loss on the answer only), Kaggle T4 (`src/train.py`).
- Benchmark: base zero-shot vs base 3-shot vs fine-tuned, same weights and greedy decoding (`src/evaluate.py`).
