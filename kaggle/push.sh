#!/bin/zsh
# Upload the code + data as a private Kaggle dataset, then start the GPU training notebook.
# Usage: KAGGLE_USER=<your username> kaggle/push.sh
set -e
ROOT=${0:A:h:h}
USER=${KAGGLE_USER:?set KAGGLE_USER}
K="$ROOT/.venv/bin/kaggle"
B=$(mktemp -d)

mkdir -p "$B/data/code"
cp -R "$ROOT/src" "$ROOT/tests" "$ROOT/data" "$ROOT/requirements.txt" "$B/data/code/"
find "$B/data/code" -name __pycache__ -prune -exec rm -rf {} +
cat > "$B/data/dataset-metadata.json" <<EOF
{"title": "text2sql-qlora-code", "id": "$USER/text2sql-qlora-code", "licenses": [{"name": "MIT"}]}
EOF
if "$K" datasets status "$USER/text2sql-qlora-code" >/dev/null 2>&1; then
  "$K" datasets version -p "$B/data" -m "update" --dir-mode zip
else
  "$K" datasets create -p "$B/data" --dir-mode zip   # private by default
fi

mkdir -p "$B/kernel"
cp "$ROOT/kaggle/run.py" "$B/kernel/run.py"
cat > "$B/kernel/kernel-metadata.json" <<EOF
{
  "id": "$USER/text2sql-qlora",
  "title": "text2sql-qlora",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$USER/text2sql-qlora-code"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
"$K" kernels push -p "$B/kernel"
echo "status: $K kernels status $USER/text2sql-qlora"
