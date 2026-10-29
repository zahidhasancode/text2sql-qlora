"""Kaggle entry point: install pinned libraries, fetch the repo, prepare data, train, evaluate, plot.

Everything the README reports is written to /kaggle/working/outputs.
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/zahidhasancode/text2sql-qlora.git"
WORK = "/kaggle/working"


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(list(cmd))


sh(sys.executable, "-m", "pip", "install", "-q",
   "transformers==4.46.3", "peft==0.13.2", "trl==0.12.2", "accelerate==1.1.1",
   "bitsandbytes==0.45.0", "datasets==3.2.0", "matplotlib")
sh("git", "clone", "--depth", "1", REPO, f"{WORK}/repo")
os.chdir(f"{WORK}/repo")
sh("nvidia-smi")
sh(sys.executable, "-m", "pytest", "-q", "tests")
sh(sys.executable, "-m", "src.train")
sh(sys.executable, "-m", "src.evaluate")
sh(sys.executable, "-m", "src.plot")

out = f"{WORK}/outputs"
shutil.copytree("results", f"{out}/results", dirs_exist_ok=True)
shutil.copytree("outputs/adapter", f"{out}/adapter", dirs_exist_ok=True)
print("finished; outputs in", out)
