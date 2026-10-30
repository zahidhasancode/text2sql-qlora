"""Kaggle entry point: install pinned libraries, copy the code, train, evaluate, plot.

The code and data come from the private Kaggle dataset `text2sql-qlora-code`
(built by kaggle/push.sh from this repo). Everything the README reports is
written to /kaggle/working/outputs.
"""
import glob
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
# Kaggle may attach two T4s; a 4-bit model cannot be split by DataParallel, so use one GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(list(cmd))


sh(sys.executable, "-m", "pip", "install", "-q",
   "transformers==4.46.3", "peft==0.13.2", "trl==0.12.2", "accelerate==1.1.1",
   "bitsandbytes==0.49.2", "datasets==3.2.0", "matplotlib")

def find_code() -> str:
    """Locate the uploaded code: a folder holding src/train.py, or a zip that contains one."""
    hits = glob.glob("/kaggle/input/**/src/train.py", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        shutil.unpack_archive(z, f"{WORK}/unzipped")
        hits = glob.glob(f"{WORK}/unzipped/**/src/train.py", recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    sh("find", "/kaggle/input", "-maxdepth", "4")
    raise SystemExit("code not found under /kaggle/input")


shutil.copytree(find_code(), f"{WORK}/repo", dirs_exist_ok=True)
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
