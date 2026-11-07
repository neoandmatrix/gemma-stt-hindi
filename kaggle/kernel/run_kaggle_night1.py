"""Kaggle runner for 04-gemma-stt-hindi (code comes from the private dataset YOUR_KAGGLE_USER/gemma-stt-hindi-code).

MODE = "smoke": 1 shard, 60 steps per setting, ~10 min. Checks packages, paths, the HF token, GPU speed.
MODE = "full":  frozen and top-2 runs side by side on the 2 T4s, at 20 h then 50 h of Kathbath Hindi,
                each scored on Kathbath valid + Lahaja (accents). Logs and best.pt in /kaggle/working.
"""
MODE = "full"
DEADLINE_H = 11.0          # stop training at this session age so outputs are saved before Kaggle's 12 h limit

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
DATA = Path("/tmp/data")                      # not /kaggle/working: data shouldn't be saved as output
LOGS = WORK / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def say(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(LOGS / "runner.log", "a") as f:
        f.write(line + "\n")


# ---- code: copy from the attached dataset to a writable dir (train.py writes runs/ next to itself)
src = Path(glob.glob("/kaggle/input/**/train.py", recursive=True)[0]).parent
CODE = WORK / "code"
shutil.copytree(src, CODE, dirs_exist_ok=True)
say(f"code from {src}")

# exact versions that work locally: Kaggle's preinstalled transformers predates EmbeddingGemma 2
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.21.0", "tokenizers==0.23.3",
                "huggingface_hub==2.3.0", "jiwer", "soundfile"], check=True)
import transformers  # noqa: E402
assert "embedding_gemma2" in transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES, transformers.__version__

from kaggle_secrets import UserSecretsClient  # noqa: E402
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

n_gpu = torch.cuda.device_count()
say(f"torch {torch.__version__}, {n_gpu} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")


def fetch(n_train_shards):
    files = [f"hindi/train-{i:05d}-of-00032.parquet" for i in range(n_train_shards)] + \
            ["hindi/valid-00000-of-00002.parquet"]
    for f in files:
        hf_hub_download("ai4bharat/Kathbath", f, repo_type="dataset", local_dir=DATA / "kathbath")
    snapshot_download("ai4bharat/Lahaja", repo_type="dataset", local_dir=DATA / "lahaja",
                      allow_patterns=["data/test-0000[0-3]-of-00004.parquet"] if MODE == "full"
                      else ["data/test-00000-of-00004.parquet"])
    snapshot_download("google/embeddinggemma-2")
    say(f"data ready: {n_train_shards} train shards")


def train_cmd(run, n_shards, unfreeze, epochs, extra=()):
    shards = [str(DATA / "kathbath" / f"hindi/train-{i:05d}-of-00032.parquet") for i in range(n_shards)]
    return [sys.executable, str(CODE / "train.py"), "--run", run, "--unit-type", "char", "--amp", "fp16",
            "--train", *shards, "--eval", str(DATA / "kathbath/hindi/valid-00000-of-00002.parquet"),
            "--eval-set", "lahaja", str(DATA / "lahaja/data/*.parquet"),
            "--unfreeze", str(unfreeze), "--epochs", str(epochs), "--lr", "1e-3", "--warmup", "300",
            "--workers", "2", *extra]


def run_parallel(jobs):
    """jobs: [(name, cmd)] -> run each on its own GPU at once (or one after another with 1 GPU)."""
    procs = []
    for k, (name, cmd) in enumerate(jobs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % max(n_gpu, 1)))
        log = open(LOGS / f"{name}.log", "w")
        say(f"start {name} on GPU {env['CUDA_VISIBLE_DEVICES']}")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=CODE)
        procs.append((name, p))
        if n_gpu < 2:
            p.wait()
    for name, p in procs:
        left = T0 + DEADLINE_H * 3600 - time.time()
        try:
            say(f"{name} exited with {p.wait(timeout=max(left, 1))}")
        except subprocess.TimeoutExpired:
            p.terminate(); p.wait()
            say(f"{name} stopped at the {DEADLINE_H} h deadline (best.pt is from its last eval)")


def summarize():
    out = {}
    for logf in sorted(glob.glob(str(CODE / "runs/*/log.jsonl"))):
        recs = [json.loads(x) for x in open(logf) if '"step"' in x]
        if recs:
            best = min(recs, key=lambda r: r["cer"])
            out[Path(logf).parent.name] = dict(final=recs[-1], best=best)
    (WORK / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, r in out.items():
        b = r["best"]
        say(f"SUMMARY {name}: best valid CER {b['cer']:.3f} WER {b['wer']:.3f} | "
            f"lahaja CER {b.get('cer_lahaja', float('nan')):.3f} | {b['hours_seen']} h seen")


if MODE == "smoke":
    fetch(1)
    run_parallel([(f"smoke_k{k}", train_cmd(f"smoke_k{k}", 1, k, 8,
                                            ["--max-steps", "60", "--eval-every", "60", "--eval-clips", "100"]))
                  for k in (0, 2)])
else:
    fetch(12)
    # T4 speed (smoke test): frozen ~48, top-2 ~36 s audio/s -> phase 1 ~2.8 h, phase 2 ~4.2 h
    for hours, shards, epochs in ((20, 5, 5), (50, 12, 3)):
        if (time.time() - T0) / 3600 > 6.5:
            say(f"skipping {hours} h: not enough session time left")
            break
        run_parallel([(f"h{hours}_k{k}", train_cmd(f"h{hours}_k{k}", shards, k, epochs, ["--eval-every", "1000"]))
                      for k in (0, 2)])
        summarize()

# keep outputs small: best.pt (weights only) stays, last.pt (weights + optimizer) goes
for f in glob.glob(str(CODE / "runs/*/last.pt")):
    os.remove(f)
summarize()
say("done")
