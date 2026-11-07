"""Kaggle runner for 04-gemma-stt-hindi, night 4: YODAS v3 Hindi on ~2000 h of clips, top-2 vs top-4 encoder layers.
Code: dataset YOUR_KAGGLE_USER/gemma-stt-hindi-code. Clips: outputs of the CPU notebooks gemma-stt-hindi-yodas-prep(2).
Earlier runners: run_kaggle_night1.py (Kathbath ladder), night2 (convert + train in one session), night3 (982 h:
YODAS -> Kathbath beat mixed; this file started as its copy).

1. Download Kathbath 50 h + Lahaja + EmbeddingGemma 2; copy every train/eval shard to /tmp as memory-mapped Arrow
   (parquet decodes fully into RAM: two trainings on ~1000 h would need ~24 GB of Kaggle's 29 GB).
2. Two top-2 runs side by side on the 2 T4s, both scored on Kathbath valid + Lahaja:
   GPU 0  yt_k2 -> yt_kb_k2: YODAS only, then a short fine-tune on Kathbath 50 h (night 3's winner, more data)
   GPU 1  yt_k4 -> yt_kb_k4: the same with the top 4 encoder layers trained (more capacity for ~2000 h)
Stage 1's final weights are kept as runs/<run>/final.pt (night 3 deleted them with last.pt).
Stops are time-based (train.py --max-minutes). Data loading uses threads, not worker processes (both earlier nights
froze with process workers); a watchdog still kills a run whose log hasn't moved in 30 min.
"""
DEADLINE_H = 11.4          # all training ends by this session age; outputs saved before the 12 h limit
FINETUNE_MIN = 45          # GPU 0 stage 2 (Kathbath fine-tune): ~2 epochs of 50 h at the measured speed
KB_SHARDS = 12             # Kathbath ~50 h
T4_HOURS_PER_HOUR = 150    # top-2 on a T4, measured night 2 (150-200 s of audio per second); for the LR schedule

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
DATA = Path("/tmp/data")
ARROW = Path("/tmp/arrow")
LOGS = WORK / "logs"
for d in (LOGS, DATA, ARROW):
    d.mkdir(parents=True, exist_ok=True)


def age_h():
    return (time.time() - T0) / 3600


def say(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(LOGS / "runner.log", "a") as f:
        f.write(line + "\n")


# ---- code + packages
src = Path(glob.glob("/kaggle/input/**/train.py", recursive=True)[0]).parent
CODE = WORK / "code"
shutil.copytree(src, CODE, dirs_exist_ok=True)
say(f"code from {src}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.21.0", "tokenizers==0.23.3",
                "huggingface_hub==2.3.0", "jiwer", "soundfile"], check=True)
from kaggle_secrets import UserSecretsClient  # noqa: E402
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

sys.path.insert(0, str(CODE))
from data import to_arrow  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

n_gpu = torch.cuda.device_count()
say(f"torch {torch.__version__}, {n_gpu} GPU(s), {os.cpu_count()} CPUs")

# ---- 1. data
for i in range(KB_SHARDS):
    hf_hub_download("ai4bharat/Kathbath", f"hindi/train-{i:05d}-of-00032.parquet", repo_type="dataset",
                    local_dir=DATA / "kathbath")
hf_hub_download("ai4bharat/Kathbath", "hindi/valid-00000-of-00002.parquet", repo_type="dataset",
                local_dir=DATA / "kathbath")
snapshot_download("ai4bharat/Lahaja", repo_type="dataset", local_dir=DATA / "lahaja",
                  allow_patterns=["data/test-0000[0-3]-of-00004.parquet"])
snapshot_download("google/embeddinggemma-2")
say("Kathbath / Lahaja / EmbeddingGemma 2 downloaded")

yodas_pq = sorted(glob.glob("/kaggle/input/**/yodas_hi_*.parquet", recursive=True))
yodas_pq = [f for f in yodas_pq if pq.ParquetFile(f).metadata.num_rows > 0]
kb_pq = [str(DATA / "kathbath" / f"hindi/train-{i:05d}-of-00032.parquet") for i in range(KB_SHARDS)]


def arrow(paths, tag):
    out = []
    for p in paths:
        a = ARROW / f"{tag}_{Path(p).stem}.arrow"
        to_arrow(p, a)
        out.append(str(a))
    return out


yodas_files = arrow(yodas_pq, "yodas")
kb_files = arrow(kb_pq, "kb")
valid = arrow([DATA / "kathbath/hindi/valid-00000-of-00002.parquet"], "valid")[0]
lahaja = arrow(sorted(glob.glob(str(DATA / "lahaja/data/*.parquet"))), "lahaja")


def hours(files):
    return sum(sum(pq.read_table(f, columns=["duration"]).column("duration").to_pylist()) for f in files) / 3600


yodas_h, kb_h = hours(yodas_pq), hours(kb_pq)
say(f"data ready at {age_h():.2f} h: YODAS {yodas_h:.0f} h in {len(yodas_files)} files, Kathbath {kb_h:.0f} h, "
    f"{shutil.disk_usage('/tmp').free / 1e9:.0f} GB disk left")
assert yodas_h > 50, f"only {yodas_h:.0f} h of YODAS clips found: is the prep notebook attached?"

# one character set for every run (so the fine-tune reuses the pretrain head): chars seen >= 50 times
cnt = Counter()
for f in yodas_pq + kb_pq:
    for x in pq.read_table(f, columns=["text"]).column("text").to_pylist():
        cnt.update(" ".join(x.split()))
UNITS = WORK / "units_char.json"
UNITS.write_text(json.dumps({"type": "char", "chars": sorted(c for c, n in cnt.items() if n >= 50)}, ensure_ascii=False))
say(f"units: {len(json.loads(UNITS.read_text())['chars'])} chars")


# ---- 2. training
def train_cmd(run, files, minutes, hours_of_data, extra=(), unfreeze=2):
    epochs = max(0.05, minutes / 60 * T4_HOURS_PER_HOUR / hours_of_data)     # LR schedule spans the time budget
    return [sys.executable, str(CODE / "train.py"), "--run", run, "--units", str(UNITS), "--amp", "fp16",
            "--train", *files, "--eval", valid, "--eval-set", "lahaja", str(ARROW / "lahaja_*.arrow"),
            "--unfreeze", str(unfreeze), "--epochs", f"{epochs:.3f}", "--max-minutes", f"{minutes:.0f}",
            "--eval-every", "2000", "--workers", "3", *extra]


left_min = (DEADLINE_H - age_h()) * 60 - 15          # 15 min for start-up / final evals
pre_min = left_min - FINETUNE_MIN - 10
chains = {
    0: [("yt_k2", train_cmd("yt_k2", yodas_files, pre_min, yodas_h, ["--lr", "1e-3", "--warmup", "500"])),
        ("yt_kb_k2", train_cmd("yt_kb_k2", kb_files, FINETUNE_MIN, kb_h,
                               ["--lr", "3e-4", "--enc-lr", "3e-5", "--warmup", "100",
                                "--init", str(CODE / "runs/yt_k2/last.pt")]))],
    1: [("yt_k4", train_cmd("yt_k4", yodas_files, pre_min, yodas_h, ["--lr", "1e-3", "--warmup", "500"], unfreeze=4)),
        ("yt_kb_k4", train_cmd("yt_kb_k4", kb_files, FINETUNE_MIN, kb_h,
                               ["--lr", "3e-4", "--enc-lr", "3e-5", "--warmup", "100",
                                "--init", str(CODE / "runs/yt_k4/last.pt")], unfreeze=4))],
}
say(f"plan: each GPU YODAS {pre_min:.0f} min then Kathbath {FINETUNE_MIN} min (GPU0 top-2, GPU1 top-4)")


def run_chain(gpu, jobs):
    for name, cmd in jobs:
        if age_h() > DEADLINE_H - 0.2:
            say(f"{name}: no time left, skipped"); return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % max(n_gpu, 1)))
        logf = LOGS / f"{name}.log"
        say(f"start {name} on GPU {env['CUDA_VISIBLE_DEVICES']}")
        p = subprocess.Popen(cmd, env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT, cwd=CODE)
        while p.poll() is None:
            time.sleep(60)
            stale = time.time() - logf.stat().st_mtime > 30 * 60
            if stale or age_h() > DEADLINE_H + 0.2:
                p.terminate(); p.wait()
                say(f"{name} killed ({'no log output for 30 min' if stale else 'past the deadline'})")
                break
        say(f"{name} exited with {p.returncode}")


ths = [threading.Thread(target=run_chain, args=(g, jobs)) for g, jobs in chains.items()]
for th in ths:
    th.start()
for th in ths:
    th.join()


def summarize():
    out = {}
    for logf in sorted(glob.glob(str(CODE / "runs/*/log.jsonl"))):
        recs = [json.loads(x) for x in open(logf) if '"step"' in x]
        if recs:
            out[Path(logf).parent.name] = dict(final=recs[-1], best=min(recs, key=lambda r: r["cer"]))
    (WORK / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, r in out.items():
        b = r["best"]
        say(f"SUMMARY {name}: best valid CER {b['cer']:.3f} WER {b['wer']:.3f} | "
            f"lahaja CER {b.get('cer_lahaja', float('nan')):.3f} | {b['hours_seen']} h seen")


for run in ("yt_k2", "yt_k4"):                       # stage 1's final weights, without the optimizer state
    lp = CODE / f"runs/{run}/last.pt"
    if lp.exists():
        torch.save({"model": torch.load(lp, map_location="cpu")["model"]}, lp.with_name("final.pt"))
for f in glob.glob(str(CODE / "runs/*/last.pt")):    # keep outputs small: best.pt / final.pt (weights only) stay
    os.remove(f)
summarize()
say("done")
