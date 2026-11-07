"""Kaggle runner for 04-gemma-stt-hindi, night 2: YODAS v3 Hindi (YouTube) pretraining.
Code comes from the private dataset YOUR_KAGGLE_USER/gemma-stt-hindi-code. Night 1's runner: run_kaggle_night1.py.

1. Download 3 YODAS v3 Hindi shards (8.5 GB each) from Hugging Face and convert them to ~300 h of 2-12 s clips (yodas_prep.py, CPU),
   while Kathbath 20 h + Lahaja + EmbeddingGemma 2 download. Clips are kept in /kaggle/working/yodas for reuse.
2. Two top-2 runs side by side on the 2 T4s, both scored on Kathbath valid + Lahaja:
   GPU 0  yt_k2 -> yt_kb_k2: YODAS only, then a short fine-tune on Kathbath 20 h (pretrain -> fine-tune)
   GPU 1  mix_k2:            YODAS + Kathbath 20 h shuffled together for the same total time
   Stops are time-based (train.py --max-minutes) so everything ends before Kaggle's 12 h limit.
A watchdog kills a run whose log hasn't moved in 30 min (night 1: a run hung for 3 h after an eval).
"""
DEADLINE_H = 11.4          # all training ends by this session age; outputs saved before the 12 h limit
FINETUNE_MIN = 80          # GPU 0 stage 2 (Kathbath fine-tune) length
YODAS_SHARDS = [0, 140, 280]    # spread across the 418 shards for more channels/topics
CONVERT_MAX_MIN = 150           # per shard; whatever is converted by then is kept
KB_SHARDS = 5                   # Kathbath 20 h (night 1: 20 h vs 50 h made no difference off-domain)
T4_HOURS_PER_HOUR = 34          # top-2 speed on a T4 (night 1), only for the LR schedule length

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
DATA = Path("/tmp/data")                      # downloads: not saved as output
YODAS = WORK / "yodas"                        # converted clips: saved as output for later sessions
LOGS = WORK / "logs"
for d in (LOGS, YODAS, DATA):
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
if not shutil.which("ffmpeg"):
    subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True, check=True)
import soundfile as sf  # noqa: E402
CODEC = "opus" if "OPUS" in sf.available_subtypes("OGG") else "flac"
say(f"ffmpeg {shutil.which('ffmpeg')}, libsndfile {sf.__libsndfile_version__}, clips as {CODEC}")

from kaggle_secrets import UserSecretsClient  # noqa: E402
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

n_gpu = torch.cuda.device_count()
say(f"torch {torch.__version__}, {n_gpu} GPU(s), {os.cpu_count()} CPUs")


# ---- 1. data: YODAS conversion (CPU) and the other downloads at the same time
def fetch_other():
    for i in range(KB_SHARDS):
        hf_hub_download("ai4bharat/Kathbath", f"hindi/train-{i:05d}-of-00032.parquet", repo_type="dataset",
                        local_dir=DATA / "kathbath")
    hf_hub_download("ai4bharat/Kathbath", "hindi/valid-00000-of-00002.parquet", repo_type="dataset",
                    local_dir=DATA / "kathbath")
    snapshot_download("ai4bharat/Lahaja", repo_type="dataset", local_dir=DATA / "lahaja",
                      allow_patterns=["data/test-0000[0-3]-of-00004.parquet"])
    snapshot_download("google/embeddinggemma-2")
    say("Kathbath / Lahaja / EmbeddingGemma 2 ready")


def convert(shard):
    """Download the shard to disk (hf_hub_download resumes after a dropped connection; night 2 try 1 streamed it with
    curl, the connections dropped after 6-44 min and each shard stopped at ~50 of ~720 videos), convert, delete."""
    meta = hf_hub_download("espnet/yodas3", f"data/hi/metadata/{shard:04d}.parquet", repo_type="dataset",
                           local_dir=DATA / "yodas3")
    t = time.time()
    for attempt in range(1, 9):
        try:
            tar = hf_hub_download("espnet/yodas3", f"data/hi/audio/{shard:04d}.tar", repo_type="dataset",
                                  local_dir=DATA / "yodas3")
            break
        except Exception as e:                        # partial file is kept; the next call resumes it
            say(f"YODAS shard {shard:04d}: download try {attempt} failed: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(30)
    else:
        say(f"YODAS shard {shard:04d}: giving up"); return
    gb = os.path.getsize(tar) / 1e9
    say(f"YODAS shard {shard:04d}: {gb:.1f} GB downloaded in {(time.time() - t) / 60:.0f} min")
    out = YODAS / f"yodas_hi_{shard:04d}.parquet"
    t = time.time()
    p = subprocess.run([sys.executable, str(CODE / "yodas_prep.py"), tar, meta, str(out), "--codec", CODEC,
                        "--max-minutes", str(CONVERT_MAX_MIN)], capture_output=True, text=True)
    stats = (p.stdout.strip().splitlines() or ["{}"])[-1]
    say(f"YODAS shard {shard:04d}: {stats} in {(time.time() - t) / 60:.0f} min"
        + (f" | stderr: {p.stderr.strip()[-300:]}" if p.returncode else ""))
    os.remove(tar)


free_gb = shutil.disk_usage("/tmp").free / 1e9
SLOTS = threading.Semaphore(max(1, min(len(YODAS_SHARDS), int(free_gb // 12))))   # 8.5 GB tar + output each
say(f"disk free in /tmp: {free_gb:.0f} GB -> {SLOTS._value} shard(s) at a time")


def convert_slot(shard):
    with SLOTS:
        convert(shard)
threads = [threading.Thread(target=fetch_other)] + [threading.Thread(target=convert_slot, args=(s,)) for s in YODAS_SHARDS]
for th in threads:
    th.start()
for th in threads:
    th.join()

import pyarrow.parquet as pq  # noqa: E402


def hours(files):
    return sum(sum(pq.read_table(f, columns=["duration"]).column("duration").to_pylist()) for f in files) / 3600


yodas_files = [str(f) for f in sorted(YODAS.glob("*.parquet")) if pq.ParquetFile(f).metadata.num_rows > 0]
kb_files = [str(DATA / "kathbath" / f"hindi/train-{i:05d}-of-00032.parquet") for i in range(KB_SHARDS)]
yodas_h, kb_h = hours(yodas_files), hours(kb_files)
say(f"data ready at {age_h():.2f} h: YODAS {yodas_h:.0f} h in {len(yodas_files)} files, Kathbath {kb_h:.0f} h")
assert yodas_h > 20, "YODAS conversion failed (see the shard lines above)"

# one character set for every run (so the fine-tune can reuse the pretrain head): chars seen >= 50 times
cnt = Counter()
for f in yodas_files + kb_files:
    for x in pq.read_table(f, columns=["text"]).column("text").to_pylist():
        cnt.update(" ".join(x.split()))
UNITS = WORK / "units_char.json"
UNITS.write_text(json.dumps({"type": "char", "chars": sorted(c for c, n in cnt.items() if n >= 50)}, ensure_ascii=False))
say(f"units: {len(json.loads(UNITS.read_text())['chars'])} chars")


# ---- 2. training
def train_cmd(run, files, minutes, hours_of_data, extra=()):
    epochs = max(0.05, minutes / 60 * T4_HOURS_PER_HOUR / hours_of_data)     # LR schedule spans the time budget
    return [sys.executable, str(CODE / "train.py"), "--run", run, "--units", str(UNITS), "--amp", "fp16",
            "--train", *files, "--eval", str(DATA / "kathbath/hindi/valid-00000-of-00002.parquet"),
            "--eval-set", "lahaja", str(DATA / "lahaja/data/*.parquet"), "--unfreeze", "2",
            "--epochs", f"{epochs:.3f}", "--max-minutes", f"{minutes:.0f}", "--eval-every", "1000",
            "--workers", "2", *extra]


left_min = (DEADLINE_H - age_h()) * 60 - 15          # 15 min for start-up / final evals
pre_min = left_min - FINETUNE_MIN - 10
chains = {
    0: [("yt_k2", train_cmd("yt_k2", yodas_files, pre_min, yodas_h, ["--lr", "1e-3", "--warmup", "500"])),
        ("yt_kb_k2", train_cmd("yt_kb_k2", kb_files, FINETUNE_MIN, kb_h,
                               ["--lr", "3e-4", "--enc-lr", "3e-5", "--warmup", "100",
                                "--init", str(CODE / "runs/yt_k2/last.pt")]))],
    1: [("mix_k2", train_cmd("mix_k2", yodas_files + kb_files, left_min, yodas_h + kb_h,
                             ["--lr", "1e-3", "--warmup", "500"]))],
}
say(f"plan: GPU0 YODAS {pre_min:.0f} min then Kathbath {FINETUNE_MIN} min | GPU1 mixed {left_min:.0f} min")


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


for f in glob.glob(str(CODE / "runs/*/last.pt")):    # keep outputs small: best.pt (weights only) stays
    os.remove(f)
summarize()
say("done")
