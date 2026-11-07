"""CPU-only Kaggle notebook (no GPU quota), round 2 (a separate notebook so round 1's output stays attachable): YODAS v3 Hindi shards -> short Opus clips for 04-gemma-stt-hindi.

One worker per CPU takes shards from a queue: download (hf_hub_download resumes after drops), convert with
yodas_prep.py, delete the tar. Each conversion is time-capped so everything is written before Kaggle's 12 h limit.
Output: /kaggle/working/yodas/yodas_hi_NNNN.parquet (~11 MB per hour of clips) + logs/prep.log.
The GPU training notebook attaches this notebook's output as a data source.
"""
DEADLINE_H = 11.3
# round 2 (prep2): shards not used by round 1 (gemma-stt-hindi-yodas-prep: 20..400 step 20 except 240/300/340/380),
# spread across the 418; more than fit in 11 h, the time limit decides how many get done
SHARDS = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390]

import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working")
OUT = WORK / "yodas"
DATA = Path("/tmp/yodas3")
LOG = WORK / "logs" / "prep.log"
for d in (OUT, DATA, LOG.parent):
    d.mkdir(parents=True, exist_ok=True)


def say(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def left_min():
    return (DEADLINE_H * 3600 - (time.time() - T0)) / 60


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av", "huggingface_hub==2.3.0"], check=True)
from huggingface_hub import hf_hub_download  # noqa: E402

CODE = Path(glob.glob("/kaggle/input/**/yodas_prep.py", recursive=True)[0]).parent
n_workers = os.cpu_count() or 4
free_gb = shutil.disk_usage("/tmp").free / 1e9
n_workers = max(1, min(n_workers, int(free_gb // 12)))
say(f"code from {CODE}, {os.cpu_count()} CPUs, {free_gb:.0f} GB free -> {n_workers} workers")

todo = queue.Queue()
for s in SHARDS:
    todo.put(s)


def worker(k):
    while left_min() > 30:
        try:
            shard = todo.get_nowait()
        except queue.Empty:
            return
        t = time.time()
        try:
            meta = hf_hub_download("espnet/yodas3", f"data/hi/metadata/{shard:04d}.parquet", repo_type="dataset",
                                   local_dir=DATA)
            for attempt in range(1, 9):
                try:
                    tar = hf_hub_download("espnet/yodas3", f"data/hi/audio/{shard:04d}.tar", repo_type="dataset",
                                          local_dir=DATA)
                    break
                except Exception as e:          # the partial file is kept; the next call resumes it
                    say(f"w{k} shard {shard:04d}: download try {attempt} failed: {type(e).__name__}: {str(e)[:120]}")
                    time.sleep(30)
            else:
                continue
            say(f"w{k} shard {shard:04d}: downloaded in {(time.time() - t) / 60:.0f} min")
            t = time.time()
            p = subprocess.run([sys.executable, str(CODE / "yodas_prep.py"), tar, meta,
                                str(OUT / f"yodas_hi_{shard:04d}.parquet"), "--max-minutes", f"{left_min() - 15:.0f}"],
                               capture_output=True, text=True)
            stats = (p.stdout.strip().splitlines() or ["{}"])[-1]
            say(f"w{k} shard {shard:04d}: {stats} in {(time.time() - t) / 60:.0f} min"
                + (f" | stderr: {p.stderr.strip()[-300:]}" if p.returncode else ""))
            os.remove(tar)
        except Exception as e:
            say(f"w{k} shard {shard:04d}: failed: {type(e).__name__}: {str(e)[:200]}")


ths = [threading.Thread(target=worker, args=(k,)) for k in range(n_workers)]
for th in ths:
    th.start()
for th in ths:
    th.join()

import pyarrow.parquet as pq  # noqa: E402
files = sorted(OUT.glob("*.parquet"))
h = sum(sum(pq.read_table(f, columns=["duration"]).column("duration").to_pylist()) for f in files) / 3600
gb = sum(f.stat().st_size for f in files) / 1e9
say(f"done: {len(files)} shard files, {h:.0f} h of clips, {gb:.1f} GB")
