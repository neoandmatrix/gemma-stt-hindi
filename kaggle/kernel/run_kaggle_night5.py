"""Kaggle runner for 04-gemma-stt-hindi, night 5: relabel the ~1,810 h of YODAS clips with Hima-ASR, then train.
Hima (Prahaladha/hima-asr, 0.46B) scored 14.2% CER on my clips vs YouTube-trained night 4's 26.3%; its author built it
mostly on relabelled audio. Question: how much of our gap is label quality?
Code: dataset YOUR_KAGGLE_USER/gemma-stt-hindi-code. Clips: outputs of the CPU notebooks gemma-stt-hindi-yodas-prep(2).
Earlier runners: run_kaggle_night1..4.py (this file started as night 4's copy).

1. Download Kathbath 50 h + Lahaja + EmbeddingGemma 2; copy every shard to /tmp as memory-mapped Arrow.
2. Relabel (both T4s, half the files each, until RELABEL_H): relabel.py -> text = Hima, yt_text = the old caption.
3. A/B on the SAME relabelled clips, same steps and recipe (night 4's yt_kb_k2), top-2 encoder layers:
   GPU 0  hl_k2 -> hl_kb_k2: Hima labels, then the Kathbath 50 h fine-tune
   GPU 1  yc_k2 -> yc_kb_k2: the original YouTube captions of those clips, then the same fine-tune
Outputs: relabel/stats_*.json and relabel_text.parquet (clip path, Hima text, YouTube text: reusable later).
"""
RELABEL_H = 4.5            # relabelling stops at this session age (or when every file is done)
DEADLINE_H = 11.4          # all training ends by this session age; outputs saved before the 12 h limit
FINETUNE_MIN = 45          # GPU 0 stage 2 (Kathbath fine-tune): ~2 epochs of 50 h at the measured speed
KB_SHARDS = 12             # Kathbath ~50 h
T4_HOURS_PER_HOUR = 210    # top-2 on a T4, measured night 4 (212 s of audio per second); for the LR schedule
                           # (night 4 assumed 150: GPU 0's stage 1 ended ~2 h early with the LR at its floor)

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
import pyarrow as pa  # noqa: E402
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
snapshot_download("Prahaladha/hima-asr", allow_patterns=["*.json", "*.py", "model.safetensors"])  # once, before 2 readers
say("Kathbath / Lahaja / EmbeddingGemma 2 / Hima-ASR downloaded")

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

# ---- 2. relabel with Hima
REL = Path("/tmp/relabel")
REL.mkdir(exist_ok=True)
until = T0 + RELABEL_H * 3600
procs = []
for g in range(max(n_gpu, 1)):
    files = yodas_files[g::max(n_gpu, 1)]
    logf = LOGS / f"relabel_{g}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    procs.append((subprocess.Popen([sys.executable, str(CODE / "relabel.py"), "--out", str(REL), "--until", str(until),
                                    *files], env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT, cwd=CODE), logf))
say(f"relabelling {len(yodas_files)} files on {len(procs)} GPU(s) until session age {RELABEL_H} h")
while any(p.poll() is None for p, _ in procs):
    time.sleep(60)
    for p, logf in procs:
        if p.poll() is None and (time.time() - logf.stat().st_mtime > 30 * 60 or time.time() > until + 1800):
            p.terminate(); p.wait(); say(f"{logf.name}: killed (stale log or past the relabel deadline)")
for p, logf in procs:
    say(f"{logf.name} exited with {p.returncode}")
shutil.copytree(REL, WORK / "relabel", ignore=shutil.ignore_patterns("*.arrow"), dirs_exist_ok=True)
hl_files = sorted(str(f) for f in REL.glob("*.arrow"))
assert hl_files, "relabelling produced nothing: see logs/relabel_*.log"

# the same clips with their YouTube captions (B side of the A/B): text <- yt_text
yc_files, keep = [], []
for f in hl_files:
    t = pa.ipc.open_file(pa.memory_map(f)).read_all()
    keep.append(pa.table({"path": [x["path"] for x in t.column("audio_filepath").to_pylist()],
                          "text": t["text"], "yt_text": t["yt_text"], "duration": t["duration"]}))
    yc = pa.table({"text": t["yt_text"], "duration": t["duration"], "audio_filepath": t["audio_filepath"]})
    out = ARROW / f"yc_{Path(f).name}"
    with pa.ipc.new_file(str(out), yc.schema) as w:
        w.write_table(yc)
    yc_files.append(str(out))
keep = pa.concat_tables(keep)
pq.write_table(keep, WORK / "relabel_text.parquet", compression="zstd")
hl_h = sum(keep["duration"].to_pylist()) / 3600
say(f"relabelled: {hl_h:.0f} h kept in {len(hl_files)} files ({keep.num_rows} clips) at {age_h():.2f} h")
assert hl_h > 50, f"only {hl_h:.0f} h relabelled"

# one character set for every run (so the fine-tune reuses the pretrain head): chars seen >= 50 times
cnt = Counter()
for col in (keep["text"].to_pylist(), keep["yt_text"].to_pylist(),
            *(pq.read_table(f, columns=["text"]).column("text").to_pylist() for f in kb_pq)):
    for x in col:
        cnt.update(" ".join(x.split()))
UNITS = WORK / "units_char.json"
UNITS.write_text(json.dumps({"type": "char", "chars": sorted(c for c, n in cnt.items() if n >= 50)}, ensure_ascii=False))
say(f"units: {len(json.loads(UNITS.read_text())['chars'])} chars")


# ---- 3. training
def train_cmd(run, files, minutes, hours_of_data, extra=(), unfreeze=2):
    epochs = max(0.05, minutes / 60 * T4_HOURS_PER_HOUR / hours_of_data)     # LR schedule spans the time budget
    return [sys.executable, str(CODE / "train.py"), "--run", run, "--units", str(UNITS), "--amp", "fp16",
            "--train", *files, "--eval", valid, "--eval-set", "lahaja", str(ARROW / "lahaja_*.arrow"),
            "--unfreeze", str(unfreeze), "--epochs", f"{epochs:.3f}", "--max-minutes", f"{minutes:.0f}",
            "--eval-every", "2000", "--workers", "3", *extra]


left_min = (DEADLINE_H - age_h()) * 60 - 15          # 15 min for start-up / final evals
pre_min = left_min - FINETUNE_MIN - 10
FT = ["--lr", "3e-4", "--enc-lr", "3e-5", "--warmup", "100"]
chains = {
    0: [("hl_k2", train_cmd("hl_k2", hl_files, pre_min, hl_h, ["--lr", "1e-3", "--warmup", "500"])),
        ("hl_kb_k2", train_cmd("hl_kb_k2", kb_files, FINETUNE_MIN, kb_h, [*FT, "--init", str(CODE / "runs/hl_k2/last.pt")]))],
    1: [("yc_k2", train_cmd("yc_k2", yc_files, pre_min, hl_h, ["--lr", "1e-3", "--warmup", "500"])),
        ("yc_kb_k2", train_cmd("yc_kb_k2", kb_files, FINETUNE_MIN, kb_h, [*FT, "--init", str(CODE / "runs/yc_k2/last.pt")]))],
}
say(f"plan: each GPU {pre_min:.0f} min on the relabelled clips then Kathbath {FINETUNE_MIN} min "
    f"(GPU0 Hima labels, GPU1 YouTube captions)")


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


for run in ("hl_k2", "yc_k2"):                       # stage 1's final weights, without the optimizer state
    lp = CODE / f"runs/{run}/last.pt"
    if lp.exists():
        torch.save({"model": torch.load(lp, map_location="cpu")["model"]}, lp.with_name("final.pt"))
for f in glob.glob(str(CODE / "runs/*/last.pt")):    # keep outputs small: best.pt / final.pt (weights only) stay
    os.remove(f)
summarize()
say("done")
