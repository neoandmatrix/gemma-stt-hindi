"""Kaggle runner for 04-gemma-stt-hindi, night 6: label the rest of YODAS with Indic-Transcribe, train on everything.
Night 5 (run_kaggle_night5.py, notebook gemma-stt-hindi-ladder v17): Hima relabelled 874 h (780 h kept) and
Hima labels beat YouTube captions on the same clips (Kathbath 3.7% vs 4.2%, Lahaja 9.6% vs 11.6%).
Code: dataset YOUR_KAGGLE_USER/gemma-stt-hindi-code. Clips: outputs of gemma-stt-hindi-yodas-prep(2). Night 5's Hima labels:
relabel_text.parquet in the attached gemma-stt-hindi-ladder output.

1. Download Kathbath 50 h + Lahaja + EmbeddingGemma 2 + Indic-Transcribe-core; copy every shard to /tmp as Arrow.
2. Rebuild night 5's 780 h of Hima-labelled clips (audio from YODAS, text from relabel_text.parquet).
3. Relabel the remaining ~940 h with Indic-Transcribe-core (bodhan-ai, 1.2B, the shoot-out's best: 9.3% CER on my
   clips vs Hima's 14.2%), batched fp16 on both T4s, until RELABEL_H. It runs with its own transformers 4.57 in
   /tmp/itpkgs (training uses 5.21). If the gated model can't be downloaded, Hima labels the rest instead.
4. (TRAIN = True only) Train on Hima 780 h + the new labels (mixed teachers):
   GPU 0  all_k2 -> all_kb_k2: top-2 encoder layers (the recipe so far), then the Kathbath 50 h fine-tune
   GPU 1  all_k4 -> all_kb_k4: top-4 (with clean labels, capacity may now matter), then the same fine-tune
Outputs: relabel/stats_*.json, it_relabel_text.parquet (path, teacher text, YouTube text: reusable later).
"""
RELABEL_H = 6.3            # relabelling stops at this session age (or when every clip is done: ~5 h at ~93 s/s/T4)
TRAIN = False              # 2026-11-06: only ~8 GPU h left this week -> tonight relabel only (labels saved to
                           # it_relabel_text.parquet); training on everything runs after the weekly quota reset
DEADLINE_H = 11.4          # all training ends by this session age; outputs saved before the 12 h limit
FINETUNE_MIN = 45          # GPU 0 stage 2 (Kathbath fine-tune): ~2 epochs of 50 h at the measured speed
KB_SHARDS = 12             # Kathbath ~50 h
T4_HOURS_PER_HOUR = {2: 210, 4: 163}   # s of audio per s on a T4 by trained top layers (night 4); for the LR schedule

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
src = Path([p for p in glob.glob("/kaggle/input/**/train.py", recursive=True) if "gemma-stt-hindi-code" in p][0]).parent
CODE = WORK / "code"
shutil.copytree(src, CODE, dirs_exist_ok=True)
say(f"code from {src}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.21.0", "tokenizers==0.23.3",
                "huggingface_hub==2.3.0", "jiwer", "soundfile"], check=True)
# Indic-Transcribe's code needs transformers 4.5x: its own copy, used only by the relabel processes (PYTHONPATH)
ITPKGS = Path("/tmp/itpkgs")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(ITPKGS),
                "transformers==4.57.3", "tokenizers==0.22.2", "huggingface_hub==0.36.2", "sentencepiece"], check=True)
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
TEACHER = "it"
try:                                                  # once, before 2 readers
    snapshot_download("bodhan-ai/indic-transcribe-core", ignore_patterns=["nemo/*", "*.nemo", "banner.png"])
except Exception as e:                                # gated: the HF token must have accepted its terms
    say(f"Indic-Transcribe download failed ({type(e).__name__}: {str(e)[:200]}): falling back to Hima")
    TEACHER = "hima"
    snapshot_download("Prahaladha/hima-asr", allow_patterns=["*.json", "*.py", "model.safetensors"])
say(f"Kathbath / Lahaja / EmbeddingGemma 2 / teacher ({TEACHER}) downloaded")

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
    def durs(f):
        t = (pa.ipc.open_file(pa.memory_map(str(f))).read_all() if str(f).endswith(".arrow")
             else pq.read_table(f, columns=["duration"]))
        return sum(t.column("duration").to_pylist())
    return sum(durs(f) for f in files) / 3600


yodas_h, kb_h = hours(yodas_pq), hours(kb_pq)
say(f"data ready at {age_h():.2f} h: YODAS {yodas_h:.0f} h in {len(yodas_files)} files, Kathbath {kb_h:.0f} h, "
    f"{shutil.disk_usage('/tmp').free / 1e9:.0f} GB disk left")
assert yodas_h > 50, f"only {yodas_h:.0f} h of YODAS clips found: is the prep notebook attached?"

# ---- 2. night 5's Hima labels: YODAS audio + relabel_text.parquet text
import pyarrow.compute as pc  # noqa: E402
done_pq = [p for p in glob.glob("/kaggle/input/**/relabel_text.parquet", recursive=True)][0]
hima = pq.read_table(done_pq, columns=["path", "text"])
hima_text = dict(zip(hima["path"].to_pylist(), hima["text"].to_pylist()))
hima_paths = hima["path"].combine_chunks()
hl_files = []
for f in (yodas_files if TRAIN else []):
    t = pa.ipc.open_file(pa.memory_map(f)).read_all()
    t = t.filter(pc.is_in(pc.struct_field(t["audio_filepath"], "path"), value_set=hima_paths))
    if t.num_rows == 0:
        continue
    texts = [hima_text[x] for x in pc.struct_field(t["audio_filepath"], "path").to_pylist()]
    t = pa.table({"text": pa.array(texts, pa.string()), "duration": t["duration"], "audio_filepath": t["audio_filepath"]})
    out = ARROW / f"hima_{Path(f).name}"
    with pa.ipc.new_file(str(out), t.schema) as w:
        w.write_table(t)
    hl_files.append(str(out))
hl_h = hours(hl_files)
say(f"Hima labels from {done_pq}: {len(hima_text)} clips, {hl_h:.0f} h rebuilt in {len(hl_files)} files at {age_h():.2f} h")

# ---- 3. relabel the rest with the teacher
REL = Path("/tmp/relabel")
REL.mkdir(exist_ok=True)
until = T0 + RELABEL_H * 3600
procs = []
for g in range(max(n_gpu, 1)):
    files = yodas_files[g::max(n_gpu, 1)]
    logf = LOGS / f"relabel_{g}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    if TEACHER == "it":
        env["PYTHONPATH"] = str(ITPKGS)
    procs.append((subprocess.Popen([sys.executable, str(CODE / "relabel.py"), "--teacher", TEACHER, "--out", str(REL),
                                    "--until", str(until), "--skip", done_pq, *files],
                                   env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT, cwd=CODE), logf))
say(f"relabelling the rest of {len(yodas_files)} files with {TEACHER} on {len(procs)} GPU(s) until {RELABEL_H} h")
while any(p.poll() is None for p, _ in procs):
    time.sleep(60)
    for p, logf in procs:
        if p.poll() is None and (time.time() - logf.stat().st_mtime > 30 * 60 or time.time() > until + 1800):
            p.terminate(); p.wait(); say(f"{logf.name}: killed (stale log or past the relabel deadline)")
for p, logf in procs:
    say(f"{logf.name} exited with {p.returncode}")
shutil.copytree(REL, WORK / "relabel", ignore=shutil.ignore_patterns("*.arrow"), dirs_exist_ok=True)
new_files = sorted(str(f) for f in REL.glob("*.arrow"))
keep = [pa.ipc.open_file(pa.memory_map(f)).read_all() for f in new_files]
keep = pa.concat_tables([pa.table({"path": pc.struct_field(t["audio_filepath"], "path"), "text": t["text"],
                                   "yt_text": t["yt_text"], "duration": t["duration"]}) for t in keep]) if keep else None
new_h = 0.0
if keep is not None:
    pq.write_table(keep, WORK / f"{TEACHER}_relabel_text.parquet", compression="zstd")
    new_h = pc.sum(keep["duration"]).as_py() / 3600
say(f"{TEACHER} relabelled: {new_h:.0f} h kept in {len(new_files)} files at {age_h():.2f} h")
if not TRAIN:
    for f in glob.glob(str(LOGS / "relabel_*.log")):
        say(f"{Path(f).name}: " + open(f).read().strip().splitlines()[-1][:400])
    say("relabel-only night (TRAIN = False): done")
    sys.exit(0)
train_files = hl_files + new_files
all_h = hl_h + new_h
assert all_h > 50, f"only {all_h:.0f} h of labelled clips"

# one character set for every run (so the fine-tune reuses the pretrain head): chars seen >= 50 times
cnt = Counter()
for col in (hima["text"].to_pylist(), *([keep["text"].to_pylist()] if keep is not None else []),
            *(pq.read_table(f, columns=["text"]).column("text").to_pylist() for f in kb_pq)):
    for x in col:
        cnt.update(" ".join(x.split()))
UNITS = WORK / "units_char.json"
UNITS.write_text(json.dumps({"type": "char", "chars": sorted(c for c, n in cnt.items() if n >= 50)}, ensure_ascii=False))
say(f"units: {len(json.loads(UNITS.read_text())['chars'])} chars")


# ---- 4. training
def train_cmd(run, files, minutes, hours_of_data, extra=(), unfreeze=2):
    epochs = max(0.05, minutes / 60 * T4_HOURS_PER_HOUR[unfreeze] / hours_of_data)     # LR schedule spans the time budget
    return [sys.executable, str(CODE / "train.py"), "--run", run, "--units", str(UNITS), "--amp", "fp16",
            "--train", *files, "--eval", valid, "--eval-set", "lahaja", str(ARROW / "lahaja_*.arrow"),
            "--unfreeze", str(unfreeze), "--epochs", f"{epochs:.3f}", "--max-minutes", f"{minutes:.0f}",
            "--eval-every", "2000", "--workers", "3", *extra]


left_min = (DEADLINE_H - age_h()) * 60 - 15          # 15 min for start-up / final evals
pre_min = left_min - FINETUNE_MIN - 10
FT = ["--lr", "3e-4", "--enc-lr", "3e-5", "--warmup", "100"]
chains = {
    0: [("all_k2", train_cmd("all_k2", train_files, pre_min, all_h, ["--lr", "1e-3", "--warmup", "500"])),
        ("all_kb_k2", train_cmd("all_kb_k2", kb_files, FINETUNE_MIN, kb_h,
                                [*FT, "--init", str(CODE / "runs/all_k2/last.pt")]))],
    1: [("all_k4", train_cmd("all_k4", train_files, pre_min, all_h, ["--lr", "1e-3", "--warmup", "500"], unfreeze=4)),
        ("all_kb_k4", train_cmd("all_kb_k4", kb_files, FINETUNE_MIN, kb_h,
                                [*FT, "--init", str(CODE / "runs/all_k4/last.pt")], unfreeze=4))],
}
say(f"plan: each GPU {pre_min:.0f} min on {all_h:.0f} h (Hima {hl_h:.0f} + {TEACHER} {new_h:.0f}) then Kathbath "
    f"{FINETUNE_MIN} min (GPU0 top-2, GPU1 top-4)")


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


for run in ("all_k2", "all_k4"):                       # stage 1's final weights, without the optimizer state
    lp = CODE / f"runs/{run}/last.pt"
    if lp.exists():
        torch.save({"model": torch.load(lp, map_location="cpu")["model"]}, lp.with_name("final.pt"))
for f in glob.glob(str(CODE / "runs/*/last.pt")):    # keep outputs small: best.pt / final.pt (weights only) stay
    os.remove(f)
summarize()
say("done")
