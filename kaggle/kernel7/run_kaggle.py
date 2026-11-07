"""Kaggle runner for 04-gemma-stt-hindi, night 7: train on every teacher-labelled YODAS clip (~1,750 h).
Runs on Kaggle (2x T4); the code dataset and the earlier notebooks it reads are attached as sources.
Night 5 (ladder v17): Hima labelled 780 h; hl_kb_k2 got Kathbath 3.7% / Lahaja 9.6% (YouTube captions: 4.2% / 11.6%).
Night 6 (gemma-stt-hindi-n6 v3): Indic-Transcribe labelled the other 970 h, no training (it_relabel_text.parquet).

1. Download Kathbath 50 h + Lahaja + EmbeddingGemma 2; copy every shard to /tmp as Arrow.
2. Rebuild both label sets: YODAS audio + text from relabel_text.parquet (Hima) and it_relabel_text.parquet (IT).
3. Train on both (mixed teachers):
   GPU 0  all_k2 -> all_kb_k2: top-2 encoder layers, then the Kathbath 50 h fine-tune
   GPU 1  all_k4 -> all_kb_k4: top-4, then the same fine-tune
"""
DEADLINE_H = 11.4          # all training ends by this session age; outputs saved before the 12 h limit
FINETUNE_MIN = 45          # stage 2 (Kathbath fine-tune): ~2 epochs of 50 h at the measured speed
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
    def durs(f):
        t = (pa.ipc.open_file(pa.memory_map(str(f))).read_all() if str(f).endswith(".arrow")
             else pq.read_table(f, columns=["duration"]))
        return sum(t.column("duration").to_pylist())
    return sum(durs(f) for f in files) / 3600


yodas_h, kb_h = hours(yodas_pq), hours(kb_pq)
say(f"data ready at {age_h():.2f} h: YODAS {yodas_h:.0f} h in {len(yodas_files)} files, Kathbath {kb_h:.0f} h, "
    f"{shutil.disk_usage('/tmp').free / 1e9:.0f} GB disk left")
assert yodas_h > 50, f"only {yodas_h:.0f} h of YODAS clips found: is the prep notebook attached?"

# ---- 2. both label sets: YODAS audio + teacher text
import pyarrow.compute as pc  # noqa: E402


def find(name):
    hits = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True)]
    assert hits, f"{name} not found: is its notebook attached ?"
    return hits[0]


def rebuild(label_pq, tag):
    lab = pq.read_table(label_pq, columns=["path", "text"])
    text = dict(zip(lab["path"].to_pylist(), lab["text"].to_pylist()))
    paths = lab["path"].combine_chunks()
    files = []
    for f in yodas_files:
        t = pa.ipc.open_file(pa.memory_map(f)).read_all()
        t = t.filter(pc.is_in(pc.struct_field(t["audio_filepath"], "path"), value_set=paths))
        if t.num_rows == 0:
            continue
        texts = [text[x] for x in pc.struct_field(t["audio_filepath"], "path").to_pylist()]
        t = pa.table({"text": pa.array(texts, pa.string()), "duration": t["duration"],
                      "audio_filepath": t["audio_filepath"]})
        out = ARROW / f"{tag}_{Path(f).name}"
        with pa.ipc.new_file(str(out), t.schema) as w:
            w.write_table(t)
        files.append(str(out))
    h = hours(files)
    say(f"{tag} labels from {label_pq}: {len(text)} clips, {h:.0f} h rebuilt in {len(files)} files at {age_h():.2f} h")
    return lab, files, h


hima, hl_files, hl_h = rebuild(find("relabel_text.parquet"), "hima")
keep, new_files, new_h = rebuild(find("it_relabel_text.parquet"), "it")
for f in yodas_files:                                 # the unlabelled originals are no longer needed
    os.remove(f)
TEACHER = "it"
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
