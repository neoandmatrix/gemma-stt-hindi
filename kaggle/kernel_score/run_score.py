"""Transcribe the shoot-out's Hindi test sets (mine, lahaja, monsoon_hi) with every best.pt in the attached
gemma-stt-hindi-ladder output, so the checkpoints never have to be downloaded (~376 MB each).
Output: /kaggle/working/stt/gemma-<run>/<set>.jsonl {"id", "hyp", "secs"} -- the 05-model-shootout results format,
scored locally with 05-model-shootout/scripts/score.py next to Indic-Transcribe, Qwen3-ASR, ...
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.21.0", "tokenizers==0.23.3",
                "huggingface_hub==2.3.0", "soundfile"], check=True)
from kaggle_secrets import UserSecretsClient  # noqa: E402
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

CODE = Path(glob.glob("/kaggle/input/**/train.py", recursive=True)[0]).parent
sys.path.insert(0, str(CODE))
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from data import Clips, load_units, loader, normalize  # noqa: E402
from model import ENCODER_ID, Config, GemmaCTC, greedy_decode, load_audio_encoder  # noqa: E402

SETS = {Path(p).stem: p for p in glob.glob("/kaggle/input/**/monsoon_hi.parquet", recursive=True)
        + glob.glob("/kaggle/input/**/lahaja.parquet", recursive=True)
        + glob.glob("/kaggle/input/**/mine.parquet", recursive=True)}
RUNS = sorted(Path(p).parent for p in glob.glob("/kaggle/input/**/runs/*/best.pt", recursive=True))
print("sets", SETS, "\nruns", RUNS, flush=True)
OUT = Path("/kaggle/working/stt")
tok = AutoTokenizer.from_pretrained(ENCODER_ID)

for run in RUNS:
    units = load_units(run / "units.json", tok)
    mc = json.loads((run / "config.json").read_text())["model"]
    mc.pop("grad_ckpt", None)
    cfg = Config(**{**mc, "amp": "fp16"})
    encoder, _ = load_audio_encoder(cfg.enc_dtype)   # fresh per run: GemmaCTC hooks into (and retypes) its layers
    model = GemmaCTC(cfg, encoder, None).cuda().eval()        # trained top-2 layers come from best.pt
    model.load_state_dict(torch.load(run / "best.pt", map_location="cuda")["model"], strict=False)
    d = OUT / f"gemma-{run.name}"
    d.mkdir(parents=True, exist_ok=True)
    for name, path in SETS.items():
        clips = Clips([path], max_seconds=60)
        ids = pq.read_table(path, columns=["id"]).column("id").to_pylist()
        hyps = {}
        t = time.time()
        with torch.no_grad():
            for b in loader(clips, units, 60, shuffle=False, workers=0):
                logp, ol = model(b["wav"].cuda(), b["wav_lens"].cuda())
                for i, seq in zip(b["idx"], greedy_decode(logp, ol)):
                    hyps[ids[clips.idx[i]]] = units.decode(seq)
        per = (time.time() - t) / max(len(hyps), 1)
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "hyp": hyps.get(i, ""), "secs": round(per, 3)}, ensure_ascii=False) + "\n")
        print(f"{run.name} {name}: {len(hyps)} clips", flush=True)
    (d / "_meta.json").write_text(json.dumps({"model": d.name, "errors": [],
                                              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
    del model, encoder
    torch.cuda.empty_cache()
print("done")
