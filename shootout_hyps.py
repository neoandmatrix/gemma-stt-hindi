"""Transcribe the 05-model-shootout Hindi test sets with trained runs, in the shoot-out's results format, so
05-model-shootout/scripts/score.py scores them next to Indic-Transcribe, Qwen3-ASR, ...

usage: uv run shootout_hyps.py kaggle/night3/code/runs/yt_kb_k2 [more runs...]
-> ../05-model-shootout/results/stt/gemma-<run>/{mine,lahaja,monsoon_hi}.jsonl + _meta.json
"""
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from data import Clips, load_units, loader
from model import ENCODER_ID, Config, GemmaCTC, greedy_decode, load_audio_encoder

SETS = Path(__file__).parent / "kaggle/mytest_ds"           # mine / lahaja / monsoon_hi as Opus parquets with ids
OUT = Path(__file__).parent.parent / "05-model-shootout/results/stt"

tok = AutoTokenizer.from_pretrained(ENCODER_ID)
for run in map(Path, sys.argv[1:]):
    units = load_units(run / "units.json", tok)
    mc = json.loads((run / "config.json").read_text())["model"]
    mc.pop("grad_ckpt", None)
    cfg = Config(**{**mc, "amp": "fp16"})
    encoder, _ = load_audio_encoder(cfg.enc_dtype)    # fresh per run: GemmaCTC hooks into (and retypes) its layers
    model = GemmaCTC(cfg, encoder, None).cuda().eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cuda")["model"], strict=False)
    d = OUT / f"gemma-{run.name}"
    d.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    for path in sorted(SETS.glob("*.parquet")):
        clips = Clips([path], max_seconds=60)
        ids = pq.read_table(path, columns=["id"]).column("id").to_pylist()
        hyps, t = {}, time.time()
        with torch.no_grad():
            for b in loader(clips, units, 60, shuffle=False, workers=0):
                logp, ol = model(b["wav"].cuda(), b["wav_lens"].cuda())
                for i, seq in zip(b["idx"], greedy_decode(logp, ol)):
                    hyps[ids[clips.idx[i]]] = units.decode(seq)
        per = (time.time() - t) / max(len(hyps), 1)
        with open(d / f"{path.stem}.jsonl", "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({"id": i, "hyp": hyps.get(i, ""), "secs": round(per, 3)}, ensure_ascii=False) + "\n")
        print(f"{run.name} {path.stem}: {len(hyps)} clips, {per:.3f} s/clip", flush=True)
    (d / "_meta.json").write_text(json.dumps({"model": d.name, "errors": [],
                                              "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
    del model, encoder
    torch.cuda.empty_cache()
