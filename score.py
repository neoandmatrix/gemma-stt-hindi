"""Score one trained run on a parquet test set, printing every clip (ref vs hyp) and CER per sentence kind.

usage: uv run score.py runs/s1_4h_frozen_char data/mytest/mytest.parquet [--amp bf16]
"""
import argparse
import csv
import json
from pathlib import Path

import jiwer
import torch
from transformers import AutoTokenizer

from data import Clips, load_units, loader, normalize
from model import ENCODER_ID, Config, GemmaCTC, greedy_decode, load_audio_encoder

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("parquet")
ap.add_argument("--amp", default="bf16")
a = ap.parse_args()

run = Path(a.run)
tok = AutoTokenizer.from_pretrained(ENCODER_ID)
units = load_units(run / "units.json", tok)
mc = json.loads((run / "config.json").read_text())["model"]
mc.pop("grad_ckpt", None)
cfg = Config(**{**mc, "amp": a.amp})
encoder, _ = load_audio_encoder(cfg.enc_dtype)
model = GemmaCTC(cfg, encoder).cuda().eval()
model.load_state_dict(torch.load(run / "best.pt", map_location="cuda")["model"], strict=False)

clips = Clips([a.parquet], max_seconds=30)
kinds = {}
tsv = Path(a.parquet).with_name("sentences.tsv")
if tsv.exists():
    kinds = {r["text"]: r["kind"] for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t")}

refs, hyps = [None] * len(clips), [None] * len(clips)
with torch.no_grad():
    for b in loader(clips, units, 60, shuffle=False, workers=0):
        logp, ol = model(b["wav"].cuda(), b["wav_lens"].cuda())
        for i, seq in zip(b["idx"], greedy_decode(logp, ol)):
            refs[i], hyps[i] = normalize(clips.texts[i]), normalize(units.decode(seq))

by_kind = {}
for i, (r, h) in enumerate(zip(refs, hyps)):
    k = kinds.get(clips.texts[i], "-")
    by_kind.setdefault(k, ([], []))[0].append(r); by_kind[k][1].append(h)
    print(f"[{i + 1:02d}] {k:<8} CER {jiwer.cer(r, h):5.1%}\n   ref: {r}\n   hyp: {h}")
print(f"\nALL: CER {jiwer.cer(refs, hyps):.1%}  WER {jiwer.wer(refs, hyps):.1%}  ({len(refs)} clips)")
for k, (r, h) in by_kind.items():
    print(f"  {k:<8} CER {jiwer.cer(r, h):5.1%}  WER {jiwer.wer(r, h):5.1%}  ({len(r)} clips)")
