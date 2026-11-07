"""Relabel YODAS clips with a strong Hindi ASR model (the "teacher"): replace the YouTube auto-captions with its
transcripts, keeping the audio.
  --teacher hima: Prahaladha/hima-asr (0.46B), night 5
  --teacher it:   bodhan-ai/indic-transcribe-core (1.2B, gated; needs transformers 4.5x and the repo's own code), night 6

usage: python relabel.py --teacher it --out DIR --until UNIX_TIME [--skip done.parquet] file1.arrow ...
-> DIR/<stem>_pNNN.arrow with columns text (teacher), yt_text (the original caption), duration, audio_filepath,
   written every --part clips so a time-out keeps what's done; DIR/stats_<gpu>.json.
--skip: a parquet with a "path" column (night 5's relabel_text.parquet): those clips are left out.
Clips whose teacher text isn't Devanagari-only after removing punctuation (digits, Latin letters) or has an
implausible characters-per-second rate are dropped, with the same rules as yodas_prep.py.
"""
import argparse
import copy
import io
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import soundfile as sf
import torch

from yodas_prep import DEVANAGARI_ONLY, MAX_CPS, MIN_CPS

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--teacher", choices=["hima", "it"], default="hima")
p.add_argument("--out", required=True)
p.add_argument("--until", type=float, required=True, help="stop at this unix time")
p.add_argument("--skip", help="parquet with a 'path' column of clips to leave out")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--part", type=int, default=4000)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def wav(row):
    x, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32")
    return x


def cer(x, y):
    import jiwer
    return jiwer.cer(x, y) if x else 1.0


def clean(t):
    """Teacher text -> our training text: punctuation to spaces, NFC, single spaces."""
    t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in unicodedata.normalize("NFC", t or ""))
    return unicodedata.normalize("NFC", " ".join(t.split()))


# ---- teachers: run(wavs, variant) -> texts; variants[0] is the reference, the rest are faster candidates
if a.teacher == "hima":
    from transformers import AutoModel
    model = AutoModel.from_pretrained("Prahaladha/hima-asr", trust_remote_code=True, dtype=torch.float32).cuda().eval()
    _autocast = torch.autocast

    def run(wavs, variant):
        """'fp16': fp16 autocast (fast on a T4, which has no bf16); 'default': the model's own bf16 autocast."""
        if variant == "fp16":
            torch.autocast = lambda device_type, dtype=None, enabled=True, **k: _autocast(
                device_type, dtype=torch.float16, enabled=enabled)
        try:
            return model.transcribe(wavs, language="hi", batch_size=len(wavs))
        finally:
            torch.autocast = _autocast
    variants = ["default", "fp16"]
else:
    from huggingface_hub import snapshot_download
    mdir = snapshot_download("bodhan-ai/indic-transcribe-core", ignore_patterns=["nemo/*", "*.nemo", "banner.png"])
    sys.path.insert(0, mdir)
    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained(mdir, device="cuda", dtype=torch.float32)
    models = {"fp32": asr.model}
    prompt = asr.tokenizer.encode_prompt("hi", itn=False, romanized=False)

    def run(wavs, variant):
        """'ref': the repo's own one-clip-at-a-time call (fp32); 'fp32' / 'fp16': our batched path."""
        if variant == "ref":
            return [asr.transcribe(w, lang="hi") for w in wavs]
        m = models[variant]
        lens = torch.tensor([len(w) for w in wavs])
        batch = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            batch[i, :len(w)] = torch.from_numpy(w)
        with torch.inference_mode():
            feats, feat_lens = asr.fe(batch.cuda(), lens.cuda())
            mask = (torch.arange(feats.size(2), device="cuda")[None, :] < feat_lens[:, None]).long()
            ids = torch.tensor([prompt] * len(wavs), device="cuda")
            outs = m.generate(input_features=feats.to(next(m.parameters()).dtype), attention_mask=mask,
                              decoder_input_ids=ids, max_new_tokens=256)
        return [asr.tokenizer.decode(asr.tokenizer.strip_prompt_and_trim(o.tolist(), prompt)) for o in outs]
    variants = ["ref", "fp32", "fp16"]

tables = [pa.ipc.open_file(pa.memory_map(f)).read_all() for f in a.files]
if a.skip:
    import pyarrow.parquet as pq
    done = pa.array(pq.read_table(a.skip, columns=["path"]).column("path").to_pylist())
    tables = [t.filter(pc.invert(pc.is_in(pc.struct_field(t["audio_filepath"], "path"), value_set=done)))
              for t in tables]
print(f"{sum(t.num_rows for t in tables)} clips to relabel "
      f"({sum(pc.sum(t['duration']).as_py() or 0 for t in tables) / 3600:.0f} h)", flush=True)

# fastest variant that agrees with the reference: the same 64 clips through each
probe_n = 64 if a.teacher == "hima" else 32
first = next(t for t in tables if t.num_rows)              # a file night 5 finished is empty after --skip
probe = [wav(r) for r in first.column("audio_filepath").slice(0, probe_n).to_pylist()]
probe.sort(key=len)
res = {}
for v in variants:
    if v == "fp16" and a.teacher == "it":
        models["fp16"] = copy.deepcopy(models["fp32"]).half()
    torch.cuda.synchronize(); t = time.time()
    texts = [x for i in range(0, probe_n, a.batch) for x in run(probe[i:i + a.batch], v)]
    torch.cuda.synchronize()
    res[v] = (round(time.time() - t, 1), " ".join(clean(x) for x in texts))
ref = variants[0]
gaps = {v: round(cer(res[ref][1], res[v][1]), 4) for v in variants[1:]}
ok = [v for v in variants[1:] if gaps[v] < 0.03]
USE = min(ok, key=lambda v: res[v][0]) if ok else ref
if a.teacher == "it":                                   # keep only the chosen copy on the GPU
    models = {USE: models[USE]} if USE != "ref" else {}
    torch.cuda.empty_cache()
stats = {"gpu": tag, "teacher": a.teacher, "probe_s": {v: res[v][0] for v in variants}, "probe_cer_vs_ref": gaps,
         "variant": USE, "clips_in": 0, "clips_kept": 0, "h_in": 0.0, "h_kept": 0.0, "drop_script": 0,
         "drop_cps": 0, "drop_empty": 0, "files_done": 0, "yt_vs_teacher_cer_sum": 0.0}
print(json.dumps(stats), flush=True)
print("probe sample:", res[USE][1][:300], flush=True)


def save_stats():
    s = dict(stats, yt_vs_teacher_cer=round(stats["yt_vs_teacher_cer_sum"] / max(stats["clips_in"], 1), 4))
    (out / f"stats_{tag}.json").write_text(json.dumps(s, indent=1))


t0, timed_out = time.time(), False
for f, tab in zip(a.files, tables):
    n = tab.num_rows
    if n == 0:
        stats["files_done"] += 1
        continue
    # random blocks of 2048 clips, sorted by length inside each block: similar lengths per batch (less padding) and
    # a time-out still leaves a random sample of the file, not only its shortest clips
    durs_all = tab.column("duration").to_numpy()
    perm = np.random.default_rng(0).permutation(n)
    order = np.concatenate([b[np.argsort(durs_all[b])] for b in np.array_split(perm, max(1, n // 2048))])
    rows = {"text": [], "yt_text": [], "duration": [], "audio_filepath": []}
    part = 0

    def flush():
        global part
        if rows["text"]:
            t = pa.table(rows, schema=pa.schema([("text", pa.string()), ("yt_text", pa.string()),
                                                 ("duration", tab.schema.field("duration").type),
                                                 ("audio_filepath", tab.schema.field("audio_filepath").type)]))
            with pa.ipc.new_file(str(out / f"{a.teacher}_{Path(f).stem}_p{part:03d}.arrow"), t.schema) as w:
                w.write_table(t)
            part += 1
            for v in rows.values():
                v.clear()

    for i in range(0, n, a.batch):
        if time.time() > a.until:
            timed_out = True
            break
        idx = order[i:i + a.batch]
        sub = tab.take(pa.array(idx))
        audio = sub.column("audio_filepath").to_pylist()
        durs = sub.column("duration").to_pylist()
        yts = sub.column("text").to_pylist()
        hyps = run([wav(r) for r in audio], USE)
        for au, d, yt, h in zip(audio, durs, yts, hyps):
            h = clean(h)
            stats["clips_in"] += 1; stats["h_in"] += d / 3600
            stats["yt_vs_teacher_cer_sum"] += min(cer(yt, h), 1.0)
            if not h:
                stats["drop_empty"] += 1; continue
            if not DEVANAGARI_ONLY.fullmatch(h):
                stats["drop_script"] += 1; continue
            if not MIN_CPS <= len(h.replace(" ", "")) / d <= MAX_CPS:
                stats["drop_cps"] += 1; continue
            rows["text"].append(h); rows["yt_text"].append(yt); rows["duration"].append(d)
            rows["audio_filepath"].append(au)
            stats["clips_kept"] += 1; stats["h_kept"] += d / 3600
        if len(rows["text"]) >= a.part:
            flush()
        if (i // a.batch) % 50 == 0:
            el = time.time() - t0
            print(f"{Path(f).stem} {i}/{n} | {stats['h_in']:.1f} h in, {stats['h_kept']:.1f} h kept | "
                  f"{stats['h_in'] * 3600 / max(el, 1):.0f} s audio/s | yt-vs-teacher CER "
                  f"{stats['yt_vs_teacher_cer_sum'] / max(stats['clips_in'], 1):.3f}", flush=True)
            save_stats()
    flush()
    if timed_out:
        break
    stats["files_done"] += 1
stats["timed_out"] = timed_out
save_stats()
print("done", json.dumps(stats), flush=True)
