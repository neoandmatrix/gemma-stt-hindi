"""CTC beam search with a word n-gram LM (pyctcdecode + KenLM) on top of a trained Gemma CTC run.

Greedy decoding takes the best unit per frame on its own. Beam search keeps the best partial transcripts and scores
each one as  log p_ctc(text | audio) + alpha * log p_lm(words) + beta * n_words,  so a spelling that's acoustically
close but isn't a likely word ("साथ बजे") loses to one the LM knows ("सात बजे").

1. Run the model once and save the per-frame log-probabilities (lm/logits/<run>/<set>.pt).
   dev sets (for tuning alpha/beta): Kathbath valid, and the Lahaja test shard 0 minus the 100 shoot-out clips.
   test sets: the shoot-out's mine / lahaja / monsoon_hi; they are only decoded with the chosen alpha/beta.
2. Grid-search alpha/beta on a dev sample, by mean WER over the two dev sets.
3. Decode the test sets -> ../05-model-shootout/results/stt/gemma-<run>-lm/*.jsonl, for scripts/score.py.

  python lm/decode_lm.py kaggle/night7/code/runs/n7_all_k2 --lm lm/hi3.arpa
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data import Clips, load_units, loader  # noqa: E402
from model import BLANK, ENCODER_ID, UNK, Config, GemmaCTC, greedy_decode, load_audio_encoder  # noqa: E402

TEST = ROOT / "kaggle/mytest_ds"
DEV = {"kathbath": ROOT / "data/kathbath/hindi/valid-00000-of-00002.parquet",
       "lahaja": ROOT / "data/lahaja/data/test-00000-of-00004.parquet"}
OUT = ROOT.parent / "05-model-shootout/results/stt"
P = re.compile(r"[^\w\sऀ-ॿ]")


def norm(t):
    return " ".join(P.sub(" ", (t or "").replace("।", " ").replace("॥", " ").lower()).split())


def dump(run, cache):
    """Per-frame log-probs for every dev and test clip, computed once per run."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_ID)
    units = load_units(run / "units.json", tok)
    mc = json.loads((run / "config.json").read_text())["model"]
    mc.pop("grad_ckpt", None)
    encoder, _ = load_audio_encoder(Config(**{**mc, "amp": "fp16"}).enc_dtype)
    model = GemmaCTC(Config(**{**mc, "amp": "fp16"}), encoder, None).cuda().eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cuda")["model"], strict=False)
    shoot_lahaja = {norm(t) for t in pq.read_table(TEST / "lahaja.parquet", columns=["text"]).column("text").to_pylist()}
    sets = {f"dev_{k}": v for k, v in DEV.items()} | {f"test_{p.stem}": p for p in sorted(TEST.glob("*.parquet"))}
    for name, path in sets.items():
        clips = Clips([path], max_seconds=60)
        cols = pq.read_table(path).column_names
        ids = pq.read_table(path, columns=["id"]).column("id").to_pylist() if "id" in cols else None
        out = {"logp": [], "ref": [], "id": [], "greedy": []}
        with torch.no_grad():
            for b in loader(clips, units, 60, shuffle=False, workers=0):
                logp, ol = model(b["wav"].cuda(), b["wav_lens"].cuda())
                for j, (i, seq) in enumerate(zip(b["idx"], greedy_decode(logp, ol))):
                    ref = clips.texts[i]
                    if name == "dev_lahaja" and norm(ref) in shoot_lahaja:
                        continue                               # keep the shoot-out clips out of tuning
                    out["logp"].append(logp[j, :int(ol[j])].float().cpu().half())
                    out["ref"].append(ref)
                    out["id"].append(ids[clips.idx[i]] if ids else str(clips.idx[i]))
                    out["greedy"].append(units.decode(seq))
        torch.save(out, cache / f"{name}.pt")
        print(f"  {name}: {len(out['ref'])} clips", flush=True)
    return units.chars


DEC = None


def _init(labels, lm, unigrams):
    global DEC
    from pyctcdecode import build_ctcdecoder
    DEC = build_ctcdecoder(labels, kenlm_model_path=lm, unigrams=unigrams)


def _decode(args):
    lp, alpha, beta, beam = args
    DEC.reset_params(alpha=alpha, beta=beta)              # cheap: two floats on the LM object
    lp = lp.astype(np.float32)
    lp[:, UNK] = -1e4                                    # the model never needs <unk>; keep it out of the beam
    return DEC.decode(lp, beam_width=beam)


class Decoder:
    """A pool of worker processes, each holding the LM once. Started with spawn, not fork: forking a process that
    has already used CUDA/torch threads left the workers deadlocked."""

    def __init__(self, labels, lm, unigrams, procs):
        self.pool = mp.get_context("spawn").Pool(procs, _init, (labels, lm, unigrams))

    def __call__(self, logps, alpha, beta, beam):
        return self.pool.map(_decode, [(x, alpha, beta, beam) for x in logps], chunksize=4)


def scores(refs, hyps):
    R, H = [norm(r) for r in refs], [norm(h) for h in hyps]
    keep = [i for i, r in enumerate(R) if r]
    R, H = [R[i] for i in keep], [H[i] for i in keep]
    return jiwer.cer(R, H), jiwer.wer(R, H)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--lm", default=str(ROOT / "lm/hi3.arpa"))
    ap.add_argument("--dev-n", type=int, default=400, help="clips per dev set for the grid search")
    ap.add_argument("--alphas", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--betas", default="0,1,2,3")
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tag", default="lm")
    a = ap.parse_args()
    run = Path(a.run)
    cache = ROOT / "lm/logits" / run.name
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "test_mine.pt").exists():
        dump(run, cache)
    chars = json.loads((run / "units.json").read_text())["chars"]
    labels = [""] * 2 + chars                            # unit 0 = blank, 1 = <unk>, 2.. = characters
    labels[BLANK], labels[UNK] = "", "⁇"
    unigrams = (ROOT / "lm/unigrams.txt").read_text(encoding="utf-8").split("\n")

    dev = {}
    for name in ("dev_kathbath", "dev_lahaja"):
        d = torch.load(cache / f"{name}.pt")
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(d["ref"]), min(a.dev_n, len(d["ref"])), replace=False))
        dev[name] = {k: [d[k][i] for i in pick] for k in d}
        dev[name]["logp"] = [x.numpy() for x in dev[name]["logp"]]
        c, w = scores(dev[name]["ref"], dev[name]["greedy"])
        print(f"{name} (n={len(pick)}) greedy: CER {c:.2%} WER {w:.2%}", flush=True)

    dec = Decoder(labels, a.lm, unigrams, a.procs)
    grid = []
    for alpha in map(float, a.alphas.split(",")):
        for beta in map(float, a.betas.split(",")):
            t = time.time()
            res = {n: scores(d["ref"], dec(d["logp"], alpha, beta, a.beam)) for n, d in dev.items()}
            mw = np.mean([w for _, w in res.values()])
            grid.append((mw, alpha, beta, res))
            print(f"alpha {alpha:.1f} beta {beta:.1f}: " + "  ".join(
                f"{n[4:]} CER {c:.2%} WER {w:.2%}" for n, (c, w) in res.items()) +
                f"  | mean WER {mw:.2%} ({time.time() - t:.0f} s)", flush=True)
    mw, alpha, beta, _ = min(grid, key=lambda g: g[0])
    print(f"best on dev: alpha {alpha} beta {beta} (mean WER {mw:.2%})", flush=True)

    d_out = OUT / f"gemma-{run.name}-{a.tag}"
    d_out.mkdir(parents=True, exist_ok=True)
    for f in sorted(cache.glob("test_*.pt")):
        d = torch.load(f)
        t = time.time()
        hyps = dec([x.numpy() for x in d["logp"]], alpha, beta, a.beam)
        per = (time.time() - t) / max(len(hyps), 1) * a.procs
        with open(d_out / f"{f.stem[5:]}.jsonl", "w", encoding="utf-8") as fo:
            for i, h in zip(d["id"], hyps):
                fo.write(json.dumps({"id": i, "hyp": h, "secs": round(per, 3)}, ensure_ascii=False) + "\n")
        g, l = scores(d["ref"], d["greedy"]), scores(d["ref"], hyps)
        print(f"{f.stem}: greedy CER {g[0]:.2%} WER {g[1]:.2%} -> LM CER {l[0]:.2%} WER {l[1]:.2%}", flush=True)
    (d_out / "_meta.json").write_text(json.dumps({"model": d_out.name, "errors": [], "lm": Path(a.lm).name,
                                                  "alpha": alpha, "beta": beta, "beam": a.beam}))
