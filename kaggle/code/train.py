"""Train the Hindi CTC reader (+ optionally the top encoder layers) on parquet shards; evaluate CER/WER as it goes.

examples:
  # step 0, overfit check: 50 clips, evaluated on the same 50
  uv run train.py --run overfit --train data/kathbath/hindi/train-00000-of-00032.parquet --max-train-clips 50 \
      --eval-on-train --epochs 60 --eval-every 50 --warmup 20
  # step 1: one shard (~4 h), frozen encoder
  uv run train.py --run s1_4h_frozen --train data/kathbath/hindi/train-00000-of-00032.parquet \
      --eval data/kathbath/hindi/valid-00000-of-00002.parquet --epochs 8

Outputs in runs/<run>/: log.jsonl (one line per eval), units.json, config.json, last.pt / best.pt (trainable weights).
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import jiwer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from data import CharUnits, Clips, Units, load_units, loader, normalize
from model import ENCODER_ID, Config, GemmaCTC, greedy_decode, load_audio_encoder

HERE = Path(__file__).parent


def expand(patterns):
    return sorted(p for pat in patterns for p in glob.glob(pat))


@torch.no_grad()
def evaluate(model, dl, units, clips, show=3):
    model.eval()
    refs, hyps = [], []
    for b in dl:
        logp, ol = model(b["wav"].cuda(non_blocking=True), b["wav_lens"].cuda())
        for i, seq in zip(b["idx"], greedy_decode(logp, ol)):
            refs.append(normalize(clips.texts[i])); hyps.append(normalize(units.decode(seq)))
    model.train()
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return dict(cer=jiwer.cer(r, h), wer=jiwer.wer(r, h), n=len(r), examples=list(zip(r, h))[:show])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="parquet shards (globs ok)")
    ap.add_argument("--eval", nargs="*", default=[], help="eval parquet shards (scored as 'valid'; picks best.pt)")
    ap.add_argument("--eval-set", nargs=2, action="append", default=[], metavar=("NAME", "GLOB"),
                    help="extra test set scored separately, e.g. --eval-set lahaja 'data/lahaja/*.parquet'")
    ap.add_argument("--eval-clips", type=int, default=1000, help="per eval set, spread evenly across its files")
    ap.add_argument("--eval-on-train", action="store_true", help="evaluate on the training clips (overfit check)")
    ap.add_argument("--max-train-clips", type=int)
    ap.add_argument("--units", help="units.json to reuse (default: build from the training text)")
    ap.add_argument("--unit-type", choices=["token", "char"], default="token",
                    help="CTC outputs: Gemma tokens (~words for Hindi) or characters")
    ap.add_argument("--unfreeze", type=int, default=0, help="top encoder layers to train")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-seconds", type=float, default=48)
    ap.add_argument("--lr", type=float, default=5e-4, help="reader/head")
    ap.add_argument("--enc-lr", type=float, default=5e-5, help="unfrozen encoder layers")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp16_enc32"], default="bf16",
                    help="bf16 needs Ampere+ (3050 ok); Kaggle T4/P100: fp16 or fp16_enc32")
    ap.add_argument("--init", help="checkpoint (best.pt/last.pt) to start from (weights only)")
    ap.add_argument("--eval-only", action="store_true", help="just evaluate --init on --eval and exit")
    ap.add_argument("--max-steps", type=int, help="stop after this many steps (the LR schedule still assumes --epochs)")
    ap.add_argument("--max-minutes", type=float, help="stop (with a final eval) after this much training wall time")
    ap.add_argument("--gate-hours", type=float, help="after this many hours of audio seen...")
    ap.add_argument("--gate-cer", type=float, help="...stop if eval CER is still above this")
    a = ap.parse_args()

    out = HERE / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ENCODER_ID)
    train = Clips(expand(a.train), max_clips=a.max_train_clips)
    evals = []                                          # (name, Clips); the first one picks best.pt and gates
    if a.eval_on_train:
        evals.append(("train", train))
    elif a.eval:
        evals.append(("valid", Clips(expand(a.eval), max_clips=a.eval_clips, max_seconds=20, spread=True)))
    for name, pat in a.eval_set:
        evals.append((name, Clips(expand([pat]), max_clips=a.eval_clips, max_seconds=20, spread=True)))
    evalc = evals[0][1] if evals else None
    units = load_units(a.units, tok) if a.units else \
        (CharUnits if a.unit_type == "char" else Units).build(train.texts, tok)
    units.save(out / "units.json")
    print(f"train {len(train)} clips / {train.hours():.2f} h | eval {', '.join(f'{n} {len(c)}' for n, c in evals)} | "
          f"units {len(units)}")

    cfg = Config(n_units=len(units), unfreeze_top_k=a.unfreeze, amp=a.amp)
    encoder, text_emb = load_audio_encoder(cfg.enc_dtype)
    (out / "config.json").write_text(json.dumps(dict(model=cfg.to_dict(), args=vars(a)), indent=2))
    model = GemmaCTC(cfg, encoder, text_emb[units.token_ids]).cuda().train()
    del text_emb
    enc_params = [p for layer in model.trainable_layers for p in layer.parameters()]
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.trainable_parameters() if id(p) not in enc_ids]
    groups = [dict(params=head_params, lr=a.lr)] + ([dict(params=enc_params, lr=a.enc_lr)] if enc_params else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    steps_per_epoch = len(loader(train, units, a.batch_seconds, shuffle=False, workers=0))
    total = math.ceil(a.epochs * steps_per_epoch)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        f = step / max(1, a.warmup) if step < a.warmup else \
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, total - a.warmup)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    # fp16 gradients can underflow: scale the loss (bf16 has fp32's range and doesn't need it)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.reader_dtype == torch.float16)
    step, seen_s, best, n_nan = 0, 0.0, None, 0
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cuda")["model"], strict=False)
        print(f"initialized from {a.init}")
    edls = [(n, c, loader(c, units, a.batch_seconds * 2, shuffle=False, workers=2)) for n, c in evals]

    def run_evals(tag):
        """Score every eval set -> {name: result}; prints and returns them."""
        res = {}
        for name, c, dl in edls:
            ev = res[name] = evaluate(model, dl, units, c)
            print(f"== {tag} [{name}]: CER {ev['cer']:.3f} WER {ev['wer']:.3f} ({ev['n']} clips)", flush=True)
            for r, h in ev["examples"][:2]:
                print(f"   ref: {r}\n   hyp: {h}")
        return res

    if a.eval_only:
        run_evals(f"EVAL-ONLY amp={a.amp}")
        return
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"])
        step, seen_s, best = ck["step"], ck["seen_s"], ck.get("best")
        print(f"resumed at step {step}")

    def save(name, with_opt=True):
        sd = {k: v for k, v in model.state_dict().items() if k in trainable_names}
        ck = dict(model=sd, step=step, seen_s=seen_s, best=best)
        if with_opt:                                   # only needed to resume (2x the weights' size)
            ck["opt"] = opt.state_dict()
        torch.save(ck, out / name)

    log = open(out / "log.jsonl", "a")
    print(f"{total} steps ({steps_per_epoch}/epoch), trainable {sum(p.numel() for p in model.trainable_parameters()) / 1e6:.0f}M")
    t0, win_s, win_loss, win_n = time.perf_counter(), 0.0, 0.0, 0
    t_start = time.perf_counter()
    done = False
    while not done:
        dl = loader(train, units, a.batch_seconds, shuffle=True, seed=step, workers=a.workers)
        for b in dl:
            set_lr(step)
            logp, ol = model(b["wav"].cuda(non_blocking=True), b["wav_lens"].cuda())
            loss = F.ctc_loss(logp.transpose(0, 1), b["tgt"].cuda(), ol, b["tgt_lens"].cuda(),
                              blank=0, zero_infinity=True)
            if not torch.isfinite(loss):
                n_nan += 1
                print(f"WARNING: non-finite loss at step {step} ({n_nan} so far)", flush=True)
                opt.zero_grad(set_to_none=True); step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            secs = b["wav_lens"].sum().item() / 16000
            seen_s += secs; win_s += secs; win_loss += loss.item(); win_n += 1

            if step % 50 == 0:
                dt = time.perf_counter() - t0
                print(f"step {step}/{total} | loss {win_loss / win_n:.3f} | {seen_s / 3600:.2f} h seen | "
                      f"{win_s / dt:.0f} s audio/s | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                t0, win_s, win_loss, win_n = time.perf_counter(), 0.0, 0.0, 0

            stop_early = (a.max_steps is not None and step >= a.max_steps) or \
                (a.max_minutes is not None and time.perf_counter() - t_start > a.max_minutes * 60)
            if edls and (step % a.eval_every == 0 or step >= total or stop_early):
                res = run_evals(f"EVAL step {step}")
                ev = res[evals[0][0]]
                rec = dict(step=step, hours_seen=round(seen_s / 3600, 3), cer=round(ev["cer"], 4),
                           wer=round(ev["wer"], 4), n_eval=ev["n"], time=time.strftime("%H:%M:%S"))
                for name, r in res.items():
                    rec[f"cer_{name}"], rec[f"wer_{name}"] = round(r["cer"], 4), round(r["wer"], 4)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
                if best is None or ev["cer"] < best:
                    best = ev["cer"]; save("best.pt", with_opt=False)
                save("last.pt")
                if a.gate_hours and a.gate_cer and seen_s / 3600 >= a.gate_hours and ev["cer"] > a.gate_cer:
                    print(f"GATE: CER {ev['cer']:.3f} > {a.gate_cer} after {seen_s / 3600:.1f} h seen, stopping")
                    log.write(json.dumps(dict(gate="stopped", step=step, cer=ev["cer"])) + "\n")
                    done = True; break
            if step >= total or stop_early:
                done = True; break
    save("last.pt")
    print(f"done: {step} steps, best CER {best}, non-finite losses {n_nan}")


if __name__ == "__main__":
    main()
