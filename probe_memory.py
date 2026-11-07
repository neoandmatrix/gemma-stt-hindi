"""Does training fit on this GPU? One forward+backward+optimizer step per setting, peak VRAM reported.

usage: uv run probe_memory.py [--batch 8] [--seconds 6]
"""
import argparse
import time

import torch

from model import Config, GemmaCTC, load_audio_encoder

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--units", type=int, default=15000)
ap.add_argument("--k", type=int, nargs="+", default=[0, 2], help="unfreeze_top_k settings to try")
a = ap.parse_args()

encoder, text_emb = load_audio_encoder()
n_enc = sum(p.numel() for p in encoder.parameters())
per_layer = sum(p.numel() for p in encoder.layers[-1].parameters())
print(f"audio encoder {n_enc / 1e6:.0f}M params ({per_layer / 1e6:.1f}M per conformer layer), text emb {tuple(text_emb.shape)}")

for k in a.k:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    enc = encoder
    model = GemmaCTC(Config(n_units=a.units, unfreeze_top_k=k), enc, text_emb[2:a.units]).cuda().train()
    trainable = sum(p.numel() for p in model.trainable_parameters())
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    n = int(a.seconds * 16000)
    wav = torch.randn(a.batch, n, device="cuda") * 0.1
    lens = torch.full((a.batch,), n, device="cuda")
    tgt = torch.randint(2, a.units, (a.batch, int(a.seconds * 3)), device="cuda")
    tl = torch.full((a.batch,), tgt.shape[1], device="cuda")
    try:
        for step in range(3):
            t = time.perf_counter()
            logp, ol = model(wav, lens)
            loss = torch.nn.functional.ctc_loss(logp.transpose(0, 1), tgt, ol, tl, blank=0, zero_infinity=True)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
        print(f"[top-{k} trained] trainable {trainable / 1e6:.0f}M | batch {a.batch}x{a.seconds}s | "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | step {dt:.2f}s "
              f"({a.batch * a.seconds / dt:.0f}s audio/s) | frames {logp.shape[1]}")
    except torch.OutOfMemoryError:
        print(f"[top-{k} trained] OOM at batch {a.batch}x{a.seconds}s")
    del model, opt
