"""Build a word n-gram language model (ARPA file) from the teacher labels, for CTC beam search with KenLM.

KenLM's own builder (lmplz) needs Boost, which this laptop doesn't have, so this is the same method in numpy:
interpolated modified Kneser-Ney (Chen & Goodman 1998), the default of lmplz.

  p(w | h) = (a(h w) - D(a(h w))) / a(h)  +  gamma(h) * p(w | shorter h)
  - a() is the raw count for the longest order and for n-grams that start with <s>; for the lower orders it is
    the "continuation count": in how many different contexts the n-gram appears. ("Francisco" is frequent but only
    after "San", so as a lower-order fallback it should be rare.)
  - D(1), D(2), D(3+) are discounts from the count-of-counts; gamma(h) is the mass they free, given to the shorter
    context. In the ARPA file, gamma(h) is the back-off weight of h.
  - Pruning (--prune 1): n-grams of order >= 3 seen only once are left out of the file, after all probabilities
    are computed from the full counts (they then fall back to gamma * the shorter context, like any unseen n-gram).

Text: teacher transcripts of the training audio (Hima 303k + Indic-Transcribe 387k clips), cleaned like the CTC
targets; sentences with a character the CTC model can't output are dropped. No test-set text is used.

  python lm/build_arpa.py --order 3 --out lm/hi3.arpa
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--order", type=int, default=3)
ap.add_argument("--out", default=str(HERE / "hi3.arpa"))
ap.add_argument("--units", default=str(HERE.parent / "kaggle/night7/code/runs/all_k2/units.json"))
ap.add_argument("--prune", type=int, default=1, help="leave out n-grams of order >= 3 with raw count <= this")
args = ap.parse_args()
N = args.order
t0 = time.time()

# ---- text -> word ids
chars = set(json.loads(Path(args.units).read_text())["chars"])
PUNCT = re.compile(r"[^\w\sऀ-ॿ]")
sents, dropped = [], 0
for f in ("relabel_text.parquet", "it_relabel_text.parquet"):
    for t in pq.read_table(HERE / "text" / f, columns=["text"]).column("text").to_pylist():
        t = " ".join(PUNCT.sub(" ", (t or "").replace("।", " ").replace("॥", " ")).split())
        if t and set(t) <= chars:
            sents.append(t.split())
        else:
            dropped += 1
vocab = {"<s>": 0, "</s>": 1}
for s in sents:
    for w in s:
        vocab.setdefault(w, len(vocab))
words = list(vocab)
V = len(vocab)
BOS, EOS = 0, 1
print(f"{len(sents)} sentences ({dropped} dropped), {sum(map(len, sents))} words, vocab {V - 2}", flush=True)
assert V ** N < 2 ** 62, "vocab too large for int64 keys"

# all sentences in one id stream: <s> w1 .. wk </s> <s> ...; an n-gram is valid if <s> appears only first and
# </s> only last, which also rules out every n-gram that crosses a sentence boundary
stream = []
for s in sents:
    stream += [BOS] + [vocab[w] for w in s] + [EOS]
stream = np.array(stream, dtype=np.int64)
del sents


def key(m):
    k = np.zeros(len(m), np.int64)
    for i in range(m.shape[1]):
        k = k * V + m[:, i]
    return k


def ngrams(n):
    L = len(stream) - n + 1
    m = np.stack([stream[i:i + L] for i in range(n)], 1)
    ok = np.ones(L, bool)
    for i in range(1, n):
        ok &= m[:, i] != BOS
    for i in range(n - 1):
        ok &= m[:, i] != EOS
    if n == 1:
        ok &= m[:, 0] != BOS                       # <s> is never predicted; written separately
    m = m[ok]
    k, c = np.unique(key(m), return_counts=True)   # sorted keys = lexicographic order of the rows
    rows = np.zeros((len(k), n), np.int64)
    kk = k.copy()
    for i in range(n - 1, -1, -1):
        rows[:, i] = kk % V; kk //= V
    return rows, k, c


raw = {n: ngrams(n) for n in range(1, N + 1)}
for n in range(1, N + 1):
    print(f"  {n}-grams: {len(raw[n][1])}", flush=True)

# adjusted counts
adj = {N: raw[N][2]}
for n in range(N - 1, 0, -1):
    grams, k, c = raw[n]
    suf = key(raw[n + 1][0][:, 1:])                # each distinct (n+1)-gram adds 1 to its suffix's count
    uk, cc = np.unique(suf, return_counts=True)
    pos = np.minimum(np.searchsorted(uk, k), len(uk) - 1)
    a = np.where(uk[pos] == k, cc[pos], 0)
    a = np.where(grams[:, 0] == BOS, c, a)
    adj[n] = np.maximum(a, 1)


def discounts(a):
    n1, n2, n3, n4 = (np.sum(a == i) for i in (1, 2, 3, 4))
    Y = n1 / (n1 + 2 * n2)
    return np.clip([0.0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3], 0, None)


prob, bow = {}, {}
for n in range(1, N + 1):
    grams, k, _ = raw[n]
    a = adj[n]
    D = discounts(a)
    d = D[np.minimum(a, 3)]
    if n == 1:
        tot = a.sum()
        gamma0 = d.sum() / tot
        prob[1] = (a - d) / tot + gamma0 / V       # interpolated with a uniform distribution over the vocab
        unk_p = gamma0 / V
    else:
        uctx, inv = np.unique(key(grams[:, :-1]), return_inverse=True)
        tot = np.bincount(inv, weights=a)
        nk = [np.bincount(inv, weights=(np.minimum(a, 3) == i)) for i in (1, 2, 3)]
        gamma = (D[1] * nk[0] + D[2] * nk[1] + D[3] * nk[2]) / tot
        lk = raw[n - 1][1]
        sk = key(grams[:, 1:])
        pos = np.searchsorted(lk, sk)
        assert (lk[np.minimum(pos, len(lk) - 1)] == sk).all(), f"missing lower-order suffix at order {n}"
        prob[n] = (a - d) / tot[inv] + gamma[inv] * prob[n - 1][pos]
        bow[n - 1] = (uctx, gamma)                 # back-off weight of every context of order n-1
    print(f"  order {n}: discounts D1..D3+ = {np.round(D[1:], 3)}", flush=True)

keep = {n: (raw[n][2] > args.prune) if (n >= 3 and args.prune) else np.ones(len(raw[n][1]), bool)
        for n in range(1, N + 1)}
# after pruning, re-solve each context's back-off weight so its distribution sums to 1 again:
#   bow(h) = (1 - sum of kept p(w|h)) / (1 - sum of the same words' p(w|shorter h))
for n in range(3, N + 1):
    if keep[n].all():
        continue
    grams, k, _ = raw[n]
    uctx, inv = np.unique(key(grams[:, :-1]), return_inverse=True)
    lk = raw[n - 1][1]
    lower = prob[n - 1][np.searchsorted(lk, key(grams[:, 1:]))]
    kp = keep[n]
    sp = np.bincount(inv[kp], weights=prob[n][kp], minlength=len(uctx))
    sl = np.bincount(inv[kp], weights=lower[kp], minlength=len(uctx))
    bow[n - 1] = (uctx, (1 - sp) / np.maximum(1 - sl, 1e-12))

# ---- write ARPA
bos_bow = 0.0
if N > 1:
    bk, bg = bow[1]
    i = np.searchsorted(bk, BOS)
    bos_bow = float(np.log10(bg[i])) if i < len(bk) and bk[i] == BOS else 0.0
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\\data\\\n")
    for n in range(1, N + 1):
        f.write(f"ngram {n}={int(keep[n].sum()) + (2 if n == 1 else 0)}\n")
    for n in range(1, N + 1):
        f.write(f"\n\\{n}-grams:\n")
        grams, k, _ = raw[n]
        lp = np.log10(prob[n])
        b = None
        if n < N:
            bk, bg = bow[n]
            pos = np.minimum(np.searchsorted(bk, k), len(bk) - 1)
            b = np.where(bk[pos] == k, np.log10(np.maximum(bg[pos], 1e-30)), 0.0)
        if n == 1:
            f.write(f"-99\t<s>\t{bos_bow:.6f}\n" if N > 1 else "-99\t<s>\n")
            f.write(f"{np.log10(unk_p):.6f}\t<unk>" + ("\t0.0\n" if N > 1 else "\n"))
        lines = []
        for i in np.flatnonzero(keep[n]):
            ws = " ".join(words[j] for j in grams[i])
            lines.append(f"{lp[i]:.6f}\t{ws}\t{b[i]:.6f}\n" if b is not None else f"{lp[i]:.6f}\t{ws}\n")
            if len(lines) >= 200000:
                f.writelines(lines); lines = []
        f.writelines(lines)
        print(f"  wrote {int(keep[n].sum())} {n}-grams", flush=True)
    f.write("\n\\end\\\n")
(HERE / "unigrams.txt").write_text("\n".join(words[2:]), encoding="utf-8")
print(f"wrote {args.out} in {time.time() - t0:.0f} s", flush=True)
