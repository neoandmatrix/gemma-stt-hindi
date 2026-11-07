"""Kathbath-style parquet shards (audio_filepath.bytes = 16 kHz FLAC, text, duration) -> duration-bucketed batches.
Tables are memory-mapped, so many shards don't fill RAM; FLAC is decoded in DataLoader workers.
"""
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

from model import BLANK, UNK

SR = 16000
_PUNCT = re.compile(r"[^\w\sऀ-ॿ]")     # keep letters, digits, spaces, Devanagari (incl. matras)


def normalize(text):
    """For scoring: drop punctuation (incl. danda), collapse spaces."""
    text = text.replace("।", " ").replace("॥", " ")
    return " ".join(_PUNCT.sub(" ", text).split())


def _read(path):
    """.arrow (IPC, from to_arrow()) is memory-mapped zero-copy: the OS pages audio in on demand and processes share
    it. Parquet always decodes into RAM (two trainings on ~1000 h would need ~24 GB of Kaggle's 29 GB)."""
    if str(path).endswith(".arrow"):
        return pa.ipc.open_file(pa.memory_map(str(path))).read_all().select(["text", "duration", "audio_filepath"])
    return pq.read_table(path, columns=["text", "duration", "audio_filepath"], memory_map=True)


def to_arrow(parquet_path, arrow_path, batch_rows=2000):
    """Parquet -> uncompressed Arrow IPC (audio bytes are already compressed), a few row batches in RAM at a time."""
    pf = pq.ParquetFile(parquet_path)
    cols = ["text", "duration", "audio_filepath"]
    with pa.ipc.new_file(str(arrow_path), pa.schema([pf.schema_arrow.field(c) for c in cols])) as w:
        for b in pf.iter_batches(batch_size=batch_rows, columns=cols):
            w.write_batch(b)


class Clips:
    """All rows of some parquet / arrow shards: text, duration, FLAC or Opus bytes (lazily)."""

    def __init__(self, paths, max_clips=None, max_seconds=12.0, spread=False):
        """spread: take max_clips evenly across the files instead of the first ones (test sets sorted by speaker)."""
        tables = [_read(p) for p in paths]
        self.table = pa.concat_tables(tables)
        texts = self.table["text"].to_pylist()
        durs = self.table["duration"].to_pylist()
        self.idx = [i for i, d in enumerate(durs) if d <= max_seconds]
        if max_clips and spread and len(self.idx) > max_clips:
            self.idx = [self.idx[int(k * len(self.idx) / max_clips)] for k in range(max_clips)]
        elif max_clips:
            self.idx = self.idx[:max_clips]
        self.texts = [texts[i] for i in self.idx]
        self.durs = [durs[i] for i in self.idx]
        self._audio = self.table["audio_filepath"]

    def __len__(self):
        return len(self.idx)

    def hours(self):
        return sum(self.durs) / 3600

    def audio(self, j):
        b = self._audio[self.idx[j]].as_py()["bytes"]
        wav, sr = sf.read(io.BytesIO(b), dtype="float32")
        assert sr == SR, sr
        return wav


class Units:
    """CTC output units = the Gemma tokens seen in the training text. id 0 blank, 1 unk, 2.. = tokens."""

    def __init__(self, token_ids, tokenizer):
        self.token_ids = list(token_ids)
        self.tok = tokenizer
        self.to_unit = {t: i + 2 for i, t in enumerate(self.token_ids)}

    @classmethod
    def build(cls, texts, tokenizer, min_count=1):
        counts = {}
        for x in texts:
            for t in tokenizer(x, add_special_tokens=False)["input_ids"]:
                counts[t] = counts.get(t, 0) + 1
        keep = sorted(t for t, c in counts.items() if c >= min_count)
        return cls(keep, tokenizer)

    @classmethod
    def load(cls, path, tokenizer):
        return cls(json.loads(Path(path).read_text())["token_ids"], tokenizer)

    def save(self, path):
        Path(path).write_text(json.dumps({"token_ids": self.token_ids}))

    def __len__(self):
        return len(self.token_ids) + 2

    def encode(self, text):
        return [self.to_unit.get(t, UNK) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    def decode(self, units):
        toks = [self.token_ids[u - 2] for u in units if u not in (BLANK, UNK)]
        return self.tok.decode(toks).strip()


class CharUnits(Units):
    """CTC output units = characters (incl. space) seen in the training text: far fewer units than Gemma tokens,
    each seen many times, so it learns from little data. token_ids = the Gemma token of each character (for the
    head init; Gemma's unk id where it has none)."""

    def __init__(self, chars, tokenizer):
        self.chars = list(chars)
        self.tok = tokenizer
        self.to_unit = {c: i + 2 for i, c in enumerate(self.chars)}
        unk = tokenizer.unk_token_id or 0
        self.token_ids = [tokenizer.convert_tokens_to_ids("▁" if c == " " else c) for c in self.chars]
        self.token_ids = [t if t is not None else unk for t in self.token_ids]

    @classmethod
    def build(cls, texts, tokenizer, min_count=1):
        counts = {}
        for x in texts:
            for c in " ".join(x.split()):
                counts[c] = counts.get(c, 0) + 1
        return cls(sorted(c for c, n in counts.items() if n >= min_count), tokenizer)

    @classmethod
    def load(cls, path, tokenizer):
        return cls(json.loads(Path(path).read_text())["chars"], tokenizer)

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "char", "chars": self.chars}, ensure_ascii=False))

    def encode(self, text):
        return [self.to_unit.get(c, UNK) for c in " ".join(text.split())]

    def decode(self, units):
        return "".join(self.chars[u - 2] for u in units if u not in (BLANK, UNK)).strip()


def load_units(path, tokenizer):
    kind = json.loads(Path(path).read_text()).get("type", "token")
    return (CharUnits if kind == "char" else Units).load(path, tokenizer)


def make_batches(durs, max_batch_seconds, shuffle, seed=0):
    """Indices grouped so each batch has <= max_batch_seconds of (padded) audio; similar lengths together."""
    order = sorted(range(len(durs)), key=lambda i: durs[i])
    batches, cur, longest = [], [], 0.0
    for i in order:
        longest_new = max(longest, durs[i])
        if cur and longest_new * (len(cur) + 1) > max_batch_seconds:
            batches.append(cur); cur, longest_new = [], durs[i]
        cur.append(i); longest = longest_new
    if cur:
        batches.append(cur)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, clips, units, batches):
        self.clips, self.units, self.batches = clips, units, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, b):
        idx = self.batches[b]
        wavs = [self.clips.audio(i) for i in idx]
        tgts = [self.units.encode(self.clips.texts[i]) for i in idx]
        n = max(len(w) for w in wavs)
        wav = np.zeros((len(wavs), n), np.float32)
        for k, w in enumerate(wavs):
            wav[k, :len(w)] = w
        return dict(wav=torch.from_numpy(wav), wav_lens=torch.tensor([len(w) for w in wavs]),
                    tgt=torch.tensor([u for t in tgts for u in t]), tgt_lens=torch.tensor([len(t) for t in tgts]),
                    idx=idx)


class ThreadedLoader:
    """Batches decoded by a few threads (libsndfile releases the GIL), at most `ahead` batches in flight.
    Replaces DataLoader worker processes: on Kaggle, two trainings with process workers froze mid-run (a worker
    died and the main process waited forever on its pipe; night 1 and night 2)."""

    def __init__(self, ds, threads, ahead=8):
        self.ds, self.threads, self.ahead = ds, threads, ahead

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.threads) as ex:
            q, nxt = deque(), 0
            while q or nxt < len(self.ds):
                while nxt < len(self.ds) and len(q) < self.ahead:
                    q.append(ex.submit(self.ds.__getitem__, nxt)); nxt += 1
                yield q.popleft().result()


def loader(clips, units, max_batch_seconds, shuffle, seed=0, workers=4):
    """workers = decoding threads (0: decode in the calling thread)."""
    ds = BatchDataset(clips, units, make_batches(clips.durs, max_batch_seconds, shuffle, seed))
    return ThreadedLoader(ds, workers) if workers else _Plain(ds)


class _Plain:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        return (self.ds[i] for i in range(len(self.ds)))
