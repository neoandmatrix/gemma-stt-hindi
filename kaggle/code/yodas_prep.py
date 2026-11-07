"""YODAS v3 Hindi (whole YouTube videos + timed auto-captions) -> Kathbath-style parquet of short clips.

Reads the audio tar as a stream, so a shard can be piped straight from Hugging Face without touching disk:
  curl -sL https://huggingface.co/datasets/espnet/yodas3/resolve/main/data/hi/audio/0000.tar | \
      python yodas_prep.py - data/yodas3/meta_0000.parquet out/yodas_0000.parquet
  python yodas_prep.py data/yodas3_sample/audio_0000_head.tar data/yodas3_sample/meta_0000.parquet out.parquet

Clips: consecutive caption words grouped into 2-15 s pieces, split at pauses. Kept only if the text is plain Devanagari
(no digits, Latin, [संगीत]-style tags) at a plausible speaking rate. Audio: 16 kHz mono Opus (6x smaller than
FLAC; data.Clips reads either).
"""
import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import time
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
DEVANAGARI_ONLY = re.compile(r"[ऀ-ॿ\s।,?!.]+")
MAX_S, MIN_S, PAUSE_S, PAD_S = 12.0, 2.0, 0.6, 0.15   # 12 s = data.Clips' default cap
MIN_CPS, MAX_CPS = 5.0, 25.0                    # characters per second; Kathbath read speech is ~13


def words_of(segments):
    """Caption segments -> [(start_s, end_s, word)]. YouTube captions roll (each segment's duration overlaps the
    next), so a word ends where the next word starts, capped at 1 s."""
    out = []
    for i, s in enumerate(segments):
        if "words" in s:
            out += [((s["start"] + w["t"]) / 1000, w["w"]) for w in s["words"]]
        else:                                    # no word timings: the whole line as one "word"
            out.append((s["start"] / 1000, s["text"]))
    out.sort(key=lambda x: x[0])
    return [(t, min(out[i + 1][0] if i + 1 < len(out) else t + 1.0, t + 1.0), w) for i, (t, w) in enumerate(out)]


def clips_of(words):
    """Group words into clips: break at a pause, a tag, or when the clip would exceed MAX_S."""
    clips, cur = [], []
    for st, en, w in words:
        if cur and (st - cur[-1][1] > PAUSE_S or en - cur[0][0] > MAX_S - 2 * PAD_S or "[" in w):
            clips.append(cur); cur = []
        if "[" not in w:
            cur.append((st, en, w))
    if cur:
        clips.append(cur)
    for c in clips:
        text = " ".join(re.sub(r"[।,?!.]", " ", " ".join(w for _, _, w in c)).split())   # Kathbath-style: no punctuation
        text = unicodedata.normalize("NFC", text)   # NFC splits ज़ (U+095B) into ज + ़, as Kathbath writes it
        st, en = max(0.0, c[0][0] - PAD_S), c[-1][1] + PAD_S
        yield st, en, text


def decode(webm_bytes):
    """Any container/codec -> 16 kHz mono float32 via ffmpeg."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       input=webm_bytes, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def encode(wav, codec):
    """Opus via PyAV/libopus (24 kbps, voip, complexity 5): ~190x real time per core, ~11 MB/h.
    libsndfile's Opus writer ran at ~23x and made Kaggle conversion the bottleneck (night 2); one ffmpeg process per
    clip was little better (start-up cost). FLAC: ~4600x but 77 MB/h."""
    if codec == "flac":
        b = io.BytesIO()
        sf.write(b, wav, SR, format="FLAC")
        return b.getvalue()
    import av
    b = io.BytesIO()
    with av.open(b, "w", format="ogg") as c:
        s = c.add_stream("libopus", rate=SR, layout="mono",
                         options={"application": "voip", "compression_level": "5", "b": "24000"})
        fr = av.AudioFrame.from_ndarray(wav[None, :].astype(np.float32), format="flt", layout="mono")
        fr.sample_rate = SR
        for pkt in [*s.encode(fr), *s.encode(None)]:
            c.mux(pkt)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", help="audio tar path, or - for stdin (a truncated tar is fine)")
    ap.add_argument("meta", help="the matching metadata parquet")
    ap.add_argument("out")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop reading after this long and write what's done")
    ap.add_argument("--codec", choices=["opus", "flac"], default="opus", help="flac: ~4600x real time but 77 MB/h")
    a = ap.parse_args()

    meta = pq.read_table(a.meta, columns=["id", "transcript", "locale_lang"]).to_pydict()
    meta = {i: (t, loc) for i, t, loc in zip(meta["id"], meta["transcript"], meta["locale_lang"])}
    t_start = time.time()
    stats = dict(videos=0, clips_kept=0, clips_dropped=0, h_kept=0.0, h_dropped=0.0, errors=0)
    rows = {"id": [], "text": [], "duration": [], "audio_filepath": [], "locale": []}
    src = tarfile.open(fileobj=sys.stdin.buffer, mode="r|") if a.tar == "-" else tarfile.open(a.tar, mode="r|")
    try:
        for m in src:
            if not m.name.endswith((".webm", ".opus", ".m4a")):
                continue
            vid = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if vid not in meta:
                continue
            try:
                wav = decode(src.extractfile(m).read())
            except Exception:
                stats["errors"] += 1
                continue
            stats["videos"] += 1
            for k, (st, en, text) in enumerate(clips_of(words_of(json.loads(meta[vid][0])))):
                dur = min(en, len(wav) / SR) - st
                n = len(text.replace(" ", ""))
                if (dur < MIN_S or not DEVANAGARI_ONLY.fullmatch(text) or not MIN_CPS <= n / dur <= MAX_CPS):
                    stats["clips_dropped"] += 1; stats["h_dropped"] += max(dur, 0) / 3600
                    continue
                piece = wav[int(st * SR):int((st + dur) * SR)]
                rows["id"].append(f"{vid}_{k:04d}"); rows["text"].append(text); rows["duration"].append(round(dur, 3))
                rows["audio_filepath"].append({"bytes": encode(piece, a.codec), "path": f"{vid}_{k:04d}.{a.codec}"})
                rows["locale"].append(meta[vid][1])
                stats["clips_kept"] += 1; stats["h_kept"] += dur / 3600
            if a.max_videos and stats["videos"] >= a.max_videos:
                break
            if a.max_minutes and time.time() - t_start > a.max_minutes * 60:
                stats["stopped_at_time_limit"] = 1
                break
    except (tarfile.ReadError, EOFError):        # truncated tar (partial download): keep what we have
        pass
    pq.write_table(pa.table(rows), a.out)
    stats = {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
