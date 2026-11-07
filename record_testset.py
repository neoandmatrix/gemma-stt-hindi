"""Record my own Hindi/Hinglish test clips, one sentence at a time, then pack them like the other test sets
(parquet with audio_filepath.bytes FLAC, text, duration) so train.py can score them: --eval-set mine data/mytest/mytest.parquet

usage: uv run record_testset.py            # record all (skips ones already recorded)
       uv run record_testset.py --redo 07  # re-record one
       uv run record_testset.py --pack     # only rebuild mytest.parquet from the wavs
"""
import argparse
import csv
import io
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

DIR = Path(__file__).parent / "data" / "mytest"
SR = 16000


def rows():
    with open(DIR / "sentences.tsv", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mic_live(path, timeout=6.0):
    """The mic delivers exact zeros for ~1-2 s after it opens (PipeWire resuming a suspended source):
    wait until real (non-zero) samples arrive before telling the user to speak."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.2)
        try:
            wav, _ = sf.read(path, dtype="int16")
        except Exception:
            continue
        if len(wav) and np.abs(wav[-1600:]).max() > 0:
            return True
    return False


def record(path):
    """Enter starts the mic, speak at 'SPEAK NOW', Enter stops. ffmpeg stops cleanly on 'q' on stdin."""
    input("   press Enter to open the mic ")
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "pulse", "-i", "default",
                          "-ac", "1", "-ar", str(SR), "-flush_packets", "1", str(path)], stdin=subprocess.PIPE)
    print("   ...waiting for the mic", end="", flush=True)
    if not mic_live(path):
        print("  (mic still silent after 6 s, speaking anyway)", end="")
    time.sleep(0.3)
    input("\r   🔴 SPEAK NOW. Press Enter when done                ")
    p.communicate(b"q")
    wav, _ = sf.read(path, dtype="float32")
    nz = np.flatnonzero(wav)                       # trim the leading/trailing digital zeros
    if len(nz):
        wav = wav[max(nz[0] - 1600, 0):nz[-1] + 1]
        sf.write(path, wav, SR)
    peak = abs(wav).max() if len(wav) else 0
    print(f"   saved {len(wav) / SR:.1f}s, peak {peak:.2f}" + ("  ⚠️ very quiet, move closer?" if peak < 0.05 else ""))


def pack():
    texts, durs, audio = [], [], []
    for r in rows():
        wav_path = DIR / f"{r['id']}.wav"
        if not wav_path.exists():
            continue
        wav, sr = sf.read(wav_path, dtype="float32")
        buf = io.BytesIO(); sf.write(buf, wav, sr, format="FLAC")
        texts.append(r["text"]); durs.append(len(wav) / sr)
        audio.append({"bytes": buf.getvalue(), "path": wav_path.name})
    pq.write_table(pa.table({"text": texts, "duration": durs, "audio_filepath": audio}), DIR / "mytest.parquet")
    print(f"packed {len(texts)} clips ({sum(durs):.0f}s) -> {DIR / 'mytest.parquet'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", nargs="*", default=[])
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    if not a.pack:
        print("Speak naturally, like talking to a friend, not like reading aloud. Quiet room, normal distance.\n")
        for r in rows():
            path = DIR / f"{r['id']}.wav"
            if path.exists() and r["id"] not in a.redo:
                continue
            print(f"[{r['id']}] ({r['kind']})\n   {r['text']}\n   {r['romanized (reading help only)']}")
            record(path)
            while input("   keep it? [Enter = yes / r = redo] ").strip().lower() == "r":
                record(path)
            print()
    pack()


if __name__ == "__main__":
    main()
