"""Hindi speech-to-text with gemma-stt-hindi: EmbeddingGemma 2's audio encoder + a CTC reader, ~40 ms per clip
on an RTX 3050 laptop GPU.

    from gemma_stt import GemmaSTT
    stt = GemmaSTT.from_pretrained("bringagi/gemma-stt-hindi")      # or a local folder
    stt.transcribe("clip.wav")                                       # greedy CTC
    stt.transcribe(["a.wav", "b.wav"], lm=True)                      # + KenLM 3-gram beam search (better WER)

Needs torch, transformers (with Gemma 4 audio support), safetensors, soundfile; for lm=True also pyctcdecode and
kenlm (pip install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip). Audio at 16 kHz mono is used as
is; other sample rates need scipy (or convert first: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 16000


class GemmaSTT:
    def __init__(self, folder, device=None, dtype=torch.float16):
        folder = Path(folder)
        sys.path.insert(0, str(folder))
        import model as M                                   # model.py shipped next to this file
        M.ENCODER_ID = str(folder)                          # log-mel settings from our preprocessor_config.json
        from safetensors.torch import load_file
        from transformers.models.gemma4 import Gemma4AudioConfig, Gemma4AudioModel

        conf = json.loads((folder / "config.json").read_text())
        self.chars = conf["units"]
        self.lm_conf = conf.get("lm", {})
        self.folder = folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        amp = "fp16" if (self.device == "cuda" and dtype == torch.float16) else \
              "bf16" if (self.device == "cuda" and dtype == torch.bfloat16) else "fp16_enc32"
        cfg = M.Config(**{**conf["model"], "amp": amp, "dropout": 0.0})
        encoder = Gemma4AudioModel(Gemma4AudioConfig(**conf["audio_config"])).to(cfg.enc_dtype)
        if cfg.enc_dtype == torch.float16:                  # see model.load_audio_encoder: -1e9 overflows fp16
            for m in encoder.modules():
                if hasattr(getattr(m, "config", None), "attention_invalid_logits_value"):
                    m.config.attention_invalid_logits_value = -1e4
        self.model = M.GemmaCTC(cfg, encoder, None)
        missing, unexpected = self.model.load_state_dict(load_file(folder / "model.safetensors"), strict=False)
        assert not unexpected and all(k.startswith("mel.") for k in missing), (missing, unexpected)
        self.model.to(self.device).eval()
        self._greedy = M.greedy_decode
        self._decoder = None

    @classmethod
    def from_pretrained(cls, repo_or_folder="bringagi/gemma-stt-hindi", **kw):
        p = Path(repo_or_folder)
        if not p.exists():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(repo_or_folder))
        return cls(p, **kw)

    @staticmethod
    def load_audio(path):
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if sr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        return wav

    def _lm_decoder(self):
        if self._decoder is None:
            from pyctcdecode import build_ctcdecoder
            unigrams = (self.folder / "kenlm/unigrams.txt").read_text(encoding="utf-8").split("\n")
            self._decoder = build_ctcdecoder(["", "⁇"] + self.chars, str(self.folder / "kenlm/hi3.arpa"),
                                             unigrams=unigrams, alpha=self.lm_conf.get("alpha", 0.6),
                                             beta=self.lm_conf.get("beta", 1.0))
        return self._decoder

    @torch.no_grad()
    def transcribe(self, audio, lm=False, beam=None):
        """audio: a path, a 16 kHz float array, or a list of them -> text (or a list of texts)."""
        single = not isinstance(audio, (list, tuple))
        items = [audio] if single else list(audio)
        wavs = [self.load_audio(a) if isinstance(a, (str, Path)) else np.asarray(a, np.float32) for a in items]
        lens = torch.tensor([len(w) for w in wavs])
        batch = torch.zeros(len(wavs), int(lens.max()))
        for i, w in enumerate(wavs):
            batch[i, :len(w)] = torch.from_numpy(w)
        logp, ol = self.model(batch.to(self.device), lens.to(self.device))
        if lm:
            dec = self._lm_decoder()
            out = []
            for i in range(len(wavs)):
                lp = logp[i, :int(ol[i])].float().cpu().numpy()
                lp[:, 1] = -1e4                              # never emit <unk>
                out.append(dec.decode(lp, beam_width=beam or self.lm_conf.get("beam", 64)))
        else:
            out = ["".join(self.chars[u - 2] for u in seq if u >= 2).strip() for seq in self._greedy(logp, ol)]
        return out[0] if single else out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="transcribe Hindi audio files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="bringagi/gemma-stt-hindi")
    ap.add_argument("--lm", action="store_true", help="KenLM beam search (needs pyctcdecode + kenlm)")
    a = ap.parse_args()
    stt = GemmaSTT.from_pretrained(a.model)
    for f, t in zip(a.files, stt.transcribe(a.files, lm=a.lm)):
        print(f"{f}\t{t}")
