"""Hindi CTC speech-to-text on the EmbeddingGemma 2 audio encoder, after QUAN-Transcribe's design
(github.com/qnguyen3/QUAN-Transcribe, inference-only there; this version trains):

    wav --log-mel--> Gemma audio encoder (12-layer conformer, 25 frames/s; bottom layers frozen)
        --> softmax-weighted mix of its 13 hidden states
        --> Linear + causal depthwise conv + N pre-norm transformer layers (the reader)
        --> head over units (Gemma tokens) living in Gemma's 512-d text-embedding space
        --> CTC
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

ENCODER_ID = "google/embeddinggemma-2"
BLANK, UNK = 0, 1          # unit ids; real tokens start at 2


@dataclass
class Config:
    n_units: int                      # incl. blank and unk
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    unfreeze_top_k: int = 0           # top encoder layers to train (0 = encoder fully frozen)
    amp: str = "bf16"                 # bf16 (Ampere+) | fp16 (T4/P100) | fp16_enc32 (frozen encoder in fp32)

    def to_dict(self):
        return asdict(self)

    @property
    def enc_dtype(self):
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp16_enc32": torch.float32}[self.amp]

    @property
    def reader_dtype(self):
        return torch.bfloat16 if self.amp == "bf16" else torch.float16


class LogMel(nn.Module):
    """Batched log-mel matching the EmbeddingGemma 2 feature extractor (as QUAN's features.py)."""

    def __init__(self):
        super().__init__()
        fe = AutoFeatureExtractor.from_pretrained(ENCODER_ID)
        self.frame_length, self.hop_length, self.fft_length = fe.frame_length, fe.hop_length, fe.fft_length
        self.mel_floor = float(fe.mel_floor)
        self.register_buffer("window", torch.tensor(fe.window, dtype=torch.float32), persistent=False)
        self.register_buffer("mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, wav, wav_lens):
        """wav (B, L) float32 zero-padded -> features (B, T, 128), mask (B, T)."""
        pad_left = self.frame_length // 2
        wav = F.pad(wav.float(), (pad_left, 0))
        frames = wav.unfold(-1, self.frame_length + 1, self.hop_length)[..., :-1]
        spec = torch.fft.rfft(frames * self.window, n=self.fft_length, dim=-1).abs()
        feats = torch.log(spec @ self.mel_filters + self.mel_floor)
        frame_end = torch.arange(feats.shape[1], device=wav.device) * self.hop_length + self.frame_length
        mask = frame_end[None, :] < (wav_lens[:, None] + pad_left)
        return feats * mask[..., None], mask


class Block(nn.Module):
    def __init__(self, d, n_heads, ff_mult, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.norm2(x)))


def load_audio_encoder(dtype=torch.bfloat16):
    """The EmbeddingGemma 2 audio tower in `dtype` (the text tower is dropped) + Gemma's text embedding matrix."""
    full = AutoModel.from_pretrained(ENCODER_ID, dtype=torch.bfloat16, vision_config=None)
    encoder = full.audio_tower.to(dtype)
    if dtype == torch.float16:
        # masked attention logits are filled with -1e9, beyond fp16's ±65504. The logits are soft-capped at ±50
        # before the fill, so -1e4 is still "minus infinity" for the softmax.
        for m in encoder.modules():
            if hasattr(getattr(m, "config", None), "attention_invalid_logits_value"):
                m.config.attention_invalid_logits_value = -1e4
    text_emb = full.get_input_embeddings().weight.detach().float().clone()   # (vocab, 512)
    del full
    return encoder, text_emb


class GemmaCTC(nn.Module):
    def __init__(self, cfg: Config, encoder, unit_init=None):
        """unit_init: (n_units - 2, 512) Gemma text embeddings of the unit tokens (head init), or None."""
        super().__init__()
        self.cfg = cfg
        self.mel = LogMel()
        self.encoder = encoder
        n_enc = encoder.config.num_hidden_layers
        enc_dim = encoder.config.hidden_size
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.trainable_layers = list(self.encoder.layers[n_enc - cfg.unfreeze_top_k:]) if cfg.unfreeze_top_k else []
        for layer in self.trainable_layers:
            layer.float()                                  # trained layers in fp32, the frozen rest stays enc_dtype
            for p in layer.parameters():
                p.requires_grad_(True)

        # subsampler output + every conformer layer output, via hooks (output_hidden_states replaces the last one)
        self._acts = []
        grab = lambda mod, inp, out: self._acts.append(out[0] if isinstance(out, tuple) else out)  # noqa: E731
        self.encoder.subsample_conv_projection.register_forward_hook(grab)
        for layer in self.encoder.layers:
            layer.register_forward_hook(grab)

        d = cfg.d_model
        self.layer_logits = nn.Parameter(torch.zeros(n_enc + 1))
        self.in_norm = nn.LayerNorm(enc_dim)
        self.in_proj = nn.Linear(enc_dim, d)
        self.pos_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.to_text = nn.Linear(d, 512)
        rows = unit_init.clone() if unit_init is not None else torch.randn(cfg.n_units - 2, 512) * 0.02
        self.unit_rows = nn.Parameter(rows)
        self.special_rows = nn.Parameter(torch.randn(2, 512) * 0.02)    # blank, unk
        self.head_bias = nn.Parameter(torch.zeros(cfg.n_units))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _encode(self, feats, mask):
        self._acts = []
        enc_ctx = torch.enable_grad() if self.trainable_layers and self.training else torch.no_grad()
        enc_dtype = self.cfg.enc_dtype
        with enc_ctx, torch.autocast("cuda", dtype=enc_dtype, enabled=feats.is_cuda and enc_dtype != torch.float32):
            out = self.encoder(feats.to(enc_dtype), mask, return_dict=True)
        acts, self._acts = self._acts, []
        w = torch.softmax(self.layer_logits.float(), 0)
        h = sum(w[i] * a.float() for i, a in enumerate(acts))
        return h, out.attention_mask

    def forward(self, wav, wav_lens):
        """-> CTC log-probs (B, T, V) fp32, output lengths (B,)."""
        feats, fmask = self.mel(wav, wav_lens)
        h, mask = self._encode(feats, fmask)
        with torch.autocast("cuda", dtype=self.cfg.reader_dtype, enabled=wav.is_cuda):
            x = self.in_proj(self.in_norm(h))
            xc = F.pad((x * mask[..., None]).transpose(1, 2), (self.cfg.conv_kernel - 1, 0))
            x = x + F.gelu(self.pos_conv(xc).transpose(1, 2))
            attn_mask = mask[:, None, None, :]
            for blk in self.blocks:
                x = blk(x, attn_mask)
            t = self.to_text(self.out_norm(x))
        rows = torch.cat([self.special_rows, self.unit_rows], 0)
        logits = t.float() @ rows.float().t() + self.head_bias.float()
        return logits.log_softmax(-1), mask.sum(1)


def greedy_decode(logp, lens):
    """Best-path CTC: argmax, collapse repeats, drop blanks -> list of unit-id lists."""
    best = logp.argmax(-1).cpu()
    out = []
    for b in range(best.shape[0]):
        seq = best[b, :int(lens[b])]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != BLANK].tolist())
    return out
