# 04 · Hindi speech-to-text on EmbeddingGemma 2 (CTC)

QUAN-Transcribe's recipe (github.com/qnguyen3/QUAN-Transcribe) applied to Hindi, trained on the RTX 3050 (4 GB):
the EmbeddingGemma 2 audio encoder (12-layer conformer, 305M, 25 frames/s), a softmax mix of its 13 hidden states,
a 6-layer reader (d=768), and a CTC head initialized from Gemma text embeddings.

- `model.py`: model (trainable version), log-mel, greedy CTC decode
- `data.py`: parquet shards → duration-bucketed batches; `Units` (Gemma tokens) / `CharUnits` (characters)
- `train.py`: training, CER/WER eval, `best.pt`/`last.pt`, `--resume`, `--gate-hours/--gate-cer` auto-stop
- `probe_memory.py`: VRAM/speed check for frozen vs top-k unfrozen
- Data and test sets: see `DATA.md`

## Memory / speed on the 3050 (random audio, batch 8×6 s)
| setup | trainable | peak VRAM | speed |
|---|---|---|---|
| frozen encoder | 51M | 1.93 GiB | 168 s audio/s |
| top-2 encoder layers trained | 102M | 3.00 GiB | 129 s audio/s |

## Results (Kathbath Hindi; eval = 1,000 clips of `valid-00000`)
| run | data | units | setup | CER | WER | notes |
|---|---|---|---|---|---|---|
| s0_overfit | 50 clips | Gemma tokens | frozen | **1.4%** (on train) | 2.0% | Pipeline correct |
| s1_4h_frozen | 3.97 h (shard 0), 8 epochs, 15 min | Gemma tokens (6,687) | frozen | 71.7% | 68.8% | Only learns frequent words (इस, के, को, किया); too many word-like units for 4 h |
| **s1_4h_frozen_char** | same | **characters (66)** | frozen | **11.9%** | **40.6%** | Errors are sound-alike spellings; data-limited (train loss 0.08, eval flat after ~1,600 steps) |

## My own voice (`data/mytest/`: 15 clips, 74 s; 5 Hindi, 7 Hinglish, numbers, names, long)
Recorded with `record_testset.py`; scored with `uv run score.py <run> data/mytest/mytest.parquet`.

| model | Kathbath valid | Lahaja (accents) | **my voice** |
|---|---|---|---|
| s1_4h_frozen_char | 11.9% | 22.3% | **45.2%** (Hindi 39%, Hinglish 44%, names 67%) |

- Gets some loanwords (मीटिंग, फ्राइडे, रैम); fails on names and fast phrases. The gap comes from read vs natural speech,
  only 4 h of data, and phone vs laptop mic. Next lever after more hours: IndicVoices (spontaneous).
- ⚠️ Mic gotcha: on this laptop the mic delivers **1–2 s of digital zeros after opening** (PipeWire resuming it).
  The first recording lost every sentence's beginning (56% CER, invalid; kept in `data/mytest/bad_start/`).
  The recorder now waits for live audio before "SPEAK NOW". The same delay explains the leading silence in
  01's `my_voice.wav` and 02's `real_turns.wav`.

## fp16 check (for Kaggle T4/P100, which lack fast bf16), 2026-10-27
| check | bf16 (reference) | fp16 | fp16_enc32 |
|---|---|---|---|
| eval of s1 char `best.pt` (1,000 clips) | CER 11.93% | **11.92%** | 11.92% |
| train from scratch, step 400 | loss 0.661, CER 18.0% | **loss 0.655, CER 18.2%, 0 non-finite** | – |

- Pure fp16 crashed at first: Gemma fills masked attention logits with `-1e9`, which overflows fp16.
  Fix in `load_audio_encoder`: use `-1e4` in fp16 (logits are soft-capped at ±50, so it's still effectively −∞).
- Training uses `GradScaler` in fp16. **Use `--amp fp16` on Kaggle**; `fp16_enc32` is the fallback.

Lessons:
- **Use character units at this data scale.** Word-like Gemma tokens 72% CER → characters 12% CER, same data and compute.
- The frozen EmbeddingGemma encoder clearly encodes Hindi phonetics, so the idea works. More hours is the next lever.
- CTC sits at blank-only output for the first ~200–400 steps; don't judge it before then.
- With little data, word-like units starve: most of the 6.7k units appear only a few times in 4 h.
