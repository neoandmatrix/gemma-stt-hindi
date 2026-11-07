# gemma-stt-hindi: Hindi speech-to-text from an embedding model, on free GPUs

This project turns Google's **EmbeddingGemma 2** audio encoder into a Hindi speech-to-text model. The training ran
on free Kaggle T4s, and the development on a 4 GB laptop GPU (RTX 3050). The final model transcribes a 5–8 s clip in
**~40 ms on the laptop** (about 140× faster than real time). With a KenLM 3-gram language model it reaches
**5.4% CER on Lahaja**, which is on par with Qwen3-ASR-1.7B.

- Model weights: [bringagi/gemma-stt-hindi](https://huggingface.co/bringagi/gemma-stt-hindi) (Indic Open Model
  License v1.0; see [Licences](#licences)).
- The idea and design come from [QUAN-Transcribe](https://github.com/qnguyen3/QUAN-Transcribe), a Vietnamese
  speech-to-text model built on the same encoder. The code here is an independent implementation that also trains.

## Results

All models below were scored by the same script on the same clips. CER and WER are computed after removing
punctuation. Lattice-WER accepts any listed spelling as correct.

| model | 15 laptop-mic clips (my voice) | Lahaja-100 CER / WER | Monsoon-hi lattice-WER |
|---|---|---|---|
| night 1: Kathbath 50 h | 41.3% | 15.0% / 48.3% | 66.5% |
| night 3: 982 h YouTube captions → Kathbath | 29.2% | 10.1% / 30.9% | 42.6% |
| night 4: 1,810 h YouTube captions → Kathbath | 26.3% | 9.8% / 32.3% | 40.2% |
| night 7: ~1,750 h of teacher labels (`n7_all_k2`) | 26.8% | 6.4% / 21.3% | 28.9% |
| **night 7 + KenLM 3-gram (released)** | **25.3%** | **5.4% / 15.9%** | **20.9%** |
| Qwen3-ASR-1.7B | 15.4% | 6.4% / 17.4% | 12.8% |
| Hima-ASR (teacher) | 14.2% | 3.4% / 11.2% | 10.6% |
| IndicConformer-600M | 11.3% | 3.3% / 10.9% | 12.0% |
| Indic-Transcribe-core (teacher) | 9.3% | 2.9% / 9.8% | 5.3% |

The full table is in `results/stt_scores.md`. Speed is ~40 ms per clip on the RTX 3050 laptop (fp16, 1.2 GB VRAM).
The KenLM beam search adds ~10 ms of CPU per clip. The teachers take 0.5–0.9 s per clip on a cloud L4.

## The model

```
wav ─log-mel─> EmbeddingGemma 2 audio encoder (12-layer conformer, 305M, 25 frames/s; top 2 layers trained)
    ─> softmax mix of its 13 hidden states
    ─> Linear + causal depthwise conv + 6 pre-norm transformer layers (d=768)      "the reader"
    ─> 512-d projection · 66 character rows (initialised from Gemma's text embeddings) + blank + unk
    ─> CTC  (greedy, or beam search with a word 3-gram KenLM)
```

There is no autoregressive decoder: one forward pass transcribes the whole clip. That is why it is ~10× faster than
the teachers, and also why it needs a language model to fix spellings that sound right but aren't words.

## How it got there (the method, night by night)

Each "night" is one free Kaggle session (2× T4, up to ~12 h). `docs/JOURNAL.md` has every entry with all its numbers.

1. **Laptop stage (`docs/stage1_laptop.md`).** First check the pipeline: overfitting 50 clips reaches 1.4% CER.
   - On 4 h of Kathbath, **characters beat Gemma word tokens: 12% vs 72% CER.** With little data, word-like units
     each get too few examples to learn.
   - fp16 for the T4s: Gemma fills masked attention with −1e9, which overflows fp16. Using −1e4 gives the same
     result.
2. **Night 1 (20–50 h of Kathbath).** Kathbath CER fell to 6.3%, but Lahaja (accents) and my own voice stayed flat
   at ~16% and ~40%. More read speech didn't help on other kinds of speech.
3. **Nights 2–3 (982 h of YouTube, YODAS v3 auto-captions).** `yodas_prep.py` cuts the videos into 2–12 s clips at
   caption pauses on a CPU-only Kaggle notebook, without using GPU quota.
   - **Order matters:** noisy YouTube first, then a clean Kathbath fine-tune, gave Lahaja 10.1%. Mixing the two
     sets gave 14.3%.
4. **Night 4 (1,810 h of captions).** Doubling the hours gained only ~1 point. The captions are noisy: they differ
   from a good ASR model's output by 27% CER.
5. **Night 5 (relabel with Hima-ASR).** In an A/B test on the same clips, **780 h with Hima labels beat 1,810 h
   with YouTube captions** (Lahaja 9.4% vs 16.0% before the fine-tune). Noisy labels made the model worse the
   longer it trained.
6. **Night 6 (relabel the rest with Indic-Transcribe).** This was a relabel-only night that kept 970 h, for a total
   of ~1,750 h with teacher labels.
7. **Night 7 (train on all of it).** Lahaja went 9.6% → 7.6% (training eval) and 6.4% on the shoot-out subset. The
   Kathbath fine-tune again traded Lahaja for Kathbath, so the released model skips it.
8. **KenLM.** A word 3-gram built from the teacher transcripts (`lm/build_arpa.py`: modified Kneser-Ney in numpy,
   because lmplz needs Boost). The weights were tuned on 800 dev clips. WER fell 23–29% relative on every set.

**Lessons:**
- At this scale, **label quality beat quantity**.
- Characters work better than word tokens.
- Train on noisy-but-varied data first and clean data last.
- An n-gram LM is a free win for a CTC model.
- The remaining gap to the teachers is on casual laptop-mic speech: 25% CER on my own clips.

## Files

| path | what |
|---|---|
| `model.py` | the model: log-mel, encoder hooks, reader, CTC head, greedy decoding |
| `data.py` | parquet/Arrow shards, duration-bucketed batches, character / token units |
| `train.py` | training and eval (CER/WER), `best.pt`/`last.pt`, resume, time budget, fp16 + GradScaler |
| `yodas_prep.py` | YODAS v3 tar → 2–12 s Opus clips with caption text, streamed, CPU only |
| `relabel.py` | batched teacher relabelling (Hima-ASR / Indic-Transcribe) with filters |
| `kaggle/kernel*/` | the Kaggle runner for each night (`run_kaggle.py` + `kernel-metadata.json`) |
| `kaggle/night*/` | what each night produced: summary.json, per-run config/log.jsonl, training logs |
| `lm/build_arpa.py`, `lm/decode_lm.py` | KenLM: build the 3-gram; dump logits, tune alpha/beta, beam-search decode |
| `shootout_hyps.py`, `score.py` | transcribe the shoot-out test sets; score a run on my clips |
| `export_hf.py`, `hf_card/` | build the standalone Hugging Face folder (+ check that it matches the checkpoint) |
| `probe_memory.py`, `record_testset.py` | VRAM/speed probe; recorder for my own test clips |
| `docs/` | the journal entries and the first laptop-stage write-up |
| `results/stt_scores.md` | the shoot-out scoreboard (all models) |

The data, checkpoints and my own voice recordings are not in this repository.

## Reproduce

```bash
uv sync                                   # torch 2.14.1, transformers 5.21.0 (Gemma 4 audio), ... from uv.lock
# KenLM decoding also needs these two, installed separately because pyctcdecode pins numpy<2 (it works on numpy 2):
uv pip install --no-deps pyctcdecode pygtrie sortedcontainers https://github.com/kpu/kenlm/archive/master.zip

# small local run (laptop): 4 h of Kathbath, characters, frozen encoder
uv run train.py --run s1_char --train data/kathbath/hindi/train-00000-of-00032.parquet \
    --eval data/kathbath/hindi/valid-00000-of-00002.parquet --unit-type char --epochs 8
```

The big runs are Kaggle scripts.
1. Upload this code as a Kaggle dataset.
2. Run `kaggle/kernel_prep*` (YODAS clips, CPU).
3. Run the relabel nights (`kaggle/kernel`, night 5; `kaggle/kernel6`).
4. Run `kaggle/kernel7` (training).

Each runner's docstring describes the plan. The HF token is read from Kaggle's secret store.

## Licences
- **Code:** no licence has been chosen yet, so all rights are reserved for now.
- **Released model weights:** the Indic Open Model License v1.0. The model is a Derivative of Indic-Transcribe-core
  because it was trained on its transcripts. "Built with Indic-Transcribe-core from Bodhan AI / AI4Bharat."
  - The Gemma Terms of Use restrictions apply too, through Hima-ASR's Gemma decoder.
  - See the model card's `NOTICE` for every upstream part.
- **Data:** YODAS v3 (CC-BY-3.0) and Kathbath (CC-BY-4.0) for training. Lahaja and MonsoonASR are used for
  evaluation only.
