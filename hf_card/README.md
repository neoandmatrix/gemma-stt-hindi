---
license: other
license_name: indic-open-model-license-v1
license_link: LICENSE
language:
- hi
pipeline_tag: automatic-speech-recognition
library_name: transformers
base_model: google/embeddinggemma-2
tags:
- automatic-speech-recognition
- hindi
- ctc
- kenlm
- embeddinggemma
datasets:
- espnet/yodas3
- ai4bharat/Kathbath
---

# gemma-stt-hindi

> **Built with Indic-Transcribe-core from Bodhan AI / AI4Bharat.**
> This model was trained on transcripts produced by Indic-Transcribe-core and Hima-ASR, so it is a Derivative under
> the Indic Open Model License v1.0 and is released under that licence (see `LICENSE` and `NOTICE`).

A small, fast Hindi speech-to-text model: Google's **EmbeddingGemma 2 audio encoder** (a 12-layer conformer, 305M
parameters) with its top 2 layers fine-tuned, plus a 6-layer CTC "reader" (d=768) that outputs Devanagari characters.
It was trained on free Kaggle T4 GPUs by distilling two larger Hindi ASR models into it. It needs no autoregressive
decoder, so one forward pass transcribes a whole clip.

- **Speed:** ~40 ms for a 5.6 s clip on an RTX 3050 Laptop GPU (4 GB), about 140× faster than real time;
  1.2 GB of VRAM in fp16. The optional KenLM beam search adds ~10 ms of CPU per clip.
- **Accuracy:** 5.4% CER on Lahaja (with the LM), on par with Qwen3-ASR-1.7B, but still about 2× the error of
  its teacher Indic-Transcribe.

## Use

```python
# pip install torch transformers safetensors soundfile huggingface_hub
# for lm=True also: pip install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip
from huggingface_hub import snapshot_download
import sys; sys.path.insert(0, snapshot_download("bringagi/gemma-stt-hindi"))
from gemma_stt import GemmaSTT

stt = GemmaSTT.from_pretrained("bringagi/gemma-stt-hindi")
print(stt.transcribe("clip.wav"))             # greedy CTC
print(stt.transcribe("clip.wav", lm=True))    # + 3-gram LM beam search: ~25% fewer word errors
```

or from the command line: `python gemma_stt.py a.wav b.wav --lm`. Input: 16 kHz mono (other rates are resampled
if scipy is installed). Output: Devanagari without punctuation or digits (numbers come out as words).
A transformers version with Gemma 4 audio support is needed (tested with transformers 5.21.0, torch 2.14.1).

## Results

CER / WER after removing punctuation; lattice-WER counts any accepted spelling as correct.

| model | 15 laptop-mic clips (the author's voice) | Lahaja-100 (CER / WER) | Monsoon-hi lattice-WER | GPU, time per clip |
|---|---|---|---|---|
| **gemma-stt-hindi + KenLM** | 25.3% / 40.6% | **5.4% / 15.9%** | 20.9% | RTX 3050 laptop, ~50 ms |
| gemma-stt-hindi, greedy | 26.8% / 53.1% | 6.4% / 21.3% | 28.9% | RTX 3050 laptop, ~40 ms |
| Qwen3-ASR-1.7B | 15.4% / 23.1% | 6.4% / 17.4% | 12.8% | L4, 2.5–4 s |
| Hima-ASR (teacher) | 14.2% / 22.5% | 3.4% / 11.2% | 10.6% | L4, ~0.9 s |
| IndicConformer-600M | 11.3% / 23.8% | 3.3% / 10.9% | 12.0% | L4, 0.25–0.38 s |
| Indic-Transcribe-core (teacher) | 9.3% / 13.8% | 2.9% / 9.8% | 5.3% | L4, 0.54–0.72 s |

On the held-out training evals it gets Kathbath-valid 4.87% CER and Lahaja 7.63% CER (greedy).

Timing caveat: the other models were timed one clip at a time on a cloud L4, including reading the file; this model
was timed on the laptop from audio already in memory.

**Limitations:**
- It is weakest on casual speech from a laptop mic: 25% CER on the author's own clips, with near-miss spellings
  such as साथ/सात.
- It has no punctuation and no digits.
- It is Hindi only; English words come out in Devanagari.
- It has not been tested on noisy, far-field or telephone audio.

## How it was trained

The full code, logs and the night-by-night journal are on GitHub:
[neoandmatrix/gemma-stt-hindi](https://github.com/neoandmatrix/gemma-stt-hindi).

1. **Audio.** 1,810 h of Hindi YouTube audio from YODAS v3 (CC-BY-3.0), cut into clips of 12 s or less.
2. **Labels (distillation).** YouTube's captions are too noisy to train on. Instead, Hima-ASR relabelled 874 h
   (780 h kept after filtering digits, Latin text and implausible speaking rates), and Indic-Transcribe-core (fp16)
   relabelled the other 970 h. That gives ~1,750 h of teacher transcripts.
3. **Model.**
   - Input: a log-mel spectrogram.
   - Encoder: the EmbeddingGemma 2 audio encoder. The bottom 10 layers are frozen and the top 2 are trained.
   - Mixing: a learned softmax mix of its 13 hidden states.
   - Reader: a Linear layer, a causal depthwise conv and 6 pre-norm transformer layers.
   - Head: a projection to 512 dimensions, multiplied with 66 character rows initialised from Gemma's text
     embeddings, plus blank and unk. Trained with CTC.
4. **Training.**
   - Hardware and length: one Kaggle T4 in fp16 with a GradScaler, for 605 minutes. That's 143k steps of 48 s
     batches, ~1.2 passes over the data.
   - Learning rates: 1e-3 for the reader and 5e-5 for the encoder layers, with 500 warm-up steps.
   - Checkpoint: the one with the best Kathbath-valid CER.
   - Earlier nights (on YouTube captions only, or on Hima labels only) reached 9.6–12% CER on Lahaja, so most of the
     gains came from better and more teacher labels.
5. **KenLM.** A word 3-gram LM built from the same teacher transcripts: 690k sentences, 17.6M words and a 261k-word
   vocabulary. It uses interpolated modified Kneser-Ney, and single-count trigrams are pruned with the back-off
   weights renormalised. alpha 0.6 and beta 1.0 were tuned on 800 dev clips (Kathbath-valid plus Lahaja clips
   outside the test subset).

## Files
- `model.safetensors`: the whole model in fp16. It includes the encoder, so the base model is not needed.
- `config.json`: the model config, the encoder's audio config, the character units and the LM settings.
- `preprocessor_config.json`: the log-mel settings, from EmbeddingGemma 2.
- `model.py`: the architecture.
- `gemma_stt.py`: inference.
- `kenlm/hi3.arpa`, `kenlm/unigrams.txt`: the LM.

## Licence
- **Licence:** Indic Open Model License v1.0, see `LICENSE`.
- **What it allows:** free use for research and commercial products, including self-hosting.
- **What needs approval:** hosting it as an API that third parties call directly needs Bodhan AI's prior written
  approval (Section 3 of the licence).
- **Restrictions and terms:** the prohibited uses in Section 10 apply. The Gemma Terms of Use restrictions also
  apply through Hima-ASR; `NOTICE` lists every upstream part and its terms.
- **Endorsement:** this model is not endorsed by Bodhan AI, AI4Bharat, IIT Madras, Google or the Hima-ASR authors.
