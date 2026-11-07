# Journal: Gemma Hindi STT

Copied from the project journal (blog/JOURNAL.md in the parent folder), only the entries about this model,
in the order they were written. Numbers are as measured at the time.

## 2026-10-27 · 04 Hindi speech-to-text on EmbeddingGemma 2 (the main project)
Idea from a bookmark: someone turned Google's *embedding* model into speech-to-text, and a Vietnamese fine-tune beat Whisper-large (QUAN-Transcribe). Can I do Hindi on a laptop?
- **Training fits on 4 GB:** frozen encoder 1.9 GB at ~170 s of audio per second; top-2 layers trained at 3.0 GB.
- **Overfit check (50 clips): 1.4% CER**, so the pipeline is correct.
- **4 h of Kathbath Hindi, frozen encoder:**
  - Gemma *word* tokens as outputs: **72% CER**. It only learned the most frequent words (इस, के, को, किया).
  - **Characters** as outputs: **12% CER**. Same data, same 15 minutes.
- **Lesson:** with little data, use small output units. Word-level units starve: 6.7k units, most seen only a few times.
- Accents (Lahaja, 4 h model): **22% CER**. **My own voice: 45% CER** (read vs natural speech, phone vs laptop mic, tiny data).
- **fp16 for Kaggle T4s:** Gemma fills masked attention with −1e9, which overflows fp16. Replacing it with −1e4 (logits are soft-capped at ±50) gives identical results (11.92% vs 11.93% CER).
- Kaggle (2× T4, free): frozen and top-2 run side by side at 20 h and 50 h. T4 ≈ 3–4× slower than the laptop 3050 for this. 4.8 h wall time, ₹0.
  | run | Kathbath CER / WER | Lahaja CER | my voice CER |
  |---|---|---|---|
  | 4 h frozen | 11.9% / 40.6% | 22.3% | 45.2% |
  | 20 h frozen | 7.5% / 26.6% | 16.4% | 38.2% |
  | 20 h top-2 | 7.0% / 24.7% | 15.8% | 40.5% |
  | 50 h top-2 | **6.3% / 22.2%** | 16.3% | 40.2% |
  | 50 h frozen | killed (exit −9, likely host RAM) | – | – |
- **Lesson 2: more of the same data stops transferring.** 20 h → 50 h still helps on Kathbath, but Lahaja (other accents) and my own voice stay flat (~16% and ~40%). The next gain needs *different* data (conversational, Hinglish, laptop/phone mics), not more read speech.
- Context: off-the-shelf Qwen3-ASR and Indic-Transcribe are near-perfect on my clips (see 05). A ₹0, 50 h home-trained model is a fun result but not a usable product yet.
- **Blog angle:** "Teaching an embedding model to transcribe Hindi on a ₹0 budget."

## 2026-10-28/29 · 04b Gemma STT + 982 h of YouTube Hindi (YODAS v3)
- **Data:** YODAS v3 has 69k h of Hindi YouTube with *auto*-captions (CC-BY-3.0). `yodas_prep.py` cuts videos into 2–12 s clips at caption pauses and keeps plain-Devanagari text (~65% of speech). A free **CPU-only** Kaggle notebook converted 13 shards → **982 h** (10.6 GB of Opus) in 11 h, with zero GPU quota used.
- **Two runs on 2 free T4s (one 9.6 h session):** stage 1 YouTube only → stage 2 Kathbath 50 h fine-tune, vs everything mixed. About 1,500–1,660 h of audio seen per GPU (T4 top-2 trains at ~150–200 s of audio per second).
- **Results** (same clips as the shoot-out; CER / WER):

  | model | my voice | Lahaja | Monsoon hi (lattice WER) |
  |---|---|---|---|
  | night 1: Kathbath 50 h | 41.3% / 81% | 15.0% / 48% | 66.5% |
  | YouTube only | 41.9% / 76% | 14.6% / 41% | 55.2% |
  | mixed | 38.4% / 68% | 14.3% / 33% | 45.0% |
  | **YouTube → Kathbath** | **29.2% / 62%** | **10.1% / 31%** | **42.6%** |
  | Indic-Transcribe (reference) | 9.3% / 14% | 2.9% / 10% | 5.3% |
- **Lesson 3: noisy-but-varied first, clean last.** Auto-captions alone don't beat clean read speech, but as a first stage they cut errors on accents, conversation and my voice by ~30%. Mixing the two is worse than doing them in order.
- **Still 3× worse than Indic-Transcribe** on my voice. ₹0 and 2 nights vs a purpose-built model.
- **Gotchas:** streaming 8.5 GB tars with curl silently truncated when the connection dropped (download with resume instead); libsndfile's Opus encoder ran at 23× real time (PyAV/libopus: 194×); DataLoader worker processes froze two parallel trainings twice (threads instead); parquet decodes fully into RAM (memory-mapped Arrow instead, ~4× less).

## 2026-11-01 · 04c Gemma STT night 4: 1,810 h of YODAS, top-2 vs top-4 layers
- Two T4s in parallel: YODAS-only stage 1 (~1,518 h seen, 0.84 epoch), then a 45 min Kathbath fine-tune.
- Results, Kathbath valid CER / WER, Lahaja CER: yt_kb_k2 3.9% / 13.6%, 11.1%; yt_kb_k4 3.7% / 13.1%, 11.3%. Night 3's yt_kb_k2 was 4.0% / 12.0% Lahaja.
- So doubling the data gained about 1 point on Lahaja. Training 4 layers instead of 2 helps a little on Kathbath but not on Lahaja. Returns from more noisy YouTube captions are flattening.
- Stage-1 validation CER bounced between 8% and 17% and ended around 11%, worse than its best at step 22k. The noisy auto-captions are pulling it away from read speech, so the clean fine-tune does most of the work.
- Waste: the epoch-based LR schedule assumed 150 s of audio per s, but the real speed was about 212. GPU0's stage 1 finished about 2 h early with the LR at its floor.
- Logs and checkpoints (k2 377 MB, k4 578 MB) are in 04-gemma-stt-hindi/kaggle/night4/.
- Shoot-out scores (CER on my clips / CER on Lahaja / lattice-WER on Monsoon hi): n4_yt_kb_k2 27.7% / 9.7% / 40.5%; n4_yt_kb_k4 26.3% / 9.8% / 40.2%. Night 3: 29.2% / 10.1% / 42.6%. Indic-Transcribe: 9.3% / 2.9% / 5.3%.

## 2026-11-06 · 04d Gemma STT night 5: relabel YouTube with Hima, then an A/B on the same clips
- Hima-ASR relabelled YODAS clips on both T4s for 4.5 h (fp16, ~4× faster than its bf16 default on a T4). 874 h went in; **780 h were kept** (303k clips). The rest were dropped: Hima's text had digits or Latin letters (~4.6%), or an implausible speaking rate (~7%).
- **The YouTube captions differ from Hima's text by 27% CER on average.** That's how noisy the auto-captions are.
- A/B with the same clips, same steps and same recipe (top-2 layers, then the 45 min Kathbath fine-tune). Columns: Kathbath valid CER / WER, then Lahaja CER.
  | run | stage 1 (best) | after Kathbath fine-tune |
  |---|---|---|
  | **Hima labels (hl)** | 5.7% / 21.0%, Lahaja 9.4% (final 8.8%) | **3.7% / 13.4%, Lahaja 9.6%** |
  | YouTube captions (yc) | 9.2% / 30.5%, Lahaja 16.0% | 4.2% / 14.8%, Lahaja 11.6% |
  | night 4 (1,810 h YouTube, for reference) | 8.5% / 28.7%, Lahaja 15.5% | 3.9% / 13.6%, Lahaja 11.1% |
- **780 h of Hima-labelled data beats 1,810 h of YouTube captions.** Label quality is worth more than 2× the hours.
- **Noisy labels hurt over time:** the YouTube run peaked at 154 h seen and got worse after that (final Kathbath 12.0%). The Hima run kept improving to the end.
- Stage 1 on Hima labels **without** the Kathbath fine-tune reaches Lahaja 8.8%, our best ever. The fine-tune helps Kathbath (read speech) and slightly hurts Lahaja.
- Checkpoints are not downloaded yet (377 MB each). Shoot-out scores (my clips, Monsoon) are pending.

## 2026-11-06 · 04e Night 6: Indic-Transcribe relabels the rest of YODAS (no training)
- `gemma-stt-hindi-n6` v3, relabel-only (the Kaggle quota was down to 8 h). 2xT4, batched fp16 Indic-Transcribe (probe: fp16 matches the one-clip reference, CER gap 0.000 / 0.001; 25-34 s ref vs 2.8-3.8 s fp16).
- **1,030 h in, 970 h kept** (388k clips) in 5.5 h, all 25 files finished, none timed out. ~111 s of audio per second per GPU. Dropped: 14k non-Devanagari (some clips came back partly in Telugu script), 15.6k implausible char rate, 23 empty.
- YouTube captions vs Indic-Transcribe: CER 0.27 / 0.26, the same ~27% gap Hima showed on night 5.
- Now every clip of the 1,810 h has a teacher label: Hima 780 h + Indic-Transcribe 970 h = **~1,750 h**. Output: `it_relabel_text.parquet` + 108 arrow files in the n6 notebook output (not downloaded).
- Next (after the quota reset): train all_k2/all_k4 -> Kathbath fine-tune on both label sets, against night 5's hl_kb_k2 (Lahaja 9.6%).

## 04f night 7: training on all ~1,750 h of teacher labels (2026-11-07)
Hima 780 h + Indic-Transcribe 970 h, 605 min per GPU, then a 45-min Kathbath fine-tune. All 4 runs exited cleanly
(663 min in total). CER, Kathbath valid / Lahaja:

| run | hours seen | Kathbath | Lahaja |
|---|---|---|---|
| night 5 hl_k2 (Hima only) | 772 | 5.72% | 9.40% |
| night 5 hl_kb_k2 (previous best) | +101 KB | 3.74% | 9.63% |
| **all_k2** | 1,842 | 4.87% | **7.63%** |
| all_k4 | 1,508 | 4.96% | 8.05% |
| **all_kb_k2** | +101 KB | 3.32% | 8.56% |
| **all_kb_k4** | +114 KB | **3.22%** | 9.05% |

- More labelled hours helped everywhere: Lahaja 9.4 -> 7.6% before the Kathbath step, Kathbath 3.7 -> 3.2–3.3% after it.
- The Kathbath fine-tune again trades Lahaja for Kathbath (7.6 -> 8.6–9.1%): it specialises on read speech. Next:
  a mixed fine-tune (Kathbath + general), or average all_k2 with all_kb_k2 weights.
- Indic-Transcribe on Lahaja is 2.9%, so there is still a large gap.

Scored on the shoot-out set (05-model-shootout, same clips and scorer as every other model):

| model | my 15 clips CER | Lahaja-100 CER | Monsoon-hi lattice-WER |
|---|---|---|---|
| night 4 n4_yt_kb_k4 (previous best) | 26.3% | 9.8% | 40.2% |
| **night 7 n7_all_k2** | 26.8% | **6.4%** | **28.9%** |
| night 7 n7_all_kb_k4 | **25.9%** | 7.9% | 35.6% |
| Qwen3-ASR-1.7B | 15.4% | 6.4% | 12.8% |
| Hima-ASR (one of the teachers) | 14.2% | 3.4% | 10.6% |
| Indic-Transcribe (the other teacher) | 9.3% | 2.9% | 5.3% |

- On Lahaja the student now ties Qwen3-ASR-1.7B (6.4%), and Monsoon WER dropped by 11 points.
- My own laptop-mic clips did not improve (~26%). The errors are near-miss spellings of common words
  (साथ for सात, होमनें for घूमने, शो for स्लो): the acoustics don't match the training audio, and nothing
  checks that the output is a real word. A language model in the decoder would target exactly these errors.

## 04g KenLM decoding for n7_all_k2 (2026-11-07)
`04-gemma-stt-hindi/lm/`: a word 3-gram LM from the teacher labels (690k transcripts, 17.6M words, vocab 261k).
lmplz needs Boost, so `build_arpa.py` does the same interpolated modified Kneser-Ney in numpy. It prunes
single-count trigrams and re-solves the back-off weights so every context sums to 1 (checked). It took 32 s.
Lahaja-text perplexity is 616. `decode_lm.py` runs pyctcdecode beam search (beam 64). alpha/beta were tuned on
800 dev clips (Kathbath valid + Lahaja shard 0 minus the shoot-out clips), giving alpha 0.6, beta 1.

| n7_all_k2 | my 15 clips CER / WER | Lahaja-100 CER / WER | Monsoon-hi lattice-WER |
|---|---|---|---|
| greedy | 26.8% / 53.1% | 6.4% / 21.3% | 28.9% |
| **+ 3-gram LM** | **25.3% / 40.6%** | **5.4% / 15.9%** | **20.9%** |
| Qwen3-ASR-1.7B | 15.4% / 23.1% | 6.4% / 17.4% | 12.8% |

- WER fell by 23–29% relative on all three sets, with no retraining; decoding is ~free (51 s for 3,300 clips).
- Lahaja CER (5.4%) is now better than Qwen3-ASR-1.7B (6.4%), and Lahaja WER (15.9%) is better too (17.4%).
- My own clips: WER 53 -> 41%, CER barely moves. The remaining errors there are acoustic.
- Gotcha: a multiprocessing fork pool started after CUDA/torch had run deadlocked silently (0% CPU). Spawn workers
  that take numpy arrays fixed it.
