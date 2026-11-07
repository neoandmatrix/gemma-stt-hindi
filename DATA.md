# Hindi / Indian-English speech data (researched 2026-10-27 via the HF API)

Hours marked ~ are my estimates from file sizes (no duration totals in the cards). Verify once downloaded.

## Training candidates
| Dataset | Hindi size | Style | License / access | Notes |
|---|---|---|---|---|
| **ai4bharat/Kathbath** (`hindi/`) | 15.4 GiB, 66,865 train + 2,905 valid utts, 32 train shards (~0.45 GiB each), 2 valid shards | Read speech, many speakers | CC-BY-4.0, gated (auto-approve) | **Measured: shard 0 = 3.97 h, 2,868 clips, 2–12 s (mean 5 s), 16 kHz FLAC in parquet** (`audio_filepath.bytes`), so ≈127 h total. **Best for the ladder** |
| **ai4bharat/IndicVoices** (`hindi/`) | 46.4 GiB, 83 shards | **Spontaneous**, conversational, many districts | CC-BY-4.0, gated (auto) | Closest to real talking; has `verbatim` + `normalized` text; next step after Kathbath |
| ai4bharat/Shrutilipi (`hindi/`) | 88.3 GiB, 196 shards | All India Radio news (read, formal) | CC-BY-4.0, gated (auto) | Lots of hours, but a narrow formal domain |
| ARTPARK-IISc/Vaani-transcription-part (`audio/Hindi/`) | 107 GiB | Image-prompted spontaneous speech, many districts | CC-BY-4.0, gated (auto) | Big and diverse; later |
| espnet/yodas3 (`data/hi/`) | **3.3 TiB**, 836 tars | YouTube (web-crawled), naturally Hinglish | CC-BY-3.0, open | Huge; noisy transcripts; stream a few tars at most |
| google/fleurs (`hi_in`) | 1.5 GiB | Read Wikipedia sentences | CC-BY-4.0, open | Small (~10 h?); mainly for its test split |

## Test sets (never train on these)
| Dataset | What | Size | License / access |
|---|---|---|---|
| **ai4bharat/Lahaja** | Hindi, **multi-accent** (132 speakers, 83 districts) | 12.5 h, 6,152 utts | CC-BY-4.0 (card says MIT in metadata), gated (auto) |
| **VoiceArena/MonsoonASR …-hi-IN** | Hindi, real phone audio; `lattice` field lists spelling variants **including romanized English** (Hinglish-aware scoring) | 753 utts (0.1 GiB) | CC-BY-4.0, open |
| **ai4bharat/Svarah** | **Indian-accented English** (117 speakers, 19 states) | 9.6 h, 6,656 utts | CC-BY-4.0, gated (auto) |
| VoiceArena/MonsoonASR …-en-IN | Indian English, phone audio | 2,102 utts (1.5 GiB) | CC-BY-4.0, open |
| google/fleurs `hi_in` test | Clean read Hindi | small | CC-BY-4.0, open |
| My own clips | Hindi + Hinglish + Indian English | ~20 clips | – |

## Script note
These Hindi datasets write English words **in Devanagari** ("इंप्रूव", not "improve"), so a model trained on them will output Hinglish in Devanagari. Monsoon's lattice accepts either spelling.

## YODAS v3 Hindi (espnet/yodas3, CC-BY-3.0): pretraining candidate
- 69k h of Hindi YouTube; 418 shards × 8.5 GB tar (≈167 h of video each) + 418 metadata parquets (~11 MB each, transcripts only).
- One row per whole video; transcript = YouTube **auto-captions** (word timestamps, `[संगीत]` tags), almost all Devanagari,
  English words transliterated (गाइस, डाउनलोड). `locale_lang` is the channel's locale, mostly not `hi`; not a filter.
- `yodas_prep.py` streams a tar (curl | python, no disk), cuts 2–12 s clips at caption pauses, keeps plain-Devanagari text at
  5–25 chars/s, writes 16 kHz mono **Opus** (12.7 MB/h; FLAC would be 77 MB/h). On the first 200 videos of shard 0000:
  kept 20.7 h, dropped 6.8 h (digits/Latin/tags), 3.2 h (speaking rate), 0.8 h (< 2 s) → ~66% kept, so ~100 usable h per shard.
- Some captions are garbage (wrong language, e.g. Tamil captioned as Hindi): plan an agreement filter with IndicConformer.
