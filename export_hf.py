"""Export a trained run as a standalone Hugging Face folder (bringagi/gemma-stt-hindi):
one model.safetensors with the whole model (the frozen EmbeddingGemma 2 audio encoder + our trained top layers,
reader and head; fp16), config.json (our config + the encoder's audio config + the character units + LM settings),
preprocessor_config.json (log-mel settings), model.py + gemma_stt.py (inference), the KenLM 3-gram, and the card.
Then checks that the exported folder gives the same transcripts as the training checkpoint.

  python export_hf.py kaggle/night7/code/runs/n7_all_k2 hf_out
"""
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from transformers import AutoFeatureExtractor

from data import Clips, load_units, loader
from model import ENCODER_ID, Config, GemmaCTC, greedy_decode, load_audio_encoder

HERE = Path(__file__).parent
run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

mc = json.loads((run / "config.json").read_text())["model"]
mc.pop("grad_ckpt", None)
cfg = Config(**{**mc, "amp": "fp16"})
encoder, _ = load_audio_encoder(cfg.enc_dtype)
model = GemmaCTC(cfg, encoder, None).eval()
model.load_state_dict(torch.load(run / "best.pt", map_location="cpu")["model"], strict=False)
state = {k: v.detach().to(torch.float16).contiguous() for k, v in model.state_dict().items()}
save_file(state, out / "model.safetensors", metadata={"format": "pt", "source_run": run.resolve().name})

chars = json.loads((run / "units.json").read_text())["chars"]
grid = json.loads((HERE.parent / "05-model-shootout/results/stt" / f"gemma-{run.name}-lm/_meta.json").read_text())
conf = {"model": {k: v for k, v in cfg.to_dict().items() if k != "amp"},
        "audio_config": encoder.config.to_dict(),
        "units": chars, "blank_id": 0, "unk_id": 1, "sample_rate": 16000,
        "lm": {"file": "kenlm/hi3.arpa", "alpha": grid["alpha"], "beta": grid["beta"], "beam": grid["beam"]}}
(out / "config.json").write_text(json.dumps(conf, indent=1, ensure_ascii=False, default=str))
AutoFeatureExtractor.from_pretrained(ENCODER_ID).save_pretrained(out)
shutil.copy(HERE / "model.py", out / "model.py")
for f in ("gemma_stt.py", "README.md", "LICENSE", "NOTICE"):
    shutil.copy(HERE / "hf_card" / f, out / f)
(out / "kenlm").mkdir(exist_ok=True)
for f in ("hi3.arpa", "unigrams.txt"):
    shutil.copy(HERE / "lm" / f, out / "kenlm" / f)
print("exported", sum(v.numel() for v in state.values()) / 1e6, "M params", flush=True)

# same transcripts from the exported folder as from the checkpoint? (Lahaja shoot-out set, greedy)
del model, encoder
sys.path.insert(0, str(out))
from gemma_stt import GemmaSTT  # noqa: E402

stt = GemmaSTT(out)
path = HERE / "kaggle/mytest_ds/lahaja.parquet"
clips = Clips([path], max_seconds=60)
ids = pq.read_table(path, columns=["id"]).column("id").to_pylist()
ref = {json.loads(x)["id"]: json.loads(x)["hyp"]
       for x in open(HERE.parent / "05-model-shootout/results/stt" / f"gemma-{run.name}/lahaja.jsonl", encoding="utf-8")}
same = total = 0
for i in range(len(clips)):
    hyp = stt.transcribe(clips.audio(i))
    same += hyp == ref[ids[clips.idx[i]]]
    total += 1
print(f"exported model vs checkpoint (Lahaja, one clip at a time): {same}/{total} identical transcripts")
