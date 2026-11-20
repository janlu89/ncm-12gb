"""Analisi in sola lettura dei file TinyStories tokenizzati."""
import os, sys, json
import numpy as np
from omegaconf import OmegaConf

root = sys.argv[1]
raw = os.path.join(root, "data", "raw", "tinystories")
sys.path.insert(0, os.path.join(root, "third_party", "ncm"))
from tokenization.bpe_tokenizer import BPETokenizer

cfg = OmegaConf.load(os.path.join(root, "artifacts", "tinystories", "config.yaml")).tokenizer.pre_tokenizer
cfg.vocab_file_path = os.path.join(root, "artifacts", "tinystories", "bpe_vocab_8192.pickle")
cfg.merges_file_path = os.path.join(root, "artifacts", "tinystories", "bpe_merges_8192.pickle")
tok = BPETokenizer.from_config(cfg)
EOS, PAD = tok.eos_token_id, tok.pad_token_id
print("EOS id", EOS, "| PAD id", PAD)

train_bytes = os.path.getsize(os.path.join(raw, "train_bpe_8192.bin"))
off = np.load(os.path.join(raw, "train_bpe_8192_offsets.npy"), allow_pickle=False)
print("offsets: dtype", off.dtype, "shape", off.shape, "primi", off[:5].tolist(), "ultimo", int(off[-1]))
for dt in (np.uint16, np.int32):
    n = train_bytes // np.dtype(dt).itemsize
    print(f"se dtype={np.dtype(dt).name}: {n:,} token; ultimo offset {'<= n' if off[-1] <= n else '> n'}")

dt = np.uint16 if off[-1] <= train_bytes // 2 and off[-1] > train_bytes // 4 else np.int32
train = np.memmap(os.path.join(raw, "train_bpe_8192.bin"), dtype=dt, mode="r")
valid = np.memmap(os.path.join(raw, "valid_bpe_8192.bin"), dtype=dt, mode="r")
print("dtype scelto:", np.dtype(dt).name, "| token train", f"{len(train):,}", "| token valid", f"{len(valid):,}")
print("max id train (primi 50M):", int(train[:50_000_000].max()), "| max id valid:", int(valid.max()))

bounds = np.append(off, len(train)) if off[-1] < len(train) else off
lens = np.diff(bounds)
print(f"storie train: {len(lens):,}")
print("primo token di una storia (prime 10):", [int(train[o]) for o in off[:10]])
print("ultimo token di una storia (prime 10):", [int(train[o - 1]) for o in off[1:11]])
print("EOS nel train (primi 50M token):", int((train[:50_000_000] == EOS).sum()), "| PAD:", int((train[:50_000_000] == PAD).sum()))
pct = np.percentile(lens, [5, 25, 50, 75, 90, 95, 99])
print("lunghezza storie (token) percentili 5/25/50/75/90/95/99:", [int(x) for x in pct], "| media", round(float(lens.mean()), 1), "| max", int(lens.max()))
for L in (128, 256, 384, 512):
    print(f"  storie <= {L} token: {(lens <= L).mean()*100:.1f}%")

ve = np.flatnonzero(np.asarray(valid) == EOS)
print(f"EOS nel valid: {len(ve):,}", "| PAD nel valid:", int((np.asarray(valid) == PAD).sum()))
if len(ve) > 1:
    vl = np.diff(np.concatenate([[-1], ve]))
    print("lunghezze storie valid (con EOS) mediana", int(np.median(vl)), "| <=128:", f"{(vl <= 128).mean()*100:.1f}%")

print("\n--- storia 0 (decodificata) ---")
print(tok.decode([int(x) for x in train[bounds[0]:bounds[1]]]))
print("\n--- storia 1, primi 128 token ---")
print(tok.decode([int(x) for x in train[bounds[1]:bounds[1] + 128]]))
