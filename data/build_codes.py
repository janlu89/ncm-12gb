"""T5: calcola una volta i codici VQ del tokenizer congelato per tutte le storie di train e li salva su disco.

Output (in data/processed/):
    <nome>.u16   matrice [N storie, codici per storia] uint16, nello stesso ordine degli offset del train
    <nome>.json  metadati: hash del tokenizer, lunghezza, righe, batch di codifica, versioni

Scrive su <nome>.u16.tmp e rinomina alla fine. Se interrotto, si rilancia: riprende dall'ultimo blocco
completato (registrato in <nome>.progress.json).

Nota: con il tokenizer ufficiale la codifica dipende dalla dimensione del batch (norme del codebook del
livello 0 ~1e21 in fp32): il batch usato è fissato e registrato nei metadati.

Uso: .venv\\Scripts\\python.exe -u -I data\\build_codes.py [--batch 256] [--name tinystories_codes_official_L128]
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import PAD_ID  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--length", type=int, default=128)
ap.add_argument("--name", default="tinystories_codes_official_L128")
ap.add_argument("--tokenizer", default="artifacts/tinystories/tokenizer.pt")
ap.add_argument("--block", type=int, default=65536, help="righe tra un salvataggio dei progressi e l'altro")
a = ap.parse_args()

raw = os.path.join(PROJECT_ROOT, "data", "raw", "tinystories")
outd = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(outd, exist_ok=True)
final = os.path.join(outd, f"{a.name}.u16")
tmp, prog_path, meta_path = final + ".tmp", os.path.join(outd, f"{a.name}.progress.json"), \
    os.path.join(outd, f"{a.name}.json")
if os.path.exists(final):
    sys.exit(f"{final} esiste già: niente da fare")

tok_path = os.path.join(PROJECT_ROOT, a.tokenizer)
h = hashlib.sha256()
with open(tok_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
tok_sha = h.hexdigest()

base = OmegaConf.load(os.path.join(PROJECT_ROOT, "artifacts", "tinystories", "config.yaml"))
art = os.path.join(PROJECT_ROOT, "artifacts", "tinystories")
base.tokenizer.vqvae.checkpoint_path = tok_path
base.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(art, base.checkpoints.bpe_vocab)
base.tokenizer.pre_tokenizer.merges_file_path = os.path.join(art, base.checkpoints.bpe_merges)
vqvae = VQVAETokenizer.from_config(base.tokenizer).vqvae.cuda().eval()
n_codes = sum(vqvae.vq.levels)
assert max(vqvae.vq.vqvae_vocab_sizes) <= 65536

stream = np.memmap(os.path.join(raw, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
off = np.load(os.path.join(raw, "train_bpe_8192_offsets.npy"), allow_pickle=False)
starts = off if off[-1] < len(stream) else off[:-1]
ends = np.append(starts[1:], len(stream))
N = len(starts)

start_row = 0
if os.path.exists(prog_path) and os.path.exists(tmp):
    pr = json.load(open(prog_path))
    assert pr["tokenizer_sha256"] == tok_sha and pr["batch"] == a.batch and pr["rows"] == N
    start_row = int(pr["done"])
    out = np.memmap(tmp, dtype=np.uint16, mode="r+", shape=(N, n_codes))
    print(f"[codici] ripresa dalla riga {start_row:,}", flush=True)
else:
    out = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(N, n_codes))

t0, done = time.time(), start_row
with torch.no_grad():
    for b0 in range(start_row, N, a.batch):
        b1 = min(b0 + a.batch, N)
        rows = np.full((b1 - b0, a.length), PAD_ID, dtype=np.int64)
        for i, (s, e) in enumerate(zip(starts[b0:b1], ends[b0:b1])):
            seg = stream[s:min(e, s + a.length)]
            rows[i, :len(seg)] = seg
        x = torch.from_numpy(rows).cuda()
        if b1 - b0 < a.batch:  # ultimo batch: completato fino a `batch` per usare gli stessi kernel
            x = torch.cat([x, x[-1:].expand(a.batch - (b1 - b0), -1)])
        idx = vqvae.encode(x).view(x.shape[0], -1)[: b1 - b0]
        out[b0:b1] = idx.to(torch.int32).cpu().numpy().astype(np.uint16)
        done = b1
        if done % a.block < a.batch or done == N:
            out.flush()
            json.dump({"done": done, "rows": N, "batch": a.batch, "tokenizer_sha256": tok_sha}, open(prog_path, "w"))
            el = time.time() - t0
            rate = (done - start_row) / max(el, 1e-9)
            print(f"[codici] {done:,}/{N:,} ({100 * done / N:.1f}%)  {rate:,.0f} storie/s  "
                  f"restano ~{(N - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)

out.flush()
del out
os.replace(tmp, final)
os.remove(prog_path)
json.dump({"file": os.path.basename(final), "rows": N, "codes_per_row": n_codes, "dtype": "uint16",
           "order": "stesso ordine di train_bpe_8192_offsets.npy", "length": a.length, "pad_id": PAD_ID,
           "encode_batch": a.batch, "tokenizer": a.tokenizer, "tokenizer_sha256": tok_sha,
           "levels": list(vqvae.vq.levels), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "secondi": round(time.time() - t0, 1), "data": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(meta_path, "w", encoding="utf-8"), indent=2)
print(f"[codici] fatto: {final} ({os.path.getsize(final) / 1024 ** 3:.2f} GB)", flush=True)
