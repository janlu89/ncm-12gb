"""Verifica dei codici pre-calcolati (T5). Aspetta che data/build_codes.py abbia finito, poi:
- ricodifica 8 blocchi allineati da `encode_batch` righe (stesso raggruppamento usato per il calcolo) e
  controlla che siano identici a quelli su disco;
- controlla che tutti i codici siano < dimensione del codebook e che non ci siano righe mai scritte (tutte 0).
Se tutto va bene scrive <nome>.verificato.json (il ciclo T6 aspetta questo file).

Uso: .venv\\Scripts\\python.exe -u -I data\\verify_codes.py [--name tinystories_codes_official_L128]
"""
import argparse
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
ap.add_argument("--name", default="tinystories_codes_official_L128")
a = ap.parse_args()
outd = os.path.join(PROJECT_ROOT, "data", "processed")
meta_path = os.path.join(outd, f"{a.name}.json")
codes_path = os.path.join(outd, f"{a.name}.u16")
ok_path = os.path.join(outd, f"{a.name}.verificato.json")

while not os.path.exists(meta_path):
    print(f"[verifica] {time.strftime('%H:%M')} in attesa di {meta_path}", flush=True)
    time.sleep(30)

meta = json.load(open(meta_path, encoding="utf-8"))
N, W, B, L = meta["rows"], meta["codes_per_row"], meta["encode_batch"], meta["length"]
codes = np.memmap(codes_path, dtype=np.uint16, mode="r", shape=(N, W))
assert os.path.getsize(codes_path) == N * W * 2, "dimensione del file diversa da righe × codici × 2"

base = OmegaConf.load(os.path.join(PROJECT_ROOT, "artifacts", "tinystories", "config.yaml"))
art = os.path.join(PROJECT_ROOT, "artifacts", "tinystories")
base.tokenizer.vqvae.checkpoint_path = os.path.join(PROJECT_ROOT, meta["tokenizer"])
base.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(art, base.checkpoints.bpe_vocab)
base.tokenizer.pre_tokenizer.merges_file_path = os.path.join(art, base.checkpoints.bpe_merges)
vqvae = VQVAETokenizer.from_config(base.tokenizer).vqvae.cuda().eval()
sizes = vqvae.vq.vqvae_vocab_sizes
levels = vqvae.vq.levels

raw = os.path.join(PROJECT_ROOT, "data", "raw", "tinystories")
stream = np.memmap(os.path.join(raw, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
off = np.load(os.path.join(raw, "train_bpe_8192_offsets.npy"), allow_pickle=False)
starts = off if off[-1] < len(stream) else off[:-1]
ends = np.append(starts[1:], len(stream))

res = {"file": os.path.basename(codes_path), "blocchi": [], "data": time.strftime("%Y-%m-%d %H:%M:%S")}
n_blocks = (N + B - 1) // B
for blk in sorted(set([0, 1, n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4, n_blocks - 2, n_blocks - 1, 12345 % n_blocks])):
    b0, b1 = blk * B, min(blk * B + B, N)
    rows = np.full((b1 - b0, L), PAD_ID, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts[b0:b1], ends[b0:b1])):
        seg = stream[s:min(e, s + L)]
        rows[i, :len(seg)] = seg
    x = torch.from_numpy(rows).cuda()
    if b1 - b0 < B:
        x = torch.cat([x, x[-1:].expand(B - (b1 - b0), -1)])
    with torch.no_grad():
        idx = vqvae.encode(x).view(x.shape[0], -1)[: b1 - b0].cpu().numpy()
    diff = int((idx != codes[b0:b1].astype(np.int64)).sum())
    res["blocchi"].append({"blocco": blk, "righe": [b0, b1], "codici_diversi": diff})
    print(f"[verifica] blocco {blk} (righe {b0}-{b1}): codici diversi {diff}", flush=True)

# range per livello e righe non scritte, a pezzi per non caricare 3,5 GB in RAM
max_per_level = np.zeros(len(levels), dtype=np.int64)
zero_rows = 0
bounds = np.cumsum([0] + list(levels))
for c0 in range(0, N, 131072):
    chunk = np.asarray(codes[c0:c0 + 131072])
    for k in range(len(levels)):
        max_per_level[k] = max(max_per_level[k], int(chunk[:, bounds[k]:bounds[k + 1]].max()))
    zero_rows += int((chunk == 0).all(axis=1).sum())
res["max_codice_per_livello"] = max_per_level.tolist()
res["righe_tutte_zero"] = zero_rows
ok = all(b["codici_diversi"] == 0 for b in res["blocchi"]) and all(
    int(m) < int(sz) for m, sz in zip(max_per_level, sizes)) and zero_rows == 0
res["esito"] = "superato" if ok else "fallito"
print(f"[verifica] max per livello {res['max_codice_per_livello']} | righe tutte zero {zero_rows} | esito {res['esito']}",
      flush=True)
if ok:
    json.dump(res, open(ok_path, "w", encoding="utf-8"), indent=2)
    print(f"[verifica] scritto {ok_path}", flush=True)
else:
    json.dump(res, open(ok_path.replace(".verificato.", ".verifica-fallita."), "w", encoding="utf-8"), indent=2)
    sys.exit(1)
