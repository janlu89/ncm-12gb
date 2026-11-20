r"""Data for the coarse-to-fine animation: one validation story, decoded using only the first k levels of the code
hierarchy (k = 1..12), with the official tokenizer and with ours (38,400 steps). No training.

Run from the repository root:  python -I figures/coarse_to_fine_data.py   -> figures/coarse_to_fine.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import EOS_ID, PAD_ID, load_valid_rows  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

TOKENIZERS = {
    "official": "artifacts/tinystories/config.yaml",
    "ours (38,400 steps)": "runs/mytok-12std-dati-rv_9830400_38400x256_s42/export/config.yaml",
}


def load(base_config):
    bc = os.path.join(PROJECT_ROOT, base_config)
    cfg = OmegaConf.load(bc)
    d = os.path.dirname(bc)
    t = cfg.tokenizer
    t.vqvae.checkpoint_path = os.path.join(d, cfg.checkpoints.tokenizer)
    t.pre_tokenizer.vocab_file_path = os.path.join(d, cfg.checkpoints.bpe_vocab)
    t.pre_tokenizer.merges_file_path = os.path.join(d, cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(t)
    tk.vqvae.cuda().eval()
    return tk


@torch.no_grad()
def decode_first_levels(m, idx, k):
    """Sum of the contributions of levels 0..k-1 only, then the decoder (same path as full decoding)."""
    v, L, K = m.vq, m.vq.levels[-1], len(m.vq.levels)
    z, o = 0, 0
    for i, l in enumerate(v.levels):
        if i < k:
            e = v.embeddings[i].weight[idx[:, o:o + l]]
            z = z + v._apply_quant_resi(v._up_to_full(e, i, L) if i != K - 1 else e, i)
        o += l
    return m.decoder(z).argmax(-1)


def text(tk, ids):
    ids = [t for t in ids if t != PAD_ID]
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    return tk.pre_tokenizer.decode(ids, skip_special_tokens=True).strip()


rows = load_valid_rows(os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), 128, 4096).cuda()
tks = {n: load(bc) for n, bc in TOKENIZERS.items()}
codes = {}
for n, tk in tks.items():
    lv = list(tk.vqvae.vq.levels)
    codes[n] = torch.cat([tk.vqvae.encode(rows[i:i + 256]).view(-1, sum(lv)) for i in range(0, len(rows), 256)])

# shortest story (at least 40 real tokens) that both tokenizers reconstruct perfectly
lengths = (rows != PAD_ID).sum(1)
chosen = None
for i in sorted(range(len(rows)), key=lambda j: int(lengths[j])):
    if int(lengths[i]) < 40:
        continue
    if all(bool((decode_first_levels(tk.vqvae, codes[n][i:i + 1], 12)[0] == rows[i]).all()) for n, tk in tks.items()):
        chosen = i
        break
assert chosen is not None
print("story", chosen, "real tokens", int(lengths[chosen]), flush=True)
out = {"story_index": chosen, "original": text(tks["official"], rows[chosen].tolist()), "frames": {}}
for n, tk in tks.items():
    out["frames"][n] = []
    for k in range(1, 13):
        pred = decode_first_levels(tk.vqvae, codes[n][chosen:chosen + 1], k)[0]
        mask = rows[chosen] != PAD_ID
        acc = float(((pred == rows[chosen]) & mask).sum() / mask.sum())
        out["frames"][n].append({"levels": k, "token_accuracy": round(acc, 4), "text": text(tk, pred.tolist())})
        print(f"{n:20s} levels 0..{k - 1:2d}: {acc:.3f}  {text(tk, pred.tolist())[:90]}", flush=True)
with open(os.path.join(ROOT, "figures", "coarse_to_fine.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
