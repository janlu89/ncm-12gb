r"""Diagnosi del tokenizer ufficiale: norme dei codebook per livello, norma dell'input del livello 0,
stabilità della codifica al variare del batch. Sola lettura, nessun training.

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-17\diagnosi_tokenizer_ufficiale.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import load_valid_rows  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

art = os.path.join(PROJECT_ROOT, "artifacts", "tinystories")
cfg = OmegaConf.load(os.path.join(art, "config.yaml"))
cfg.tokenizer.vqvae.checkpoint_path = os.path.join(art, cfg.checkpoints.tokenizer)
cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(art, cfg.checkpoints.bpe_vocab)
cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(art, cfg.checkpoints.bpe_merges)
vqvae = VQVAETokenizer.from_config(cfg.tokenizer).vqvae.cuda().eval()
vq = vqvae.vq
rows = load_valid_rows(os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), 128, 1024).cuda()
out = {"levels": list(vq.levels)}
with torch.no_grad():
    z = vqvae._encode_pre_quant(rows[:256]).float()
    out["norma_z_per_token"] = float(z.norm(dim=-1).mean())
    x0 = vq._down_for_quant(z, 0, vq.levels[0])
    out["norma_input_livello0"] = {"media": float(x0.norm(dim=-1).mean()), "max": float(x0.norm(dim=-1).max())}
    out["norme_codebook"] = []
    for i, emb in enumerate(vq.embeddings):
        n = emb.weight.float().norm(dim=-1)
        cs = getattr(vq, f"ema_cluster_size_{i}")
        out["norme_codebook"].append({"livello": i, "mediana": float(n.median()), "max": float(n.max()),
                                      "min": float(n.min()), "codici_con_ema_sotto_1e-3": int((cs < 1e-3).sum())})
    a = torch.cat([vqvae.encode(rows[i:i + 256]).view(-1, sum(vq.levels)) for i in range(0, 1024, 256)])
    b = torch.cat([vqvae.encode(rows[i:i + 16]).view(-1, sum(vq.levels)) for i in range(0, 1024, 16)])
    out["codici_diversi_batch256_vs_16"] = float((a != b).float().mean())
    o, per = 0, []
    for lv in vq.levels:
        per.append(round(float((a[:, o:o + lv] != b[:, o:o + lv]).float().mean()), 4))
        o += lv
    out["codici_diversi_per_livello"] = per
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnosi_tokenizer_ufficiale.json")
json.dump(out, open(p, "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=1))
