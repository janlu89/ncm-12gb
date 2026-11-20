r"""Tokenizer ufficiale: ricostruzione del testo con codici calcolati a 32 o a 64 bit, e decodifica a 32 o a 64 bit.
Nessun training. Dice se i codici "esatti" si accordano con la decodifica a 32 bit usata nella valutazione.

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-19\ricostruzione_fp64.py
"""
import copy
import json
import os
import sys

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(QUI)))
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import PAD_ID, load_valid_rows  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402
from tokenizer.codifica64 import codifica_fp64  # noqa: E402

art = os.path.join(PROJECT_ROOT, "artifacts", "tinystories")
cfg = OmegaConf.load(os.path.join(art, "config.yaml"))
cfg.tokenizer.vqvae.checkpoint_path = os.path.join(art, cfg.checkpoints.tokenizer)
cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(art, cfg.checkpoints.bpe_vocab)
cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(art, cfg.checkpoints.bpe_merges)
m32 = VQVAETokenizer.from_config(cfg.tokenizer).vqvae.cuda().eval()
m64 = copy.deepcopy(m32).double()
rows = load_valid_rows(os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), 128, 4096).cuda()
lv = list(m32.vq.levels)


@torch.no_grad()
def decodifica(m, idx):
    """Somma dei contributi dei livelli nel tipo numerico di m (senza operazioni in place), poi decoder."""
    v, L, K = m.vq, m.vq.levels[-1], len(m.vq.levels)
    z, o = 0, 0
    for i, l in enumerate(v.levels):
        e = v.embeddings[i].weight[idx[:, o:o + l]]
        z = z + v._apply_quant_resi(v._up_to_full(e, i, L) if i != K - 1 else e, i)
        o += l
    return m.decoder(z).argmax(-1)


res, cod = {}, {}
with torch.no_grad():
    for nome, bs in (("32bit_batch256", 256), ("32bit_batch16", 16)):
        cod[nome] = torch.cat([m32.encode(rows[i:i + bs]).view(-1, sum(lv)) for i in range(0, len(rows), bs)])
    cod["64bit"] = torch.cat([codifica_fp64(m64, rows[i:i + 128]) for i in range(0, len(rows), 128)])
    for nc, idx in cod.items():
        for nd, md in (("dec32", m32), ("dec64", m64)):
            pred = torch.cat([decodifica(md, idx[i:i + 128]) for i in range(0, len(rows), 128)])
            ok = pred == rows
            nopad = rows != PAD_ID
            res[f"{nc}+{nd}"] = {"token": round(float(ok.float().mean()), 5),
                                 "token_senza_pad": round(float((ok & nopad).sum() / nopad.sum()), 5),
                                 "storie_perfette": round(float(ok.all(1).float().mean()), 4)}
            print(f"codici {nc:15s} decodifica {nd}: {res[f'{nc}+{nd}']}", flush=True)
    res["codici_diversi_64_vs_32_batch256"] = round(float((cod["64bit"] != cod["32bit_batch256"]).float().mean()), 4)
    usati = lambda idx: [int(idx[:, sum(lv[:k]):sum(lv[:k + 1])].unique().numel()) for k in range(4)]  # noqa: E731
    res["codici_distinti_livelli_0_3"] = {k: usati(v) for k, v in cod.items()}
print(json.dumps({k: v for k, v in res.items() if not k.startswith(("32bit", "64bit"))}))
json.dump(res, open(os.path.join(QUI, "ricostruzione_fp64.json"), "w", encoding="utf-8"), indent=2)
