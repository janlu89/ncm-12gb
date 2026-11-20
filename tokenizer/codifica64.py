# Portions of this file are adapted from models/quant.py of https://github.com/HazyResearch/ncm
# (MIT License, Copyright (c) 2025 hermann kumbong). The upstream licence text is in third_party/NCM_LICENSE.
"""Codifica "esatta" (a 64 bit) con un tokenizer già addestrato.

Il tokenizer ufficiale a 32 bit sbaglia il 13,6% dei codici per una perdita di precisione nel percorso del livello 0
(docs/esperimenti/2026-11-18_verifica-correzione-tokenizer.md). Qui le stesse operazioni di `vq.z_to_idx` vengono
fatte nel tipo numerico del modello: passando una copia del tokenizer convertita con `.double()` si ottengono i codici
esatti, che non dipendono dalla dimensione del batch. (La funzione upstream converte sempre l'input a 32 bit.)

    vqvae64 = copy.deepcopy(vqvae).double()
    idx = codifica_fp64(vqvae64, tokens)        # [B, somma dei livelli]
"""
import torch


@torch.no_grad()
def codifica_fp64(m, x):
    v = m.vq
    z = m._encode_pre_quant(x)
    B, L, C = z.shape
    z_rest, K, out = z.clone(), len(v.levels), []
    for i, lv in enumerate(v.levels):
        e = v.embeddings[i].weight
        flat = (v._down_for_quant(z_rest, i, lv) if i != K - 1 else z_rest).reshape(-1, C)
        d = (flat * flat).sum(1, keepdim=True) + (e * e).sum(1) - 2.0 * (flat @ e.t())
        idx = d.argmin(1).view(B, lv)
        out.append(idx)
        q = v._up_to_full(e[idx], i, L) if i != K - 1 else e[idx]
        z_rest = z_rest - v._apply_quant_resi(q, i)
    return torch.cat(out, 1)
