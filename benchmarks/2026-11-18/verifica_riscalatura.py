r"""Prova diretta sul tokenizer ufficiale, senza training: se si riscala il percorso del livello 0 in modo
matematicamente equivalente (uscita della riduzione e codici divisi per s, ingresso della salita moltiplicato per s),
l'instabilità in fp32 sparisce? Se sì, la causa è la scala ~1e11 del livello 0.

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-18\verifica_riscalatura.py
"""
import json, os, sys
QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)
import torch
from verifica_fix import DEV, RAW, a_blocchi, carica, per_livello
from common.data import load_valid_rows

rows = load_valid_rows(RAW, 128, 1024).to(DEV)
m = carica("artifacts/tinystories/config.yaml")
v = m.vq
lv = list(v.levels)
print("riduzione:", [type(x).__name__ for x in v.learned_down_l1], "| salita:", [type(x).__name__ for x in v.learned_up_l1])
with torch.no_grad():
    z = m._encode_pre_quant(rows[:256]).float()
    x0 = v._down_for_quant(z, 0, lv[0])
    s = float(x0.norm(dim=-1).mean()) / 19.6
    print(f"norma input livello 0: {float(x0.norm(dim=-1).mean()):.3g}; fattore di scala s = {s:.3g}")
    pesi_salita = [float(c.weight.abs().max()) for c in v.learned_up_l1]
    pesi_riduz = [float(c.weight.abs().max()) for c in v.learned_down_l1]
    print("peso massimo per strato — riduzione:", ["%.3g" % p for p in pesi_riduz], "salita:", ["%.3g" % p for p in pesi_salita])
    a32, b32 = a_blocchi(m, rows, 256), a_blocchi(m, rows, 16)
    m64 = m.double()
    vero = a_blocchi(m64, rows, 256)
    m = m64.float()
    ultimo, primo = v.learned_down_l1[-1], v.learned_up_l1[0]
    ultimo.weight.div_(s); ultimo.bias.div_(s)
    v.embeddings[0].weight.div_(s)
    primo.weight.mul_(s)
    x0r = v._down_for_quant(m._encode_pre_quant(rows[:256]).float(), 0, lv[0])
    ar, br = a_blocchi(m, rows, 256), a_blocchi(m, rows, 16)
    res = {"s": s, "norma_input_l0_dopo": float(x0r.norm(dim=-1).mean()),
           "originale_fp32_256_vs_16": float((a32 != b32).float().mean()),
           "originale_fp32_vs_fp64": float((a32 != vero).float().mean()),
           "riscalato_fp32_256_vs_16": float((ar != br).float().mean()),
           "riscalato_fp32_vs_fp64_originale": float((ar != vero).float().mean()),
           "per_livello_riscalato_256_vs_16": per_livello(ar, br, lv),
           "per_livello_riscalato_vs_fp64": per_livello(ar, vero, lv)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(QUI, "verifica_riscalatura.json"), "w", encoding="utf-8"), indent=2)
