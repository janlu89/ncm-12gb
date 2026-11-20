r"""Quanto sono prevedibili i codici di un tokenizer conoscendo solo la loro frequenza, livello per livello,
e quanto i generatori addestrati sopra hanno imparato oltre quella frequenza. Nessun training.

Per ogni tokenizer: codifica le 4.096 storie di validazione (le stesse della valutazione del generatore) e calcola,
per ogni livello predetto dal generatore (1..K-1), l'entropia della distribuzione dei codici (in nat, come la loss).
Un generatore che ha imparato solo le frequenze ha loss ≈ entropia; quello che ha imparato la struttura scende sotto.

Gira sulla GPU con al massimo il 20% della memoria e batch piccoli, così non toglie memoria a un training in corso
(sulla CPU era troppo lento: oltre 20 minuti). Output: evaluation/results/entropia_codici.json

Uso: .venv\Scripts\python.exe -u -X utf8 -I evaluation\entropia_codici.py
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import load_valid_rows  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

DEV = "cuda"
torch.cuda.set_per_process_memory_fraction(0.2)
BATCH = 32
TOKENIZER = {
    "ufficiale (12)": "artifacts/tinystories/config.yaml",
    "nostro 12": "runs/mytok-12std_614400_2400x256_s42/export/config.yaml",
    "nostro 12 dati": "runs/mytok-12std-dati_614400_2400x256_s42/export/config.yaml",
    "nostro 12 dati+rv": "runs/mytok-12std-dati-rv_614400_2400x256_s42/export/config.yaml",
    "nostro 26": "runs/mytok-26lin_614400_2400x256_s42/export/config.yaml",
    "nostro 40": "runs/mytok-40lin_614400_2400x256_s42/export/config.yaml",
}
GENERATORI = {   # tokenizer -> {etichetta: run}
    "ufficiale (12)": {"paper": "gen-tokuff_51200_400x128_lin75-25_lr3e-4_s42",
                       "migliore": "gen-tokuff_51200_400x128_c50_lr1e-3_flex_s42"},
    "nostro 12": {"paper": "gen-mytok12std-2400x256_51200_400x128_lin75-25_lr3e-4_s42",
                  "migliore": "gen-mytok12std-2400x256_51200_400x128_c50_lr1e-3_flex_s42"},
    "nostro 12 dati": {"paper": "gen-mytok12std-dati-2400x256_51200_400x128_lin75-25_lr3e-4_s42",
                       "migliore": "gen-mytok12std-dati-2400x256_51200_400x128_c50_lr1e-3_flex_s42"},
    "nostro 12 dati+rv": {"paper": "gen-mytok12std-dati-rv-2400x256_51200_400x128_lin75-25_lr3e-4_s42",
                          "migliore": "gen-mytok12std-dati-rv-2400x256_51200_400x128_c50_lr1e-3_flex_s42"},
    "nostro 26": {"paper": "gen-mytok26lin-2400x256_51200_400x128_lin75-25_lr3e-4_s42",
                  "migliore": "gen-mytok26lin-2400x256_51200_400x128_c50_lr1e-3_flex_s42"},
    "nostro 40": {"paper": "gen-mytok40lin-2400x256_51200_400x128_lin75-25_lr3e-4_s42",
                  "migliore": "gen-mytok40lin-2400x256_51200_400x128_c50_lr1e-3_flex_s42"},
}


def carica(base_config):
    bc = os.path.join(PROJECT_ROOT, base_config)
    cfg = OmegaConf.load(bc)
    d = os.path.dirname(bc)
    t = cfg.tokenizer
    t.vqvae.checkpoint_path = os.path.join(d, cfg.checkpoints.tokenizer)
    t.pre_tokenizer.vocab_file_path = os.path.join(d, cfg.checkpoints.bpe_vocab)
    t.pre_tokenizer.merges_file_path = os.path.join(d, cfg.checkpoints.bpe_merges)
    return VQVAETokenizer.from_config(t).vqvae.to(DEV).eval()


def entropia(counts):
    p = counts[counts > 0].double() / counts.sum()
    return float(-(p * p.log()).sum())


def loss_generatore(run):
    p = os.path.join(PROJECT_ROOT, "runs", run, "log.jsonl")
    if not os.path.exists(p):
        return None
    ev = [json.loads(l) for l in open(p, encoding="utf-8") if '"event": "eval"' in l]
    return ev[-1]["val_per_livello"] if ev else None


rows = load_valid_rows(os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), 128, 4096)
uff = json.load(open(os.path.join(PROJECT_ROOT, "evaluation", "results", "2026-11-16_riferimento_ufficiale.json"),
                     encoding="utf-8"))["generatore"]["ufficiale"]["per_livello"]
out = {"storie": len(rows), "data": time.strftime("%Y-%m-%d %H:%M:%S"), "tokenizer": {}}
for nome, bc in TOKENIZER.items():
    t0 = time.time()
    m = carica(bc)
    lv = list(m.vq.levels)
    with torch.no_grad():
        idx = torch.cat([m.encode(rows[i:i + BATCH].to(DEV)).view(-1, sum(lv)).cpu() for i in range(0, len(rows), BATCH)])
    H, usati, o = [], [], 0
    for k, l in enumerate(lv):
        c = torch.bincount(idx[:, o:o + l].reshape(-1), minlength=m.vq.vqvae_vocab_sizes[k])
        H.append(entropia(c))
        usati.append(int((c > 0).sum()))
        o += l
    pesi = lv[1:]
    Hm = sum(h * w for h, w in zip(H[1:], pesi)) / sum(pesi)
    rec = {"livelli": lv, "entropia_per_livello": [round(h, 3) for h in H], "codici_usati_per_livello": usati,
           "entropia_media_livelli_predetti": round(Hm, 3), "generatori": {}}
    gens = dict(GENERATORI[nome])
    for et, run in gens.items():
        pl = loss_generatore(run)
        if pl is None:
            continue
        Lm = sum(x * w for x, w in zip(pl, pesi)) / sum(pesi)
        rec["generatori"][et] = {"run": run, "loss_media": round(Lm, 3),
                                 "imparato_oltre_frequenze_nat": round(Hm - Lm, 3),
                                 "imparato_percento": round(100 * (Hm - Lm) / Hm, 1),
                                 "per_livello": [round(x, 3) for x in pl]}
    if nome.startswith("ufficiale"):
        Lm = sum(r["loss"] * w for r, w in zip(uff, pesi)) / sum(pesi)
        rec["generatori"]["ufficiale addestrato (50.000 passi × 2.048)"] = {
            "loss_media": round(Lm, 3), "imparato_oltre_frequenze_nat": round(Hm - Lm, 3),
            "imparato_percento": round(100 * (Hm - Lm) / Hm, 1), "per_livello": [r["loss"] for r in uff]}
    out["tokenizer"][nome] = rec
    print(f"{nome}: {len(lv)} livelli, entropia media {Hm:.3f} nat ({time.time() - t0:.0f}s)", flush=True)
    for et, g in rec["generatori"].items():
        print(f"    {et:45s} loss {g['loss_media']:.3f}  imparato oltre le frequenze {g['imparato_oltre_frequenze_nat']:+.3f} nat"
              f" ({g['imparato_percento']:+.1f}%)", flush=True)
    del m
    torch.cuda.empty_cache()

with open(os.path.join(PROJECT_ROOT, "evaluation", "results", "entropia_codici.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
