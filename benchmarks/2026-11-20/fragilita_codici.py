r"""Fragilità dei codici: quanto si rovina il testo decodificato quando una parte dei codici è sbagliata.
Nessun training. Confronta il tokenizer ufficiale con i nostri. Scheda: docs/esperimenti/2026-11-20_fragilita-codici.md

Per ogni tokenizer: codifica le 4.096 storie di validazione, sostituisce una quota dei codici (un livello alla volta,
oppure tutti insieme), decodifica e conta i token sbagliati rispetto al testo originale.
  scambio = codice di un'altra storia a caso, stesso livello e stessa posizione (errore plausibile)
  vicino  = codice più vicino nel codebook tra quelli usati a quel livello (errore minimo)

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-20\fragilita_codici.py
"""
import json
import os
import sys
import time

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(QUI)))
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import PAD_ID, load_valid_rows  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

DEV = "cuda"
TOKENIZER = {
    "ufficiale": "artifacts/tinystories/config.yaml",
    "nostro 4.800 passi": "runs/mytok-12std-dati-rv_1228800_4800x256_s42/export/config.yaml",
    "nostro 38.400 passi": "runs/mytok-12std-dati-rv_9830400_38400x256_s42/export/config.yaml",
}
QUOTA_LIVELLO = 0.25
QUOTE_TUTTI = [0.05, 0.10, 0.25, 0.50]
SEED = 1234
BS = 128


def carica(base_config):
    bc = os.path.join(PROJECT_ROOT, base_config)
    cfg = OmegaConf.load(bc)
    d = os.path.dirname(bc)
    t = cfg.tokenizer
    t.vqvae.checkpoint_path = os.path.join(d, cfg.checkpoints.tokenizer)
    t.pre_tokenizer.vocab_file_path = os.path.join(d, cfg.checkpoints.bpe_vocab)
    t.pre_tokenizer.merges_file_path = os.path.join(d, cfg.checkpoints.bpe_merges)
    return VQVAETokenizer.from_config(t).vqvae.to(DEV).eval()


@torch.no_grad()
def decodifica(m, idx):
    """Come benchmarks/2026-11-19/ricostruzione_fp64.py: somma dei contributi dei livelli, poi decoder."""
    v, L, K = m.vq, m.vq.levels[-1], len(m.vq.levels)
    z, o = 0, 0
    for i, l in enumerate(v.levels):
        e = v.embeddings[i].weight[idx[:, o:o + l]]
        z = z + v._apply_quant_resi(v._up_to_full(e, i, L) if i != K - 1 else e, i)
        o += l
    return m.decoder(z).argmax(-1)


@torch.no_grad()
def errori(m, idx, rows):
    pred = torch.cat([decodifica(m, idx[i:i + BS]) for i in range(0, len(idx), BS)])
    nopad = rows != PAD_ID
    sb = (pred != rows) & nopad
    return int(sb.sum()), int(nopad.sum()), float((~sb.any(1)).float().mean())


def vicini(m, idx, o, l, k):
    """Per ogni codice usato al livello k: il codice usato più vicino nel codebook (distanza a 64 bit)."""
    usati = idx[:, o:o + l].unique()
    e = m.vq.embeddings[k].weight[usati].double()
    mappa = torch.arange(m.vq.embeddings[k].weight.shape[0], device=idx.device)
    if len(usati) < 2:
        return mappa, len(usati)
    d = torch.cdist(e, e)
    d.fill_diagonal_(float("inf"))
    mappa[usati] = usati[d.argmin(1)]
    return mappa, len(usati)


rows = load_valid_rows(os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), 128, 4096).to(DEV)
out = {"storie": len(rows), "quota_per_livello": QUOTA_LIVELLO, "seed": SEED,
       "data": time.strftime("%Y-%m-%d %H:%M:%S"), "tokenizer": {}}
for nome, bc in TOKENIZER.items():
    t0 = time.time()
    m = carica(bc)
    lv = list(m.vq.levels)
    with torch.no_grad():
        idx = torch.cat([m.encode(rows[i:i + 256]).view(-1, sum(lv)) for i in range(0, len(rows), 256)])
    e0, tot, perf0 = errori(m, idx, rows)
    rec = {"livelli": lv, "pulito": {"token_sbagliati": round(e0 / tot, 5), "storie_perfette": round(perf0, 4)},
           "per_livello": [], "tutti_i_livelli_scambio": []}
    g = torch.Generator(device=DEV).manual_seed(SEED)
    o = 0
    for k, l in enumerate(lv):
        riga = {"livello": k, "codici_per_storia": l}
        mappa, n_usati = vicini(m, idx, o, l, k)
        riga["codici_usati"] = n_usati
        for tipo in ("scambio", "vicino"):
            sporco = idx.clone()
            sel = torch.rand(len(idx), l, device=DEV, generator=g) < QUOTA_LIVELLO
            blocco = sporco[:, o:o + l]
            if tipo == "scambio":
                altra = torch.randint(0, len(idx), (len(idx),), device=DEV, generator=g)
                nuovo = idx[altra, o:o + l]
            else:
                nuovo = mappa[blocco]
            blocco[sel] = nuovo[sel]
            cambiati = int((sporco[:, o:o + l] != idx[:, o:o + l]).sum())
            e1, _, perf = errori(m, sporco, rows)
            riga[tipo] = {"codici_cambiati": cambiati,
                          "quota_cambiata": round(cambiati / (len(idx) * l), 4),
                          "token_sbagliati": round(e1 / tot, 5),
                          "storie_perfette": round(perf, 4),
                          "danno_per_codice": round((e1 - e0) / cambiati, 4) if cambiati else None}
        rec["per_livello"].append(riga)
        print(f"{nome:20s} livello {k:2d} ({l:3d} codici, {n_usati:4d} usati): "
              f"scambio cambiati {riga['scambio']['quota_cambiata']:.3f} danno/codice {riga['scambio']['danno_per_codice']} | "
              f"vicino danno/codice {riga['vicino']['danno_per_codice']}", flush=True)
        o += l
    for q in QUOTE_TUTTI:
        sporco = idx.clone()
        sel = torch.rand(idx.shape, device=DEV, generator=g) < q
        altra = torch.randint(0, len(idx), (len(idx),), device=DEV, generator=g)
        sporco[sel] = idx[altra][sel]
        cambiati = int((sporco != idx).sum())
        e1, _, perf = errori(m, sporco, rows)
        rec["tutti_i_livelli_scambio"].append({"quota": q, "quota_cambiata": round(cambiati / idx.numel(), 4),
                                               "token_sbagliati": round(e1 / tot, 5), "storie_perfette": round(perf, 4),
                                               "danno_per_codice": round((e1 - e0) / cambiati, 4) if cambiati else None})
        print(f"{nome:20s} tutti i livelli {q:.0%}: token sbagliati {e1 / tot:.4f}, storie perfette {perf:.4f}", flush=True)
    rec["secondi"] = round(time.time() - t0, 1)
    out["tokenizer"][nome] = rec
    del m
    torch.cuda.empty_cache()

with open(os.path.join(QUI, "fragilita_codici.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("scritto", os.path.join(QUI, "fragilita_codici.json"))
