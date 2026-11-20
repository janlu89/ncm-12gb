r"""Quanto il generatore tiene conto del codice del livello 0. Nessun training.

In generazione il livello 0 non viene predetto: si estrae a caso dalle frequenze (unigramma), poi il generatore
produce i livelli 1..K-1. Qui, sulle 4.096 storie di validazione: loss del generatore per livello con il livello 0
vero, e con il livello 0 preso da un'altra storia a caso (gli altri livelli restano quelli veri).
Se la loss non cambia, il generatore ignora il livello 0. Scheda: docs/esperimenti/2026-11-20_fragilita-codici.md

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-20\uso_livello0.py
"""
import json
import os
import sys
import time

QUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(QUI))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from generator.stage2 import build  # noqa: E402

RUN = {
    "ufficiale, 51.200 storie": "gen-tokuff_51200_400x128_c50_lr1e-3_flex_s42",
    "nostro 4.800 passi, 51.200 storie": "gen-mytok12std-dati-rv-4800x256_51200_400x128_c50_lr1e-3_flex_s42",
    "nostro 38.400 passi, 51.200 storie": "gen-mytok12std-dati-rv-38400x256_51200_400x128_c50_lr1e-3_flex_s42",
    "ufficiale, 409.600 storie": "gen-tokuff_409600_3200x128_c50_lr1e-3_flex_s42",
    "nostro 2.400 passi, 409.600 storie": "gen-mytok12std-dati-rv-2400x256_409600_3200x128_c50_lr1e-3_flex_s42",
}


@torch.no_grad()
def loss_per_livello(s, model, idx):
    L0 = s.levels[0]
    tot = torch.zeros(len(s.seg), dtype=torch.float64)
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        with torch.autocast("cuda", enabled=False):
            nll, _ = s.per_code_nll(model, s.inputs_from_idx(b), b[:, L0:])
        for j, (a, c) in enumerate(s.seg):
            tot[j] += nll[:, a:c].sum().item()
    n = torch.tensor([len(idx) * (c - a) for a, c in s.seg], dtype=torch.float64)
    per = tot / n
    return float(tot.sum() / n.sum()), [round(float(x), 4) for x in per]


out = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "run": {}}
for nome, run in RUN.items():
    d = os.path.join(ROOT, "runs", run)
    cfg = OmegaConf.load(os.path.join(d, "config.yaml"))
    s = build(cfg)
    sd = torch.load(os.path.join(d, "export", "generator.pt"), map_location="cuda", weights_only=False)["model"]
    s.net.load_state_dict(sd)
    model = s.net.eval()
    idx = s.vq and None
    # stesse storie e stessi codici della valutazione della run
    from common.data import load_valid_rows  # noqa: E402
    rows = load_valid_rows(os.path.join(ROOT, str(cfg.data.dir)), int(cfg.data.length), int(cfg.eval.n_stories)).cuda()
    idx = torch.cat([s.encode(rows[i:i + 256]) for i in range(0, len(rows), 256)])
    L0 = s.levels[0]
    g = torch.Generator(device="cuda").manual_seed(1234)
    perm = torch.randperm(len(idx), device="cuda", generator=g)
    scambiato = idx.clone()
    scambiato[:, :L0] = idx[perm, :L0]
    diversi = float((scambiato[:, :L0] != idx[:, :L0]).float().mean())
    v, pv = loss_per_livello(s, model, idx)
    w, pw = loss_per_livello(s, model, scambiato)
    out["run"][nome] = {"run": run, "livello0_diverso_dopo_scambio": round(diversi, 4),
                        "loss_vero": round(v, 4), "loss_scambiato": round(w, 4), "aumento": round(w - v, 4),
                        "per_livello_vero": pv, "per_livello_scambiato": pw}
    print(f"{nome:38s} livello 0 diverso {diversi:.3f} | loss vero {v:.4f} scambiato {w:.4f} aumento {w - v:+.4f}", flush=True)
    print(f"    per livello, aumento: {[round(b - a, 3) for a, b in zip(pv, pw)]}", flush=True)
    del s, model, sd
    torch.cuda.empty_cache()

with open(os.path.join(QUI, "uso_livello0.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("scritto", os.path.join(QUI, "uso_livello0.json"))
