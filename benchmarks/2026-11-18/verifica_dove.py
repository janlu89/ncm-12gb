r"""Dove si perde la precisione nel tokenizer ufficiale: errore fp32 (contro fp64) lungo il percorso del livello 0.
Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-18\verifica_dove.py
"""
import json, os, sys
QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)
import torch
from verifica_fix import DEV, RAW, carica
from common.data import load_valid_rows

rows = load_valid_rows(RAW, 128, 256).to(DEV)
out = {}
for nome, bc in (("ufficiale", "artifacts/tinystories/config.yaml"),
                 ("nostro 12", "runs/mytok-12std_614400_2400x256_s42/export/config.yaml")):
    m = carica(bc); v = m.vq; L = v.levels[-1]
    def passi(m, x):
        v = m.vq
        z = m._encode_pre_quant(x)
        x0 = v._down_for_quant(z, 0, v.levels[0])
        e = v.embeddings[0].weight
        f = x0.reshape(-1, z.shape[-1])
        idx = ((f * f).sum(1, keepdim=True) + (e * e).sum(1) - 2 * f @ e.t()).argmin(1)
        h = e[idx].view(x.shape[0], 1, -1).transpose(1, 2)
        norme = []
        for strato in v.learned_up_l1:
            h = strato(h); norme.append(float(h.norm(dim=1).mean()))
        q = v._apply_quant_resi(h.transpose(1, 2).contiguous(), 0)
        return z, x0, idx, q, z - q, norme
    with torch.no_grad():
        z32, x32, i32, q32, r32, n32 = passi(m, rows)
        m = m.double()
        z64, x64, i64, q64, r64, n64 = passi(m, rows)
        # stessi codici del livello 0 per isolare l'errore della salita
        stessi = (i32 == i64)
        err_q = (q32.double() - q64).norm(dim=-1)[stessi].mean()
        r = {"norma_z": float(z64.norm(dim=-1).mean()),
             "errore_relativo_input_l0": float(((x32.double() - x64).norm(dim=-1) / x64.norm(dim=-1)).mean()),
             "codici_l0_diversi": float((~stessi).float().mean()),
             "norme_lungo_la_salita": ["%.3g" % x for x in n64],
             "norma_uscita_livello0": float(q64.norm(dim=-1).mean()),
             "errore_assoluto_uscita_livello0": float(err_q),
             "norma_residuo_dopo_l0": float(r64.norm(dim=-1).mean()),
             "errore_assoluto_residuo": float((r32.double() - r64).norm(dim=-1)[stessi].mean()),
             "norma_tipica_codici_ultimi_livelli": float(v.embeddings[-2].weight.double().norm(dim=-1).median())}
    out[nome] = r
    print(nome, json.dumps(r), flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open(os.path.join(QUI, "verifica_dove.json"), "w", encoding="utf-8"), indent=2)
