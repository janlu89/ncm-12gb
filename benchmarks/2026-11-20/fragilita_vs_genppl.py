r"""La fragilità dei codici segue la qualità del testo? Nessun training.

Per ogni tokenizer a 12 livelli (schema standard) di cui esiste un generatore standard (51.200 storie, ricetta
migliore, seed 42): fragilità misurata come in fragilita_codici.py (scambio con il codice di un'altra storia, stesso
livello e posizione) su tutti i livelli insieme al 5% e al 10%, e sul solo livello 0; poi correlazione di rango
(Spearman) con la Gen. PPL del generatore. Regola di lettura in docs/esperimenti/2026-11-20_fragilita-codici.md.

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-20\fragilita_vs_genppl.py
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

from common.data import PAD_ID, load_valid_rows  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

DEV = "cuda"
SUF = "_51200_400x128_c50_lr1e-3_flex_s42"
T = "runs/mytok-12std{}/export{}/config.yaml"
CASI = [  # etichetta, base_config del tokenizer, run del generatore
    ("ufficiale", "artifacts/tinystories/config.yaml", "gen-tokuff" + SUF),
    ("codici casuali, 2.400 passi", T.format("_614400_2400x256_s42", ""), "gen-mytok12std-2400x256" + SUF),
    ("dati, 2.400 passi", T.format("-dati_614400_2400x256_s42", ""), "gen-mytok12std-dati-2400x256" + SUF),
    ("R12, 2.400 passi", T.format("-dati-rv_614400_2400x256_s42", ""), "gen-mytok12std-dati-rv-2400x256" + SUF),
    ("R12, 4.800 passi", T.format("-dati-rv_1228800_4800x256_s42", ""), "gen-mytok12std-dati-rv-4800x256" + SUF),
    ("R12, 4.800 passi, senza norma L0", T.format("-dati-rv-nonorm_1228800_4800x256_s42", ""),
     "gen-mytok12std-dati-rv-nonorm-4800x256" + SUF),
    ("R12, batch 1.024", T.format("-dati-rv_1228800_1200x1024_s42", ""), "gen-mytok12std-dati-rv-1200x1024" + SUF),
    ("R12, batch 4.096", T.format("-dati-rv_1228800_300x4096_s42", ""), "gen-mytok12std-dati-rv-300x4096" + SUF),
    ("R12, lungo: copia passo 9.600", T.format("-dati-rv_9830400_38400x256_s42", "/passo_00009600"),
     "gen-mytok12std-dati-rv-38400x256-p9600" + SUF),
    ("R12, lungo: copia passo 19.200", T.format("-dati-rv_9830400_38400x256_s42", "/passo_00019200"),
     "gen-mytok12std-dati-rv-38400x256-p19200" + SUF),
    ("R12, 38.400 passi", T.format("-dati-rv_9830400_38400x256_s42", ""), "gen-mytok12std-dati-rv-38400x256" + SUF),
]
QUOTE = [0.05, 0.10]
SEED = 1234
BS = 128


def carica(base_config):
    bc = os.path.join(ROOT, base_config)
    cfg = OmegaConf.load(bc)
    d = os.path.dirname(bc)
    t = cfg.tokenizer
    t.vqvae.checkpoint_path = os.path.join(d, cfg.checkpoints.tokenizer)
    t.pre_tokenizer.vocab_file_path = os.path.join(d, cfg.checkpoints.bpe_vocab)
    t.pre_tokenizer.merges_file_path = os.path.join(d, cfg.checkpoints.bpe_merges)
    return VQVAETokenizer.from_config(t).vqvae.to(DEV).eval()


@torch.no_grad()
def decodifica(m, idx):
    v, L, K = m.vq, m.vq.levels[-1], len(m.vq.levels)
    z, o = 0, 0
    for i, l in enumerate(v.levels):
        e = v.embeddings[i].weight[idx[:, o:o + l]]
        z = z + v._apply_quant_resi(v._up_to_full(e, i, L) if i != K - 1 else e, i)
        o += l
    return m.decoder(z).argmax(-1)


@torch.no_grad()
def quota_errata(m, idx, rows):
    pred = torch.cat([decodifica(m, idx[i:i + BS]) for i in range(0, len(idx), BS)])
    nopad = rows != PAD_ID
    return float(((pred != rows) & nopad).sum() / nopad.sum())


def ranghi(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for pos, i in enumerate(o):
        r[i] = float(pos)
    return r


def spearman(a, b):
    ra, rb = ranghi(a), ranghi(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


rows = load_valid_rows(os.path.join(ROOT, "data", "raw", "tinystories"), 128, 4096).to(DEV)
out = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "storie": len(rows), "seed": SEED, "casi": []}
for et, bc, run in CASI:
    gp = json.load(open(os.path.join(ROOT, "runs", run, "eval", "genppl", "genppl.json"), encoding="utf-8"))
    m = carica(bc)
    lv = list(m.vq.levels)
    with torch.no_grad():
        idx = torch.cat([m.encode(rows[i:i + 256]).view(-1, sum(lv)) for i in range(0, len(rows), 256)])
    g = torch.Generator(device=DEV).manual_seed(SEED)
    rec = {"tokenizer": et, "base_config": bc, "generatore": run, "genppl": gp["genppl_mean"],
           "entropia": gp["entropy_mean"], "pulito": round(quota_errata(m, idx, rows), 5),
           "codici_usati_livello0": int(idx[:, 0].unique().numel())}
    for q in QUOTE:
        sp = idx.clone()
        sel = torch.rand(idx.shape, device=DEV, generator=g) < q
        altra = torch.randint(0, len(idx), (len(idx),), device=DEV, generator=g)
        sp[sel] = idx[altra][sel]
        rec[f"tutti_{int(q * 100)}%"] = round(quota_errata(m, sp, rows) - rec["pulito"], 5)
        sp2 = sp.clone()
        sp2[:, 0] = idx[:, 0]   # stesso disturbo, ma livello 0 lasciato vero
        rec[f"tutti_{int(q * 100)}%_senza_livello0"] = round(quota_errata(m, sp2, rows) - rec["pulito"], 5)
    sp = idx.clone()
    perm = torch.randperm(len(idx), device=DEV, generator=g)
    sp[:, 0] = idx[perm, 0]
    cambiati = float((sp[:, 0] != idx[:, 0]).float().mean())
    rec["livello0_scambiato"] = round(quota_errata(m, sp, rows) - rec["pulito"], 5)
    rec["livello0_quota_cambiata"] = round(cambiati, 4)
    out["casi"].append(rec)
    print(f"{et:34s} Gen.PPL {rec['genppl']:7.2f} | pulito {rec['pulito']:.4f} | tutti 5% {rec['tutti_5%']:.4f} "
          f"(senza L0 {rec['tutti_5%_senza_livello0']:.4f}) | tutti 10% {rec['tutti_10%']:.4f} | "
          f"L0 scambiato {rec['livello0_scambiato']:.4f} (cambiati {cambiati:.2f}, usati {rec['codici_usati_livello0']})",
          flush=True)
    del m
    torch.cuda.empty_cache()

ppl = [c["genppl"] for c in out["casi"]]
nostri = [i for i, c in enumerate(out["casi"]) if c["tokenizer"] != "ufficiale"]
out["spearman"] = {}
for k in ("pulito", "tutti_5%", "tutti_5%_senza_livello0", "tutti_10%", "tutti_10%_senza_livello0", "livello0_scambiato"):
    v = [c[k] for c in out["casi"]]
    out["spearman"][k] = {"tutti": round(spearman(v, ppl), 3),
                          "solo_nostri": round(spearman([v[i] for i in nostri], [ppl[i] for i in nostri]), 3)}
    print(f"Spearman con la Gen. PPL, {k:26s}: tutti {out['spearman'][k]['tutti']:+.3f}, "
          f"solo nostri {out['spearman'][k]['solo_nostri']:+.3f}", flush=True)
with open(os.path.join(QUI, "fragilita_vs_genppl.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("scritto", os.path.join(QUI, "fragilita_vs_genppl.json"))
