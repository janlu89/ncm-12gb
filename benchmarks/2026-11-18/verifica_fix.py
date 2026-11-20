r"""Verifica rapida della "correzione" del tokenizer (Fase 1a). Due modalità, entrambe sulla GPU, pochi minuti.

  precisione   Nessun training. Sul tokenizer ufficiale e sul nostro: quanti codici cambiano tra batch 256 e batch 16
               in fp32 e in fp64. Se in fp64 l'instabilità sparisce, la causa è la precisione numerica.
  training V   Training breve (default 600 passi × 256) di una variante, con misure ogni 100 passi:
               V0 = come l'originale (niente normalizzazione del livello 0, statistiche EMA iniziali 0)
               V1 = solo normalizzazione        V2 = solo EMA iniziale 1        V3 = entrambe (la nostra correzione)
               Non crea cartelle di run: scrive solo benchmarks/2026-11-18/verifica_fix_<V>.json

Uso: .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-18\verifica_fix.py precisione
     .venv\Scripts\python.exe -u -X utf8 -I benchmarks\2026-11-18\verifica_fix.py training V0 [passi]
"""
import json
import os
import sys
import time

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(QUI)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import TrainStories, load_valid_rows  # noqa: E402
from common.trainer import lr_at, make_optimizer  # noqa: E402

DEV = "cuda"
RAW = os.path.join(PROJECT_ROOT, "data", "raw", "tinystories")


@torch.no_grad()
def codifica(m, x):
    """Stesse operazioni di vq.z_to_idx, ma nel tipo numerico del modello (fp32 o fp64)."""
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


def a_blocchi(m, rows, bs):
    return torch.cat([codifica(m, rows[i:i + bs]) for i in range(0, len(rows), bs)])


def per_livello(a, b, levels):
    o, out = 0, []
    for lv in levels:
        out.append(round(float((a[:, o:o + lv] != b[:, o:o + lv]).float().mean()), 5))
        o += lv
    return out


def carica(base_config):
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    bc = os.path.join(PROJECT_ROOT, base_config)
    cfg = OmegaConf.load(bc)
    d = os.path.dirname(bc)
    t = cfg.tokenizer
    t.vqvae.checkpoint_path = os.path.join(d, cfg.checkpoints.tokenizer)
    t.pre_tokenizer.vocab_file_path = os.path.join(d, cfg.checkpoints.bpe_vocab)
    t.pre_tokenizer.merges_file_path = os.path.join(d, cfg.checkpoints.bpe_merges)
    return VQVAETokenizer.from_config(t).vqvae.to(DEV).eval()


def precisione():
    rows = load_valid_rows(RAW, 128, 1024).to(DEV)
    res = {}
    for nome, bc in (("ufficiale", "artifacts/tinystories/config.yaml"),
                     ("nostro 12 (V3, 2.400 passi)", "runs/mytok-12std_614400_2400x256_s42/export/config.yaml")):
        m = carica(bc)
        lv = list(m.vq.levels)
        up = torch.cat([m.encode(rows[i:i + 256]).view(-1, sum(lv)) for i in range(0, 1024, 256)])
        a32, b32, c32 = a_blocchi(m, rows, 256), a_blocchi(m, rows, 16), a_blocchi(m, rows, 256)
        m = m.double()
        a64, b64 = a_blocchi(m, rows, 256), a_blocchi(m, rows, 16)
        r = {"nostra_codifica_uguale_a_upstream_fp32": float((up == a32).float().mean()),
             "fp32_stesso_batch_due_volte": float((a32 != c32).float().mean()),
             "fp32_batch256_vs_16": float((a32 != b32).float().mean()),
             "fp64_batch256_vs_16": float((a64 != b64).float().mean()),
             "fp32_vs_fp64_batch256": float((a32 != a64).float().mean()),
             "fp32_vs_fp64_batch16": float((b32 != b64).float().mean()),
             "per_livello_fp32_256_vs_16": per_livello(a32, b32, lv),
             "per_livello_fp64_256_vs_16": per_livello(a64, b64, lv),
             "per_livello_fp32_vs_fp64": per_livello(a32, a64, lv)}
        res[nome] = r
        print(nome, json.dumps(r, indent=1), flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(QUI, "verifica_fix_precisione.json"), "w", encoding="utf-8"), indent=2)


def training(variante, passi):
    from tokenizer import stage1
    # V0-V3: le due modifiche (1a). R1/R2/R12: la nostra correzione (V3) + rimedi contro il collasso (1c):
    # R1 codebook dai dati, R2 riavvio dei codici quasi inutilizzati nella prima metà dei passi, R12 entrambi.
    norma = variante in ("V1", "V3", "R1", "R2", "R12")
    ema1 = variante in ("V2", "V3", "R1", "R2", "R12")
    init_dati = variante in ("R1", "R12")
    riavvio = passi // 2 if variante in ("R2", "R12") else 0
    cfg = OmegaConf.merge(OmegaConf.load(os.path.join(PROJECT_ROOT, "configs", "stage1_tokenizer.yaml")),
                          OmegaConf.from_dotlist([f"tokenizer.norma_livello0={str(norma).lower()}", f"train.steps={passi}",
                                                  "train.micro_batch=256", "train.accum=1", f"train.warmup={max(1, passi // 20)}",
                                                  "eval.n_stories=1024", "eval.batch=256",
                                                  f"tokenizer.rimedi.riavvio_fino_al_passo={riavvio}"]))
    s = stage1.build(cfg)
    m, v = s.net, s.net.vq
    if init_dati:
        s.init_dai_dati()
    if not ema1:  # come upstream: statistiche a zero
        for i in range(len(v.levels)):
            getattr(v, f"ema_cluster_size_{i}").zero_()
    opt = make_optimizer(m, cfg.train)
    data = TrainStories(RAW, 128, int(cfg.data.seed))
    rows = s.valid_rows.to(DEV)
    storia = []

    @torch.no_grad()
    def misura(step):
        m.eval()
        e = s.eval_fn(m)
        z = m._encode_pre_quant(rows[:512]).float()
        x0 = v._down_for_quant(z, 0, v.levels[0])
        n0 = v.embeddings[0].weight.float().norm(dim=-1)
        idx = torch.cat([m.encode(rows[i:i + 256]).view(-1, sum(v.levels)) for i in range(0, 1024, 256)])
        usati0 = idx[:, 0].unique()
        r = {"passo": step, "acc_token": round(e["acc_token"], 4), "sequenze_perfette": round(e["sequenze_perfette"], 4),
             "norma_input_l0": float(x0.norm(dim=-1).mean()), "norma_z": float(z.norm(dim=-1).mean()),
             "norma_codici_l0_usati": float(n0[usati0].median()), "norma_codici_l0_max": float(n0.max()),
             "codici_l0_distinti_su_1024_storie": int(usati0.numel()),
             "codici_l1_distinti": int(idx[:, 1:9].unique().numel()),
             "instabili_256_vs_16": round(s.stabilita(m, 512), 5),
             "norma_codebook_max_tutti": round(e["norma_codebook_max"], 2)}
        storia.append(r)
        print(variante, json.dumps(r), flush=True)
        m.train()

    m.train()
    misura(0)
    t0 = time.time()
    for step in range(passi):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train)
        x = data.next_batch(256).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = s.loss_fn(m, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.train.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        v.ema_step()
        if (step + 1) % 100 == 0:
            if not torch.isfinite(loss):
                print(variante, "loss non finita al passo", step + 1, flush=True)
                break
            misura(step + 1)
    json.dump({"variante": variante, "norma_livello0": norma, "ema_iniziale_1": ema1, "init_dati": init_dati,
               "riavvio_fino_al_passo": riavvio, "passi": passi,
               "secondi": round(time.time() - t0, 1), "misure": storia},
              open(os.path.join(QUI, f"verifica_fix_{variante}.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    if sys.argv[1] == "precisione":
        precisione()
    else:
        training(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 600)
