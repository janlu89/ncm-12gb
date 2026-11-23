"""Prove di correttezza del tokenizer nostro (docs/piani/tokenizer-operativo.md, T1-T8).

T1 caricamento  i pesi ufficiali si caricano nella nostra classe (strict)
T2 codifica     con i pesi ufficiali, stessi codici del codice upstream
T3 forward      con i pesi ufficiali, in valutazione: stessi logit e stessa loss VQ di upstream
T4 EMA          un passo di training aggiorna il codebook del livello 0 come il forward upstream
T5 accumulo     statistiche su 4 micro-batch = statistiche di un batch unico
T6 export       il file esportato si carica nel codice upstream e dà gli stessi codici
T7 ripresa      training interrotto e ripreso = training continuo
T8 generatore   il generatore si costruisce, fa un passo e genera su tokenizer con livelli diversi

Uso: .venv/Scripts/python.exe -u -X utf8 -I tests/test_tokenizer.py [--salta-ripresa]
"""
import argparse
import copy
import gc
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import EOS_ID, load_valid_rows  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from tokenizer import stage1  # noqa: E402
from tokenizer.quantizer import NOME_NOSTRO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--salta-ripresa", action="store_true")
a = ap.parse_args()
dev = "cuda"
# Stessa numerica su Windows e Linux: con il TF32 di cuDNN (default di PyTorch) T5 dipende dagli algoritmi
# che cuDNN sceglie su ogni sistema (2026-11-22: su Linux differenza 3,3e-01, senza TF32 6e-07).
torch.backends.cudnn.allow_tf32 = False
esiti = []


def libera():
    gc.collect()
    torch.cuda.empty_cache()


def esito(nome, ok, dettaglio):
    esiti.append((nome, ok))
    print(f"{'SUPERATO' if ok else 'FALLITO '}  {nome}: {dettaglio}", flush=True)


art = os.path.join(PROJECT_ROOT, "artifacts", "tinystories")
official = OmegaConf.load(os.path.join(art, "config.yaml"))
rows = load_valid_rows(os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), 128, 1024).to(dev)
sd_uff = torch.load(os.path.join(art, "tokenizer.pt"), map_location="cpu", weights_only=False)["model"]

# ---------- T1-T3: pesi ufficiali nella nostra classe ----------
cfg_up = copy.deepcopy(official.tokenizer.vqvae)
cfg_up.checkpoint_path = None
cfg_no = copy.deepcopy(cfg_up)
cfg_no.vector_quantizer_config.clss = NOME_NOSTRO
up = VQVAE(**cfg_up).to(dev).eval()
no = VQVAE(**cfg_no).to(dev).eval()
up.load_state_dict(sd_uff, strict=True)
try:
    no.load_state_dict(sd_uff, strict=True)
    esito("T1 caricamento", True, f"{len(sd_uff)} tensori, strict=True")
except Exception as e:  # noqa: BLE001
    esito("T1 caricamento", False, repr(e))
    sys.exit(1)

with torch.no_grad():
    cu = torch.cat([up.encode(rows[i:i + 256]) for i in range(0, 1024, 256)])
    cn = torch.cat([no.encode(rows[i:i + 256]) for i in range(0, 1024, 256)])
    uguali = float((cu == cn).float().mean())
    esito("T2 codifica", uguali == 1.0, f"codici uguali {uguali:.6f} su {cu.numel():,}")

    lu, vu, *_ = up(rows[:256])
    ln, vn, *_ = no(rows[:256])
    d_logit, d_vq = float((lu - ln).abs().max()), abs(float(vu) - float(vn))
    esito("T3 forward", d_logit < 1e-4 and d_vq < 1e-4 * max(1.0, abs(float(vu))),
          f"differenza massima dei logit {d_logit:.2e}; loss VQ upstream {float(vu):.6g}, nostra {float(vn):.6g}")
del up, no
libera()

# ---------- T4-T5: aggiornamento EMA ----------
cfg = OmegaConf.merge(OmegaConf.load(os.path.join(PROJECT_ROOT, "configs", "stage1_tokenizer.yaml")),
                      OmegaConf.from_dotlist(["eval.n_stories=512", "eval.batch=64"]))
_, vq_cfg, levels = stage1.config_vqvae(cfg)
cfg_no = copy.deepcopy(vq_cfg)
cfg_no.vector_quantizer_config.clss = NOME_NOSTRO
torch.manual_seed(0)
nostro = VQVAE(**cfg_no).to(dev).train()
upstream = VQVAE(**copy.deepcopy(vq_cfg)).to(dev).train()
upstream.load_state_dict(nostro.state_dict(), strict=True)
nostro.vq.fine_drop_prob = upstream.vq.fine_drop_prob = 0.0
with torch.no_grad():
    z = nostro.eval()._encode_pre_quant(rows[:256]).float()
    nostro.train()
    upstream.vq(z)
    nostro.vq(z)
    nostro.vq.ema_step()
    d = max(float((getattr(upstream.vq, n) - getattr(nostro.vq, n)).abs().max())
            for n in ("ema_cluster_size_0", "ema_w_0"))
    d = max(d, float((upstream.vq.embeddings[0].weight - nostro.vq.embeddings[0].weight).abs().max()))
    esito("T4 EMA", d < 1e-4, f"livello 0 dopo un passo: differenza massima da upstream {d:.2e}")

    torch.manual_seed(0)
    m1 = VQVAE(**cfg_no).to(dev).train()
    m2 = VQVAE(**cfg_no).to(dev).train()
    m2.load_state_dict(m1.state_dict(), strict=True)
    m1.vq.fine_drop_prob = m2.vq.fine_drop_prob = 0.0
    z = m1.eval()._encode_pre_quant(rows[:256]).float()
    m1.train()
    m1.vq(z)
    for i in range(0, 256, 64):
        m2.vq(z[i:i + 64])
    m1.vq.ema_step()
    m2.vq.ema_step()
    d = max(float((m1.vq.embeddings[i].weight - m2.vq.embeddings[i].weight).abs().max()) for i in range(len(levels)))
    scala = max(float(m1.vq.embeddings[i].weight.abs().max()) for i in range(len(levels)))
    esito("T5 accumulo", d < 1e-3 * scala,
          f"codebook dopo 1 batch da 256 contro 4 da 64: differenza massima {d:.2e} (valori fino a {scala:.1f})")
del nostro, upstream, m1, m2, z
libera()

# ---------- T6 e T8: export e generatore su tokenizer con livelli diversi ----------
tmp = os.path.join(PROJECT_ROOT, "runs", "_test_tokenizer")
shutil.rmtree(tmp, ignore_errors=True)
from generator import stage2  # noqa: E402

casi = [(12, "ufficiale"), (16, "lineare"), (32, "esponenziale"), (40, "lineare")]
ok6, ok8, det6, det8 = True, True, [], []
for k, schema in casi:
    c = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"tokenizer.livelli={k}", f"tokenizer.schema={schema}"]))
    s = stage1.build(c)
    d_run = os.path.join(tmp, f"tok_{k}_{schema}")
    info = s.export(s.net, d_run, passo=0)
    ok6 &= info["codici_uguali_a_upstream"] == 1.0
    det6.append(f"K={k} {schema}: {info['codici_uguali_a_upstream']:.6f}")
    del s
    libera()
    g = OmegaConf.merge(OmegaConf.load(os.path.join(PROJECT_ROOT, "configs", "stage2_generator.yaml")),
                        OmegaConf.from_dotlist([f"base_config={os.path.join(d_run, 'export', 'config.yaml')}",
                                                "eval.n_stories=64", "train.micro_batch=2"]))
    try:
        s2 = stage2.build(g)
        x = rows[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = s2.loss_fn(s2.net, x)
        loss.backward()
        s2.net.eval()
        with torch.no_grad():
            L0 = s2.levels[0]
            l0 = torch.zeros(2, L0, dtype=torch.long, device=dev)
            gen = s2.net.generate(l0, torch.full((2, 1), EOS_ID, device=dev), temperature=1.0, top_k=0, top_p=1.0)
            _, zl = s2.vq.idx_to_zhats(gen, return_zlevels=True)
            testo = s2.vqvae.decoder(zl[-1]).argmax(-1)
        buono = bool(torch.isfinite(loss)) and gen.shape == (2, sum(s2.levels)) and testo.shape == (2, 128)
        ok8 &= buono
        det8.append(f"K={k} {schema}: loss {float(loss):.3f}, {gen.shape[1]} codici generati")
        del s2, loss, gen, zl, testo
    except Exception as e:  # noqa: BLE001
        ok8 = False
        det8.append(f"K={k} {schema}: ERRORE {e!r}")
    libera()
esito("T6 export", ok6, "codici uguali a upstream dopo il caricamento — " + "; ".join(det6))
esito("T8 generatore", ok8, "; ".join(det8))
shutil.rmtree(tmp, ignore_errors=True)

# ---------- T7: ripresa ----------
if not a.salta_ripresa:
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    comuni = ["train.steps=12", "train.micro_batch=64", "train.accum=2", "train.warmup=4", "train.eval_every=0",
              "train.log_every=1", "train.ckpt_every_minutes=1e9", "train.deterministic=true", "eval.n_stories=256"]

    def lancia(nome, extra=()):
        p = subprocess.run([sys.executable, "-u", "-X", "utf8", "-I", os.path.join(ROOT, "tokenizer", "train_vqvae.py"),
                            "--run", nome] + comuni + list(extra), cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8")
        ultima = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        # train_vqvae.py esce con 1 quando va in pausa (serve alle catene): qui la pausa è voluta.
        if p.returncode != 0 and ultima != "paused":
            print(p.stdout[-2000:], p.stderr[-4000:])
            raise SystemExit(f"run {nome} fallito")
        return ultima

    def stato(nome):
        from glob import glob
        return torch.load(sorted(glob(os.path.join(ROOT, "runs", nome, "checkpoints", "step_*.pt")))[-1],
                          map_location="cpu", weights_only=False)

    def diff(x, y):
        if isinstance(x, torch.Tensor):
            if x.dtype.is_floating_point:
                return float((x.double() - y.double()).abs().max()) if x.numel() else 0.0
            return 0.0 if torch.equal(x, y) else float("inf")
        if isinstance(x, dict):
            return max([diff(x[k], y[k]) for k in x] or [0.0])
        if isinstance(x, (list, tuple)):
            return max([diff(p, q) for p, q in zip(x, y)] or [0.0])
        return 0.0 if x == y else float("inf")

    nomi = ["_test_tok_A1", "_test_tok_A2", "_test_tok_B"]
    for n in nomi:
        shutil.rmtree(os.path.join(ROOT, "runs", n), ignore_errors=True)
    lancia(nomi[0])
    lancia(nomi[1])
    r1 = lancia(nomi[2], ["train.debug_stop_at_step=6"])
    r2 = lancia(nomi[2])
    sA1, sA2, sB = stato(nomi[0]), stato(nomi[1]), stato(nomi[2])
    chiavi = ["model", "model_non_persistent", "optimizer", "data"]
    rumore = {k: diff(sA1[k], sA2[k]) for k in chiavi}
    ripresa = {k: diff(sA1[k], sB[k]) for k in chiavi}
    ok = r1 == "paused" and r2 == "done" and all(ripresa[k] <= max(10 * rumore[k], 1e-6) for k in chiavi)
    esito("T7 ripresa", ok, f"pausa al passo 6 poi ripresa: differenze A1-B {ripresa}; rumore A1-A2 {rumore}")
    for n in nomi:
        shutil.rmtree(os.path.join(ROOT, "runs", n), ignore_errors=True)

print()
print("ESITO: SUPERATO" if all(ok for _, ok in esiti) else "ESITO: FALLITO — " + ", ".join(n for n, ok in esiti if not ok))
sys.exit(0 if all(ok for _, ok in esiti) else 1)
