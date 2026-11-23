"""Pausa -> valutazione (2026-11-22): Gen. PPL di un checkpoint senza chiudere la run.

Crea `runs/<nome>_val<passo>/` con la config della run e l'export del generatore preso dal checkpoint, poi misura la
Gen. PPL (evaluation/gen_ppl_run.py). La run di partenza non viene toccata: si riprende rilanciando il suo comando.
Va usato con la run in pausa (la GPU serve alla valutazione): se il training è vivo lo script si ferma.
Con una run WSD a lr costante il numero è "senza discesa"; per il valore con la discesa usare scripts/discesa.py.

Uso:  scripts/avvia_detached.sh 'python -u -I scripts/valuta_checkpoint.py --run <nome>' runs/_cicli/valuta.log valuta
Opzioni: --checkpoint <file> (default: l'ultimo), --solo-export, --device cpu (solo per prove).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import checkpoint as ck  # noqa: E402
from common.rundirs import run_dir as prepara_run_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--checkpoint", default="")
ap.add_argument("--solo-export", action="store_true")
ap.add_argument("--device", default="cuda")
ap.add_argument("--prova-veloce", action="store_true", help="solo per prove: unigramma su 512 storie, 2 storie di prova")
a = ap.parse_args()

src = os.path.join(ROOT, "runs", a.run)
pid_f = os.path.join(src, "pid.txt")
if os.path.exists(pid_f):
    try:
        pid = int(open(pid_f).read().strip())
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore") if os.path.exists(f"/proc/{pid}") else ""
        if a.run in cmd:
            sys.exit(f"la run {a.run} è in corso (pid {pid}): mettila in pausa prima (file runs/{a.run}/STOP)")
    except ValueError:
        pass

if a.checkpoint:
    path = a.checkpoint
else:
    lista = ck.list_checkpoints(os.path.join(src, "checkpoints"))
    assert lista, f"nessun checkpoint in {src}/checkpoints"
    path = lista[-1][1]
state = torch.load(path, map_location="cpu", weights_only=False)
passo = int(state["step"])
cfg = OmegaConf.load(os.path.join(src, "config.yaml"))
c = cfg.get("cicli")
if c and int(c.get("ripetizioni", 1)) > 1:
    tappe = list(c.get("crescita_ai_passi", None) or []) or ([int(c.get("crescita_al_passo", 0) or 0)] if c.get("crescita_al_passo") else [])
    if any(int(t) > passo for t in tappe):
        sys.exit(f"crescita non completa al passo {passo} (tappe {tappe}): il generatore esportato non sarebbe quello allenato")
if a.prova_veloce:
    cfg.export.level0_sequences, cfg.export.samples = 512, 2
if a.device == "cpu" and cfg.get("model_overrides"):
    cfg.model_overrides.use_flex_attention = False

from generator.stage2 import build  # noqa: E402

nome = f"{a.run}_val{passo}"
dst = prepara_run_dir(nome)
OmegaConf.save(cfg, os.path.join(dst, "config.yaml"))
with open(os.path.join(dst, "VALUTAZIONE.txt"), "w", encoding="utf-8") as f:
    f.write(f"valutazione di {a.run}, checkpoint {path} (passo {passo}), senza altro training\n")
s = build(cfg, dev=a.device)
s.net.load_state_dict(state["model"], strict=True)
s.crescita(passo)
s.export(s.net, dst)
print(f"[valuta] export di {a.run} al passo {passo} in {dst}/export", flush=True)
if a.solo_export:
    sys.exit(0)
del s, state
torch.cuda.empty_cache()
r = subprocess.run([sys.executable, "-u", "-X", "utf8", "-I", os.path.join(ROOT, "evaluation", "gen_ppl_run.py"), "--run", nome], cwd=ROOT)
sys.exit(r.returncode)
