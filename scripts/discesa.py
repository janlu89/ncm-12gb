"""Ramo di discesa da una run WSD (2026-11-22).

Una run con `train.schedule=wsd train.decay_steps=0` ha il learning rate costante e si può allungare quanto si vuole
(`generator/train_generator.py --run <nome> --estendi <passi>`). Per sapere quanto vale il modello a un certo punto
si fa un ramo: si copia l'ultimo checkpoint in una run nuova `<nome>_disc<passo>-<D>`, si fanno D passi con il
learning rate che scende fino a min_lr, si esporta e si misura la Gen. PPL. La run principale non viene toccata e
può continuare.

Uso (dalla cartella del progetto, staccato):
    scripts/avvia_detached.sh 'python -u -I scripts/discesa.py --da <run> --passi 400' runs/_cicli/discesa.log discesa
Opzioni: --checkpoint <file> (default: l'ultimo), --solo-prepara (crea la run e non lancia nulla).
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402

from common import checkpoint as ck  # noqa: E402
from common.rundirs import run_dir as prepara_run_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--da", required=True, help="run di partenza (runs/<nome>), con train.schedule=wsd")
ap.add_argument("--passi", type=int, required=True, help="passi della discesa")
ap.add_argument("--checkpoint", default="", help="checkpoint di partenza (default: l'ultimo della run)")
ap.add_argument("--solo-prepara", action="store_true")
a = ap.parse_args()

src = os.path.join(ROOT, "runs", a.da)
cfg = OmegaConf.load(os.path.join(src, "config.yaml"))
assert str(cfg.train.get("schedule", "cosine")) == "wsd", "la run di partenza deve avere train.schedule=wsd"
assert int(cfg.train.get("decay_steps", 0) or 0) == 0, "la run di partenza è già in discesa (decay_steps > 0)"
if a.checkpoint:
    path = a.checkpoint
    passo = int(os.path.basename(path).split("_")[1].split(".")[0])
else:
    lista = ck.list_checkpoints(os.path.join(src, "checkpoints"))
    assert lista, f"nessun checkpoint in {src}/checkpoints"
    passo, path = lista[-1]
c = cfg.get("cicli")
if c and int(c.get("ripetizioni", 1)) > 1:
    tappe = list(c.get("crescita_ai_passi", None) or []) or ([int(c.get("crescita_al_passo", 0) or 0)] if c.get("crescita_al_passo") else [])
    if any(int(t) > passo for t in tappe):
        sys.exit(f"crescita non completa al passo {passo} (tappe {tappe}): la discesa accenderebbe ripetizioni a metà")

nome = f"{a.da}_disc{passo}-{a.passi}"
dst = prepara_run_dir(nome)
if os.path.exists(os.path.join(dst, "config.yaml")):
    print(f"[discesa] {nome} esiste già: riprendo", flush=True)
else:
    os.makedirs(os.path.join(dst, "checkpoints"), exist_ok=True)
    shutil.copy2(path, os.path.join(dst, "checkpoints", os.path.basename(path)))
    cfg.train.steps = int(passo) + int(a.passi)
    cfg.train.decay_steps = int(a.passi)
    OmegaConf.save(cfg, os.path.join(dst, "config.yaml"))
    for f in ("meta.json", "pip-freeze.txt"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    with open(os.path.join(dst, "DISCESA.txt"), "w", encoding="utf-8") as f:
        f.write(f"ramo di discesa da {a.da}, checkpoint {path} (passo {passo}), {a.passi} passi fino a min_lr\n")
    print(f"[discesa] creata {nome}: passi {passo} -> {cfg.train.steps}, discesa negli ultimi {a.passi}", flush=True)
if a.solo_prepara:
    sys.exit(0)

py = sys.executable
r = subprocess.run([py, "-u", "-I", os.path.join(ROOT, "generator", "train_generator.py"), "--run", nome], cwd=ROOT)
if r.returncode != 0:
    sys.exit(r.returncode)
r = subprocess.run([py, "-u", "-X", "utf8", "-I", os.path.join(ROOT, "evaluation", "gen_ppl_run.py"), "--run", nome], cwd=ROOT)
sys.exit(r.returncode)
