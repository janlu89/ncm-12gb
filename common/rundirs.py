r"""Cartelle delle run. Le run nuove possono stare su un altro disco: la cartella vera viene creata lì e
collegata in runs/<nome> con una giunzione NTFS (non serve l'amministratore). Per script e dashboard
runs/<nome> è una cartella come le altre.

Il disco si sceglie con il file configs/locale.yaml (chiave `disco_run`, es. D:\ncm-runs).
Senza quel file le run nascono in runs/ come sempre. Le run già esistenti non vengono mai spostate.

Su Linux (dual boot, 2026-11-22) `disco_run` è un percorso Windows e le giunzioni non si creano: le run nuove
nascono direttamente in runs/<nome>.
"""
import os
import subprocess

from . import PROJECT_ROOT

RUNS = os.path.join(PROJECT_ROOT, "runs")


def disco_run():
    p = os.path.join(PROJECT_ROOT, "configs", "locale.yaml")
    if not os.path.exists(p):
        return None
    import yaml
    with open(p, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("disco_run") or None


def run_dir(name, crea=True):
    """Percorso di runs/<nome>. Se non esiste e `crea` è vero la crea (sul disco configurato, con giunzione)."""
    d = os.path.join(RUNS, name)
    if os.path.exists(d) or not crea:
        return d
    os.makedirs(RUNS, exist_ok=True)
    # Le cartelle che iniziano con "_" (prove automatiche, che poi vengono cancellate) restano locali.
    disco = None if name.startswith("_") or os.name != "nt" else disco_run()
    if disco and os.path.isdir(os.path.splitdrive(disco)[0] + os.sep):
        target = os.path.normpath(os.path.join(disco, name))
        os.makedirs(target, exist_ok=True)
        try:
            r = subprocess.run(["cmd", "/c", "mklink", "/J", d, target], capture_output=True, text=True)
            esito = f"{r.stdout.strip()} {r.stderr.strip()}"
            if r.returncode == 0:
                return d
        except OSError as e:
            esito = str(e)
        print(f"[rundirs] giunzione non riuscita ({esito}): uso {d}", flush=True)
    os.makedirs(d, exist_ok=True)
    return d
