"""Verifica che un training interrotto e ripreso dia gli stessi numeri di uno mai interrotto.

Esegue tre run della VQ-VAE (pesi nuovi, pochi passi), ognuno in processi separati come nella realtà:
  A1, A2: continui da 0 a N passi (misurano il non-determinismo di base della GPU);
  B:      fermato al passo N/2 con la stessa procedura della pausa, poi ripreso in un nuovo processo.
Confronta loss per passo, pesi, buffer (anche non persistenti), optimizer e stato dei dati.

Uso: .venv\\Scripts\\python.exe tests\\test_resume.py [--steps 12] [--keep]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "_resume_worker.py")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--keep", action="store_true", help="non cancellare le cartelle dei run di prova")
a = ap.parse_args()
base = os.path.join(ROOT, "runs", "_test_resume")
shutil.rmtree(base, ignore_errors=True)
env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}


def run(name, steps, stop_at=None):
    d = os.path.join(base, name)
    cmd = [sys.executable, "-I", WORKER, d, str(steps)] + ([str(stop_at)] if stop_at else [])
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-5000:])
        raise SystemExit(f"run {name} fallito (codice {p.returncode})")
    return d, p.stdout.strip().splitlines()[-1]


def losses(d):
    out = {}
    with open(os.path.join(d, "log.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "train":
                out[r["step"]] = r["loss"]
    return out


def final_state(d):
    from glob import glob
    path = sorted(glob(os.path.join(d, "checkpoints", "step_*.pt")))[-1]
    return torch.load(path, map_location="cpu", weights_only=False)


def max_diff(a, b, path=""):
    if isinstance(a, torch.Tensor):
        if a.dtype.is_floating_point:
            return (a.double() - b.double()).abs().max().item() if a.numel() else 0.0
        return 0.0 if torch.equal(a, b) else float("inf")
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"chiavi diverse in {path}"
        return max([max_diff(a[k], b[k], f"{path}.{k}") for k in a] or [0.0])
    if isinstance(a, (list, tuple)):
        return max([max_diff(x, y, path) for x, y in zip(a, b)] or [0.0])
    return 0.0 if a == b else float("inf")


N, half = a.steps, a.steps // 2
print(f"run A1: {N} passi continui ...", flush=True)
dA1, _ = run("A1", N)
print(f"run A2: {N} passi continui ...", flush=True)
dA2, _ = run("A2", N)
print(f"run B: pausa al passo {half} ...", flush=True)
dB, r1 = run("B", N, stop_at=half)
assert r1 == "paused", f"la prima parte di B doveva fermarsi in pausa, invece: {r1}"
print("run B: ripresa in un nuovo processo ...", flush=True)
_, r2 = run("B", N)
assert r2 == "done", f"la ripresa di B non è arrivata alla fine: {r2}"

lA1, lA2, lB = losses(dA1), losses(dA2), losses(dB)
assert sorted(lB) == list(range(1, N + 1)), f"passi registrati in B: {sorted(lB)}"
base_noise = max(abs(lA1[s] - lA2[s]) for s in lA1)
resume_diff = max(abs(lA1[s] - lB[s]) for s in lA1)
sA1, sA2, sB = final_state(dA1), final_state(dA2), final_state(dB)
keys = ["model", "model_non_persistent", "optimizer", "data"]
w_noise = {k: max_diff(sA1[k], sA2[k], k) for k in keys}
w_resume = {k: max_diff(sA1[k], sB[k], k) for k in keys}

print("\npasso  loss A1        loss A2        loss B (ripreso dopo il passo %d)" % half)
for s in range(1, N + 1):
    print(f"{s:5d}  {lA1[s]:.10f}  {lA2[s]:.10f}  {lB[s]:.10f}")
print(f"\ndifferenza massima di loss  A1-A2 (rumore GPU): {base_noise:.3e}   A1-B (ripresa): {resume_diff:.3e}")
for k in keys:
    print(f"differenza massima {k:22s} A1-A2: {w_noise[k]:.3e}   A1-B: {w_resume[k]:.3e}")
print(f"stato dati finale: A1 {sA1['data']}  B {sB['data']}")

exact = base_noise == 0.0 and all(v == 0.0 for v in w_noise.values())
if exact:
    ok = resume_diff == 0.0 and all(v == 0.0 for v in w_resume.values())
    criterio = "run continui identici bit per bit: la ripresa deve esserlo anch'essa"
else:
    tol = lambda x: max(10 * x, 1e-6)  # noqa: E731
    ok = resume_diff <= tol(base_noise) and all(w_resume[k] <= tol(w_noise[k]) for k in keys)
    criterio = "run continui non identici (non-determinismo GPU): la ripresa deve restare nello stesso ordine di grandezza"
print(f"\ncriterio: {criterio}")
print("ESITO: SUPERATO" if ok else "ESITO: FALLITO")
if not a.keep:
    shutil.rmtree(base, ignore_errors=True)
sys.exit(0 if ok else 1)
