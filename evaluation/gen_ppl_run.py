"""Gen. PPL con il protocollo del paper, usando gli script ufficiali senza modificarli.

Genera 256 storie per ciascuno dei 5 seed (42, 0, 1, 2, 3) con eval_shard.py, poi le valuta con
GPT-2 large tramite eval_merge.py (metrica "genppl"). Tutto offline.

Uso (dalla cartella del progetto):
    .venv\\Scripts\\python.exe -I evaluation\\gen_ppl_run.py --official
    .venv\\Scripts\\python.exe -I evaluation\\gen_ppl_run.py --run <nome>      (usa runs\\<nome>\\export)
"""
import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM = os.path.join(ROOT, "third_party", "ncm")
ART = os.path.join(ROOT, "artifacts", "tinystories")
REF = os.path.join(ROOT, "evaluation", "ref_models", "gpt2-large")

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--official", action="store_true", help="checkpoint ufficiale in artifacts/tinystories")
g.add_argument("--run", help="nome del run: usa runs/<nome>/export/generator.pt")
ap.add_argument("--n", type=int, default=256, help="campioni per seed")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", default="")
a = ap.parse_args()

if a.official:
    ck_dir = ART
    out = a.out or os.path.join(ROOT, "evaluation", "results", "genppl_ufficiale")
else:
    ck_dir = os.path.join(ROOT, "runs", a.run, "export")
    out = a.out or os.path.join(ROOT, "runs", a.run, "eval", "genppl")
    if not os.path.exists(os.path.join(ck_dir, "generator.pt")):
        sys.exit(f"manca {ck_dir}\\generator.pt: il run non ha esportato il generatore")
out = os.path.abspath(out)
os.makedirs(out, exist_ok=True)
if not os.path.exists(os.path.join(REF, "model.safetensors")):
    sys.exit(f"manca il modello di riferimento in {REF}")

cfg = OmegaConf.load(os.path.join(NCM, "config", "eval", "tinystories.yaml"))
tok_dir = ART
if not a.official:
    # Se la run usa un tokenizer nostro (base_config diverso da quello ufficiale), si valuta con il config
    # esportato insieme a quel tokenizer (runs/<tokenizer>/export/config.yaml): stessi livelli e stesso file.
    run_cfg_path = os.path.join(ROOT, "runs", a.run, "config.yaml")
    if os.path.exists(run_cfg_path):
        bc = str(OmegaConf.load(run_cfg_path).get("base_config", ""))
        bc = bc if os.path.isabs(bc) else os.path.join(ROOT, bc)
        if bc and os.path.normcase(os.path.normpath(bc)) != os.path.normcase(os.path.join(ART, "config.yaml")):
            cfg = OmegaConf.load(bc)
            tok_dir = os.path.dirname(bc)
            print(f"[gen_ppl] tokenizer della run: {tok_dir}", flush=True)
    # os.path.join con un percorso assoluto restituisce quello: tokenizer e vocabolario non stanno in export/.
    for key in ("tokenizer", "bpe_vocab", "bpe_merges"):
        cfg.checkpoints[key] = os.path.join(tok_dir, cfg.checkpoints[key])
# Generatore a cicli (2026-11-22): il checkpoint ha un blocco per ogni strato applicato (i pesi legati sono salvati
# come copie), quindi si valuta con un generatore normale di prima + blocco x ripetizioni + dopo strati: stessa funzione.
_rc = None if a.official else os.path.join(ROOT, "runs", a.run, "config.yaml")
if _rc and os.path.exists(_rc):
    _c = OmegaConf.load(_rc).get("cicli")
    if _c and int(_c.get("ripetizioni", 1)) > 1:
        cfg.model.n_layer = int(_c.prima) + int(_c.blocco) * int(_c.ripetizioni) + int(_c.dopo)
        print(f"[gen_ppl] generatore a cicli: {cfg.model.n_layer} strati", flush=True)
cfg.eval.ref_model = REF
cfg_path = os.path.join(out, "eval_config.yaml")
OmegaConf.save(cfg, cfg_path)

env = {**os.environ, "NCM_ROOT": NCM, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "PYTHONIOENCODING": "utf-8"}
py = sys.executable
shard = os.path.join(out, "shard_0.pkl")
cmd_gen = [py, os.path.join(NCM, "scripts", "eval_shard.py"), "--config", cfg_path, "--checkpoint-dir", ck_dir,
           "--shard", "0", "--nshards", "1", "--n", str(a.n), "--bs", str(a.bs), "--out", shard]
cmd_merge = [py, os.path.join(NCM, "scripts", "eval_merge.py"), "--config", cfg_path,
             "--glob", os.path.join(out, "shard_*.pkl")]

t0 = time.time()
print("[gen_ppl] generazione:", " ".join(cmd_gen), flush=True)
with open(os.path.join(out, "generazione.log"), "w", encoding="utf-8") as log:
    p = subprocess.run(cmd_gen, cwd=NCM, env=env, stdout=log, stderr=subprocess.STDOUT)
if p.returncode != 0:
    sys.exit(f"generazione fallita (codice {p.returncode}): vedi {out}\\generazione.log")
t_gen = time.time() - t0

print("[gen_ppl] valutazione con GPT-2 large ...", flush=True)
p = subprocess.run(cmd_merge, cwd=NCM, env=env, capture_output=True, text=True, encoding="utf-8")
with open(os.path.join(out, "merge.txt"), "w", encoding="utf-8") as f:
    f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
if p.returncode != 0:
    sys.exit(f"valutazione fallita (codice {p.returncode}): vedi {out}\\merge.txt")

per_seed, mean = [], None
for line in p.stdout.splitlines():
    m = re.match(r"^\s*(-?\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
    if m:
        per_seed.append({"seed": int(m.group(1)), "genppl": float(m.group(2)), "entropy": float(m.group(3))})
    m = re.match(r"^\s*mean\s+([\d.]+)\s+([\d.]+)\s+\(\+/-\s*([\d.]+)", line)
    if m:
        mean = {"genppl": float(m.group(1)), "entropy": float(m.group(2)), "std": float(m.group(3))}
if mean is None or not per_seed:
    sys.exit(f"risultato non trovato nell'output: vedi {out}\\merge.txt")

with open(shard, "rb") as f:  # file prodotto da questo stesso processo
    hyps = pickle.load(f)["hyps"]
with open(os.path.join(out, "campioni.txt"), "w", encoding="utf-8") as f:
    for i, h in enumerate(hyps[:10]):
        f.write(f"===== campione {i + 1} =====\n{h}\n\n")

res = {"genppl_mean": mean["genppl"], "genppl_std": mean["std"], "entropy_mean": mean["entropy"],
       "per_seed": per_seed, "n_samples": len(hyps), "samples_per_seed": a.n,
       "ref_model": REF, "checkpoint_dir": ck_dir, "secondi_generazione": round(t_gen, 1),
       "secondi_totali": round(time.time() - t0, 1), "comando": [py] + sys.argv,
       "data": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(out, "genppl.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"[gen_ppl] Gen. PPL {mean['genppl']:.2f} ± {mean['std']:.2f} | entropia {mean['entropy']:.4f} | "
      f"{len(hyps)} campioni | {res['secondi_totali']:.0f}s -> {out}", flush=True)
