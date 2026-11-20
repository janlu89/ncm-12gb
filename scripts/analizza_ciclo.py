"""Stampa i dati di dettaglio di un ciclo: loss per livello contro l'ufficiale, Gen. PPL per seed, velocità.

Uso: .venv\\Scripts\\python.exe -X utf8 -I scripts\\analizza_ciclo.py <nome ciclo> [id ...]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ciclo = sys.argv[1]
spec = json.load(open(os.path.join(ROOT, "runs", "_cicli", f"{ciclo}.json"), encoding="utf-8"))
ids = sys.argv[2:] or [r["id"] for r in spec["righe"]]
off = json.load(open(os.path.join(ROOT, "evaluation", "results", "2026-11-16_riferimento_ufficiale.json"),
                     encoding="utf-8"))["generatore"]["ufficiale"]["per_livello"]

print("loss di validazione per livello (più bassa = meglio)")
print(f"{'':12s}" + "".join(f"{'L' + str(i + 1):>7s}" for i in range(len(off))))
print(f"{'ufficiale':12s}" + "".join(f"{p['loss']:7.2f}" for p in off))
rows = {}
for r in spec["righe"]:
    if r["id"] not in ids:
        continue
    d = os.path.join(ROOT, "runs", r["run"])
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"), encoding="utf-8") if l.strip()]
    ev = [x for x in recs if x["event"] == "eval"]
    tr = [x for x in recs if x["event"] == "train"]
    g = json.load(open(os.path.join(d, "eval", "genppl", "genppl.json"), encoding="utf-8"))
    rows[r["id"]] = (ev, tr, g)
    print(f"{r['id']:12s}" + "".join(f"{v:7.2f}" for v in ev[-1]["val_per_livello"]))

print("\ndettaglio")
for i, (ev, tr, g) in rows.items():
    last = tr[-20:]
    vals = " -> ".join("{:.3f}@{}".format(e["val_loss"], e["step"]) for e in ev)
    seeds = [s["genppl"] for s in g["per_seed"]]
    speed = sum(x["seq_per_s"] for x in last) / len(last)
    tmax = max(x.get("gpu_temp_c", 0) for x in tr)
    print("{:5s} val {} | train {:.3f} | GenPPL {:.2f} per seed {} | entropia {:.3f} | {:.0f} seq/s | picco {:.2f} GB | "
          "temp max {} °C | generazione {:.0f} s".format(i, vals, tr[-1]["loss"], g["genppl_mean"], seeds,
                                                         g["entropy_mean"], speed, tr[-1]["peak_vram_gb"], tmax,
                                                         g["secondi_generazione"]))
