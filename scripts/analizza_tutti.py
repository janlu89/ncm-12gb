"""Analisi di tutte le run valutate a un dato budget: coppie di seed, rumore, GPU, tempi, correlazioni.

Uso: .venv\\Scripts\\python.exe -X utf8 -I scripts\\analizza_tutti.py [storie=51200]
"""
import json
import math
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 51200
default = yaml.safe_load(open(os.path.join(ROOT, "configs", "stage2_generator.yaml"), encoding="utf-8"))


def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        out.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else out.__setitem__(p + k, v)
    return out


dflat = flat(default)
IGNORE = {"train.seed", "train.steps", "train.accum", "train.warmup", "train.eval_every", "train.keep_ckpts",
          "train.log_every", "data.codes_file"}
rows = []
for n in sorted(os.listdir(RUNS)):
    d = os.path.join(RUNS, n)
    g = os.path.join(d, "eval", "genppl", "genppl.json")
    if n.startswith("_") or not os.path.exists(g):
        continue
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml"), encoding="utf-8"))
    t = cfg["train"]
    if t["steps"] * t["micro_batch"] * t["accum"] != budget:
        continue
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"), encoding="utf-8") if l.strip()]
    tr = [r for r in recs if r["event"] == "train"]
    ev = [r for r in recs if r["event"] == "eval"]
    gp = json.load(open(g, encoding="utf-8"))
    f = flat(cfg)
    variant = " ".join(f"{k.split('.', 1)[1]}={v}" for k, v in sorted(f.items())
                       if k not in IGNORE and dflat.get(k, "§") != v) or "base"
    ts = lambda r: time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))  # noqa: E731
    start = next(r for r in recs if r["event"] == "start")
    done = next(r for r in recs if r["event"] == "done")
    mid = tr[len(tr) // 4:]
    rows.append({"run": n, "variant": variant, "seed": t["seed"], "genppl": gp["genppl_mean"],
                 "sem": gp["genppl_std"] / math.sqrt(len(gp["per_seed"])), "ent": gp["entropy_mean"],
                 "val": ev[-1]["val_loss"], "train": tr[-1]["loss"],
                 "seq_s": sum(r["seq_per_s"] for r in mid) / len(mid),
                 "util": sum(r["gpu_util_pct"] for r in mid) / len(mid),
                 "watt": sum(r["gpu_power_w"] for r in mid) / len(mid),
                 "temp": max(r["gpu_temp_c"] for r in tr), "gb": tr[-1]["peak_vram_gb"],
                 "min_tot": (ts(done) - ts(start)) / 60, "min_primo_log": (ts(tr[0]) - ts(start)) / 60})

print(f"budget {budget:,} storie — {len(rows)} run\n")
print(f"{'variante':58s} seed  GenPPL  ±sem   entr   val    train  seq/s util%  W   °C  GB   min  compil.")
for r in sorted(rows, key=lambda r: r["genppl"]):
    print(f"{r['variant'][:58]:58s} {r['seed']:4d} {r['genppl']:7.2f} {r['sem']:5.2f} {r['ent']:6.3f} {r['val']:6.3f} "
          f"{r['train']:6.3f} {r['seq_s']:5.0f} {r['util']:5.0f} {r['watt']:4.0f} {r['temp']:3d} {r['gb']:5.2f} "
          f"{r['min_tot']:5.1f} {r['min_primo_log']:6.1f}")

# coppie di seed: stesso variante, seed diversi
by = {}
for r in rows:
    by.setdefault(r["variant"], {})[r["seed"]] = r
pairs = {v: s for v, s in by.items() if len(s) >= 2}
print("\ncoppie con due seed (stessa configurazione):")
d2 = []
for v, s in pairs.items():
    vals = [s[k]["genppl"] for k in sorted(s)]
    d = vals[0] - vals[1]
    d2.append(d * d / 2)
    print(f"  {v[:58]:58s} {vals}  differenza {d:+.2f}")
sd = math.sqrt(sum(d2) / len(d2)) if d2 else float("nan")
print(f"\ndeviazione standard tra seed di training (stima da {len(d2)} coppie): {sd:.2f} punti di Gen. PPL")
print(f"-> differenza tra due run singole: sd {sd * math.sqrt(2):.2f}; soglia a 2 sd: {2 * sd * math.sqrt(2):.1f}")

base = by.get("base", {})
print("\ndifferenze appaiate rispetto alla base (stesso seed):")
for v, s in sorted(by.items(), key=lambda kv: min(x["genppl"] for x in kv[1].values())):
    if v == "base":
        continue
    ds = [(k, s[k]["genppl"] - base[k]["genppl"]) for k in sorted(s) if k in base]
    if not ds:
        continue
    mean = sum(d for _, d in ds) / len(ds)
    se = sd * math.sqrt(2) / math.sqrt(len(ds))
    print(f"  {v[:58]:58s} " + "  ".join(f"seed {k}: {d:+.2f}" for k, d in ds) +
          f"  | media {mean:+.2f} = {mean / se:+.1f} errori standard ({len(ds)} seed)")


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


print("\ncorrelazione con la Gen. PPL su tutte le run (1 = stessa classifica):")
for k, lab in (("val", "val loss"), ("train", "loss di training finale"), ("ent", "entropia")):
    print(f"  {lab:26s} r = {corr([r[k] for r in rows], [r['genppl'] for r in rows]):+.2f}")
