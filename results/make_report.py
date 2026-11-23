"""Rebuild results/runs_index.csv and results/RUNS.md (the run-by-run report) from the files in results/runs/.

    python results/make_report.py

Only the standard library and PyYAML are needed. Nothing is typed by hand: every number comes from a run's
config.yaml, log.jsonl, eval/genppl/genppl.json or export/export.json.
"""
import csv
import glob
import json
import os
import re
from datetime import datetime

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def events(run):
    p = os.path.join(RUNS, run, "log.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_json(*parts):
    p = os.path.join(RUNS, *parts)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def g(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def minutes(ev):
    """Wall-clock time from the first to the last logged event, without the time spent paused or switched off
    (a gap that ends with a `resume` or a second `start` event is not counted)."""
    ev = [e for e in ev if "time" in e]
    ts = [datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S") for e in ev]
    tot = sum((b - a).total_seconds() for a, b, e in zip(ts, ts[1:], ev[1:]) if e.get("event") not in ("resume", "start"))
    return tot / 60 if ts else None


def tok_key(label):
    """Sort key for a generator's tokenizer: official first, then by levels, stories seen, snapshot step."""
    if label.startswith("official"):
        return (0, 0, 0, 0, label)
    name, _, snap = label.partition(" @ step ")
    m = re.match(r"mytok-(\d+)[a-z]+[^_]*_(\d+)_", name)
    lv, seen = (int(m.group(1)), int(m.group(2))) if m else (99, 0)
    return (1, lv, seen, int(snap.replace(",", "")) if snap else 10 ** 9, name)


def tok_title(label):
    if label == "official":
        return "official (authors)"
    if label.startswith("official"):
        return "official, codes computed in fp64"
    name, _, snap = label.partition(" @ step ")
    return f"`{name}`" + (f", snapshot at step {snap}" if snap else "")


def tok_label(cfg):
    bc = str(cfg.get("base_config", "")).replace("\\", "/")
    if "/runs/" in "/" + bc or bc.startswith("runs/"):
        parts = bc.split("/")
        name = parts[parts.index("runs") + 1]
        snap = [x for x in parts if x.startswith("passo_")]
        return name + (f" @ step {int(snap[0][6:]):,}" if snap else "")
    return "official" + (" (fp64 codes)" if cfg.get("encode_fp64") else "")


def corruption(cfg):
    c = cfg.get("corruption", {})
    if not c.get("enabled", True):
        return "none"
    if c.get("mode") == "per_example_uniform":
        return f"uniform 0-{c.get('rate_max', 1.0)}"
    a, b = float(c["rate_first"]), float(c["rate_last"])
    s = f"constant {a:g}" if abs(a - b) < 1e-9 else f"linear {a:g} to {b:g}"
    return s + ("" if c.get("skip_level0", True) else ", incl. level 0")


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
    run = os.path.basename(d)
    if not os.path.isdir(d) or run.startswith("_"):
        continue
    cp = os.path.join(d, "config.yaml")
    cfg = yaml.safe_load(open(cp, encoding="utf-8")) if os.path.exists(cp) else {}
    ev = events(run)
    t = cfg.get("train", {})
    stage = "tokenizer" if "tokenizer" in cfg and isinstance(cfg["tokenizer"], dict) and "livelli" in cfg["tokenizer"] \
        else "generator"
    evals = [e for e in ev if e.get("event") == "eval"]
    trains = [e for e in ev if e.get("event") == "train"]
    last = ev[-1]["event"] if ev else ""
    batch = int(t.get("micro_batch", 0)) * int(t.get("accum", 0))
    r = {"run": run, "stage": stage, "status": last, "steps": t.get("steps"), "batch": batch,
         "stories_seen": (t.get("steps") or 0) * batch, "seed": t.get("seed"), "lr": t.get("lr"),
         "warmup": t.get("warmup"), "last_step": max([e.get("step", 0) for e in ev] or [0]),
         "resumed": sum(1 for e in ev if e.get("event") == "resume"),
         "minutes": minutes(ev), "peak_vram_gb": max([e.get("peak_vram_gb", 0) for e in trains] or [0]) or None,
         "seq_per_s": (sorted(e["seq_per_s"] for e in trains if "seq_per_s" in e) or [None])[len(trains) // 2] if trains else None,
         "val_loss": evals[-1].get("val_loss") if evals else None,
         "notes": "interrupted, kept for the record" if run.endswith("_interrotta") else
                  "cancelled, kept for the record" if run.endswith("_annullata") else ""}
    if stage == "tokenizer":
        tk = cfg["tokenizer"]
        rim = tk.get("rimedi") or {}
        ex = load_json(run, "export", "export.json") or (evals[-1] if evals else {})
        spacing = {"ufficiale": "official spacing", "lineare": "linear", "quadratica": "quadratic",
                   "esponenziale": "exponential"}
        r.update({"levels": tk.get("livelli"), "schema": spacing.get(tk.get("schema"), tk.get("schema")),
                  "init_from_data": bool(rim.get("init_dati")), "restart_until_step": rim.get("riavvio_fino_al_passo", 0),
                  "level0_norm": bool(tk.get("norma_livello0", True)),
                  "token_accuracy": ex.get("acc_token"), "perfect_stories": ex.get("sequenze_perfette"),
                  "codes_used_mean": ex.get("codici_usati_media"), "codes_used_per_level": ex.get("codici_usati_per_livello"),
                  "tokenizer": "", "gen_ppl": None, "gen_ppl_std": None, "entropy": None})
    else:
        gp = load_json(run, "eval", "genppl", "genppl.json") or {}
        r.update({"tokenizer": tok_label(cfg), "corruption": corruption(cfg),
                  "flex": bool(g(cfg, "model_overrides.use_flex_attention", False)),
                  "gen_ppl": gp.get("genppl_mean"), "gen_ppl_std": gp.get("genppl_std"),
                  "entropy": gp.get("entropy_mean"), "per_seed": [s["genppl"] for s in gp.get("per_seed", [])],
                  "val_per_level": evals[-1].get("val_per_livello") if evals else None})
    rows.append(r)

# ---------------------------------------------------------------- CSV
cols = ["run", "stage", "status", "steps", "last_step", "batch", "stories_seen", "seed", "lr", "warmup", "tokenizer", "gen_ppl",
        "gen_ppl_std", "entropy", "val_loss", "minutes", "peak_vram_gb", "resumed", "notes"]
with open(os.path.join(HERE, "runs_index.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow(["" if r.get(c) is None else (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols])

# ---------------------------------------------------------------- Markdown
toks = [r for r in rows if r["stage"] == "tokenizer"]
gens = [r for r in rows if r["stage"] == "generator"]
done = [r for r in gens if r["gen_ppl"] is not None]
other = [r for r in gens if r["gen_ppl"] is None]
L = []
A = L.append
A("# Run-by-run report\n")
A("Generated by `results/make_report.py` from the files in `results/runs/`. Do not edit by hand.\n")
A(f"{len(rows)} runs: {len(toks)} tokenizers, {len(done)} generators with a Gen. PPL measurement, "
  f"{len(other)} generators interrupted or cancelled before evaluation.\n")
A("How to read it:\n")
A("- **Gen. PPL**: generative perplexity of 1,280 generated stories (256 for each of 5 sampling seeds) under GPT-2 "
  "large; lower is better. **± eval** is the standard deviation across the 5 sampling seeds of the *same* model, not "
  "across training seeds.")
A("- **Entropy**: entropy (nats) of the token frequencies inside each generated story, averaged over the stories; a "
  "drop signals repetitive text.")
A("- **Val loss**: for generators, mean cross-entropy (nats) over the predicted codes on 4,096 validation stories; for "
  "tokenizers, their own validation loss. Neither is comparable between different tokenizers.")
A("- **Corruption** is the fraction of input codes replaced by random ones during training, per level: `constant 0.5` "
  "on every level, or `linear 0.75 to 0.25` from the coarsest to the finest level (not over time). **Flex** = upstream "
  "FlexAttention switched on. **LR** is the peak learning rate.")
A("- **Init from data** = codebooks initialised with k-means; **dead-code restart** = unused codes re-seeded from the "
  "batch, only up to the given step; **level-0 norm** = normalisation of the level-0 path.")
A("- **Stories seen** = steps × batch. **Seed** is the training seed (model initialisation, corruption noise); the "
  "order of the training data is the same in every run.")
A("- **Min** is wall-clock time from the first to the last logged event, including validation, compilation and "
  "checkpoints, excluding time spent paused. A run resumed after an interruption counts the repeated steps twice. "
  "**VRAM** is the peak memory allocated by PyTorch, in GiB.")
A("- In `runs_index.csv`, `steps` and `stories_seen` are the planned values; `last_step` is the last step logged.")
A("- Run names are explained in `results/README.md`.")
A("- Each run's exact settings are in `results/runs/<run>/config.yaml`.\n")

A("## 1. Tokenizers\n")
A("| Run | Levels | Steps × batch | Stories seen | Seed | Init from data | Dead-code restart until step | Level-0 norm | "
  "Val loss | Token accuracy | Perfect stories | Codes in use (mean) | Min |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(toks, key=lambda r: (r["levels"], r["stories_seen"], r["run"])):
    A(f"| `{r['run']}` | {r['levels']} ({r['schema']}) | {r['steps']:,} × {r['batch']:,} | {r['stories_seen']:,} | {r['seed']} | "
      f"{'yes' if r['init_from_data'] else 'no'} | {r['restart_until_step'] or '-'} | {'yes' if r['level0_norm'] else 'no'} | "
      f"{fmt(r['val_loss'], 4)} | {fmt(r['token_accuracy'], 4)} | {fmt(r['perfect_stories'], 3)} | {fmt(r['codes_used_mean'], 3)} | {fmt(r['minutes'], 0)} |")
A("\nToken accuracy and perfect stories are measured on 4,096 validation stories after encoding and decoding. "
  "\"Codes in use\" is the fraction of the 4,096 codes of a level that appear at least once on those stories, "
  "averaged over levels.\n")

A("### Codes in use per level (fraction of 4,096)\n")
A("Only 12-level tokenizers (level 0 = one code per story, level 11 = 128 codes).\n")
A("| Run | " + " | ".join(f"L{i}" for i in range(12)) + " |")
A("|---|" + "---|" * 12)
for r in sorted(toks, key=lambda r: (r["stories_seen"], r["run"])):
    u = r.get("codes_used_per_level") or []
    if len(u) == 12:
        A(f"| `{r['run']}` | " + " | ".join(f"{x:.3f}" for x in u) + " |")

A("\n## 2. Generators\n")
A("Grouped by tokenizer, then by stories seen. A generator on a 12-level tokenizer has 88.6M parameters; "
  "the count changes with the number of levels (87.4M with 8, 89.8M with 16, 92.8M with 26, 96.9M with 40). "
  "Names containing `cicli<a>-<b>x<r>-<c>` are looped generators: a layers, then a block of b layers applied r times "
  "with shared weights, then c layers (4+2x2+4: 74.4M parameters; 4+4x2+4 and 4+4x3+4: 88.6M). `-cresce<k>` = the "
  "repetitions after the first are switched on at step k; `-cresce<k2>-<k3>` = repetition 2 at step k2, 3 at step k3.\n")
groups = {}
for r in done:
    groups.setdefault(r["tokenizer"], []).append(r)
for tk in sorted(groups, key=tok_key):
    A(f"### Tokenizer: {tok_title(tk)}\n")
    A("| Run | Stories seen | Steps × batch | Corruption | LR | Flex | Seed | Val loss | Gen. PPL | ± eval | Entropy | Min | VRAM GiB |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(groups[tk], key=lambda r: (r["stories_seen"], r["corruption"], r["lr"], r["flex"], r["seed"], r["run"])):
        A(f"| `{r['run']}` | {r['stories_seen']:,} | {r['steps']:,} × {r['batch']} | {r['corruption']} | {r['lr']:g} | "
          f"{'yes' if r['flex'] else 'no'} | {r['seed']} | {fmt(r['val_loss'], 3)} | **{fmt(r['gen_ppl'])}** | {fmt(r['gen_ppl_std'])} | "
          f"{fmt(r['entropy'], 3)} | {fmt(r['minutes'], 0)} | {fmt(r['peak_vram_gb'], 1)} |")
    A("")

A("### Validation loss per level\n")
A("Final validation loss of each generator on the levels it predicts (1 to 11), 12-level tokenizers only.\n")
A("| Run | " + " | ".join(f"L{i}" for i in range(1, 12)) + " |")
A("|---|" + "---|" * 11)
for r in sorted(done, key=lambda r: (tok_key(r["tokenizer"]), r["stories_seen"], r["run"])):
    v = r.get("val_per_level") or []
    if len(v) == 11:
        A(f"| `{r['run']}` | " + " | ".join(f"{x:.2f}" for x in v) + " |")

A("\n### Gen. PPL for each sampling seed\n")
A("Sampling seeds 42, 0, 1, 2, 3 (256 stories each).\n")
A("| Run | 42 | 0 | 1 | 2 | 3 |")
A("|---|---|---|---|---|---|")
for r in sorted(done, key=lambda r: (tok_key(r["tokenizer"]), r["stories_seen"], r["run"])):
    ps = r.get("per_seed") or []
    if len(ps) == 5:
        A(f"| `{r['run']}` | " + " | ".join(f"{x:.2f}" for x in ps) + " |")

A("\n## 3. Runs without a result\n")
A("Kept so that the record is complete. Any validation loss listed for them in `runs_index.csv` is an intermediate one.\n")
A("| Run | Last event | Last step | Of | Why |")
A("|---|---|---|---|---|")
for r in other:
    why = r["notes"] or "not evaluated"
    if r["last_step"] >= (r["steps"] or 0):
        why += "; all steps logged, stopped before the final validation and export"
    A(f"| `{r['run']}` | {r['status']} | {r['last_step']:,} | {r['steps']:,} | {why} |")
A("")
with open(os.path.join(HERE, "RUNS.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L))
print(f"runs_index.csv: {len(rows)} rows; RUNS.md: {len(L)} lines; tokenizers {len(toks)}, generators {len(done)} + {len(other)}")
