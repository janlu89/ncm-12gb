"""Esegue un ciclo di micro-training in sequenza: per ogni run training (train_generator.py) e poi Gen. PPL
(gen_ppl_run.py). Alla fine scrive la tabella di confronto. Riprendibile: salta ciò che è già completato.

Il file del ciclo (YAML) contiene:
    nome: 2026-11-17_ciclo-corruzione
    base_overrides: [train.steps=1500, ...]        # comuni a tutti i run
    riferimento: A42                                # run con cui confrontare
    rumore: [A42, A43]                              # run che differiscono solo per il seed
    runs:
      - {id: A42, descr: "base", overrides: [train.seed=42]}
      - {id: V1,  descr: "livello 0 corrotto", overrides: [corruption.skip_level0=false]}

Avvio staccato (consigliato):
    .\\scripts\\avvia_detached.ps1 -Command '.venv\\Scripts\\python.exe -u -I scripts\\ciclo.py configs\\cicli\\<file>.yaml' -Log 'runs\\_cicli\\<nome>.log'
Pausa: creare runs\\_cicli\\STOP (il ciclo si ferma dopo il run in corso).
"""
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CICLI = os.path.join(ROOT, "runs", "_cicli")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_dir(name):
    return os.path.join(ROOT, "runs", name)


def last_event(name):
    p = os.path.join(run_dir(name), "log.jsonl")
    if not os.path.exists(p):
        return None
    ev = None
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") in ("start", "resume", "paused", "done", "error"):
                ev = r["event"]
    return ev


def summary(name):
    d = run_dir(name)
    out = {"run": name}
    p = os.path.join(d, "log.jsonl")
    if os.path.exists(p):
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        ev = [r for r in recs if r.get("event") == "eval"]
        tr = [r for r in recs if r.get("event") == "train"]
        if ev:
            out["val_loss"] = ev[-1]["val_loss"]
            out["val_L1"] = ev[-1].get("val_L1")
            out["val_L11"] = ev[-1].get("val_L11")
        if tr:
            out["seq_s"] = sum(r["seq_per_s"] for r in tr[-20:]) / len(tr[-20:])
            out["gpu_temp_max"] = max(r.get("gpu_temp_c", 0) for r in tr)
        st = [r for r in recs if r.get("event") == "start"]
        dn = [r for r in recs if r.get("event") == "done"]
        if st and dn:
            t = lambda r: time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))  # noqa: E731
            out["minuti"] = (t(dn[-1]) - t(st[0])) / 60
    g = os.path.join(d, "eval", "genppl", "genppl.json")
    if os.path.exists(g):
        j = json.load(open(g, encoding="utf-8"))
        out.update(genppl=j["genppl_mean"], genppl_std=j["genppl_std"], entropia=j["entropy_mean"])
    return out


def _mappa_nomi():
    """Nomi vecchi -> nomi standard (mappa facoltativa in runs/_cicli/rinomina_run.json)."""
    p = os.path.join(CICLI, "rinomina_run.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)["mappa"]


def nome_da_impostazioni(spec, r):
    """Nome standard (common/nomi.py) calcolato dalla config completa della run."""
    from omegaconf import OmegaConf
    from common.nomi import nome_standard
    base = OmegaConf.load(os.path.join(ROOT, spec.get("config", "configs/stage2_generator.yaml")))
    cfg = OmegaConf.merge(base, OmegaConf.from_dotlist(list(spec.get("base_overrides", [])) +
                                                       list(r.get("overrides", []))))
    return nome_standard(cfg)


def run_name(spec, r):
    """Nome della cartella del run. Con `run: <nome>` nel file del ciclo si riusa un run già fatto invece di
    rifarlo. I nomi vecchi passano dalla mappa della rinomina; le run nuove prendono il nome standard, quindi una
    run con le stesse impostazioni di una già fatta viene riusata, non rifatta."""
    m = _mappa_nomi()
    if r.get("run"):
        return m.get(r["run"], r["run"])
    vecchio = f"{spec['nome']}_{r['id']}"
    if vecchio in m:
        return m[vecchio]
    if os.path.exists(run_dir(vecchio)):
        return vecchio
    return nome_da_impostazioni(spec, r)


def rows_of(spec):
    rows = []
    for r in spec["runs"]:
        s = summary(run_name(spec, r))
        s.update(id=r["id"], descr=r.get("descr", ""), overrides=" ".join(r.get("overrides", [])))
        rows.append(s)
    return rows


STATO = {"testo": "avvio"}


def set_state(spec, testo):
    STATO["testo"] = testo
    print(f"[ciclo] {time.strftime('%H:%M')} {testo}", flush=True)
    write_outputs(spec)


def write_outputs(spec):
    with open(os.path.join(CICLI, f"{spec['nome']}.md"), "w", encoding="utf-8") as f:
        f.write(table(spec))
    json.dump({"nome": spec["nome"], "riferimento": spec.get("riferimento"), "rumore": spec.get("rumore", []),
               "aggiornato": time.strftime("%Y-%m-%d %H:%M:%S"), "stato": STATO["testo"], "righe": rows_of(spec)},
              open(os.path.join(CICLI, f"{spec['nome']}.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def table(spec):
    rows = []
    for r in spec["runs"]:
        s = summary(run_name(spec, r))
        s.update(id=r["id"], descr=r.get("descr", ""), overrides=" ".join(r.get("overrides", [])))
        rows.append(s)
    by = {r["id"]: r for r in rows}
    noise = None
    nz = [by[i].get("genppl") for i in spec.get("rumore", []) if by.get(i, {}).get("genppl") is not None]
    if len(nz) >= 2:
        noise = max(nz) - min(nz)
    ref = by.get(spec.get("riferimento"), {})
    lines = [f"# Ciclo {spec['nome']}", "",
             f"Aggiornato: {time.strftime('%Y-%m-%d %H:%M')}. Comuni a tutti i run: `{' '.join(spec.get('base_overrides', []))}`", "",
             f"Rumore tra seed (Gen. PPL, run {', '.join(spec.get('rumore', []))}): "
             + (f"{noise:.2f}" if noise is not None else "non ancora misurato"), "",
             "| Run | Differenza dalla base | Val loss | Val L1 | Val L11 | Gen. PPL | ± seed valutazione | Entropia | Δ Gen. PPL vs rif. | seq/s | Minuti |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    for r in rows:
        delta = None
        if r.get("genppl") is not None and ref.get("genppl") is not None:
            delta = r["genppl"] - ref["genppl"]
        lines.append(f"| {r['id']} | {r['descr']} (`{r['overrides']}`) | {f(r.get('val_loss'))} | {f(r.get('val_L1'))} | "
                     f"{f(r.get('val_L11'))} | {f(r.get('genppl'), 2)} | {f(r.get('genppl_std'), 2)} | {f(r.get('entropia'))} | "
                     f"{'—' if delta is None else f'{delta:+.2f}'} | {f(r.get('seq_s'), 0)} | {f(r.get('minuti'), 0)} |")
    lines += ["", "Regola di decisione (il piano dell'esperimento, 1b): una variante vince solo se la sua Gen. PPL è più bassa "
              "di quella del riferimento di oltre il 5% del riferimento, senza abbassare l'entropia di più di 0,1."]
    if ref.get("genppl") is not None:
        lines.append("")
        for r in rows:
            if r["id"] in spec.get("rumore", []) or r["id"] == spec.get("riferimento") or r.get("genppl") is None:
                continue
            rel = (r["genppl"] - ref["genppl"]) / ref["genppl"]
            win = (ref["genppl"] - r["genppl"] > 0.05 * ref["genppl"]) and (r["entropia"] >= ref["entropia"] - 0.1)
            lines.append(f"- {r['id']} ({rel:+.1%}): {'MIGLIORA' if win else 'nessun miglioramento oltre il 5%'}")
    return "\n".join(lines) + "\n"


def main():
    spec = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
    os.makedirs(CICLI, exist_ok=True)
    stop_file = os.path.join(CICLI, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    out_md = os.path.join(CICLI, f"{spec['nome']}.md")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    wait_for = spec.get("attendi")
    if wait_for:
        wf = os.path.join(ROOT, wait_for)
        while not os.path.exists(wf):
            set_state(spec, f"in attesa di {wait_for}")
            time.sleep(30)
    for r in spec["runs"]:
        name = run_name(spec, r)
        d = run_dir(name)
        if last_event(name) != "done" or not os.path.exists(os.path.join(d, "export", "generator.pt")):
            if os.path.exists(stop_file):
                set_state(spec, "fermato (file STOP)")
                break
            if r.get("run") and not os.path.exists(os.path.join(d, "config.yaml")):
                # This row reuses a run made by an earlier cycle; its overrides here are incomplete on purpose.
                print(f"[ciclo] {r['id']}: run '{name}' not found in runs/. It is a reference to an earlier run: "
                      f"train it first with the settings in results/runs/{name}/config.yaml.", flush=True)
                set_state(spec, f"fermato: manca il run di riferimento {name}")
                break
            cmd = [PY, "-u", "-X", "utf8", "-I",os.path.join(ROOT, "generator", "train_generator.py"), "--run", name] + \
                  list(spec.get("base_overrides", [])) + list(r.get("overrides", []))
            set_state(spec, f"training {r['id']} ({name})")
            from common.rundirs import run_dir as prepara_run_dir
            prepara_run_dir(name)
            with open(os.path.join(d, "console.log"), "a", encoding="utf-8") as log:
                p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if last_event(name) != "done":
                set_state(spec, f"fermato: {r['id']} non completato (codice {p.returncode}, evento {last_event(name)})")
                break
        if not os.path.exists(os.path.join(d, "eval", "genppl", "genppl.json")):
            set_state(spec, f"valutazione Gen. PPL di {r['id']}")
            p = subprocess.run([PY, "-u", "-X", "utf8", "-I",os.path.join(ROOT, "evaluation", "gen_ppl_run.py"), "--run", name],
                               cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            print((p.stdout or "")[-400:], (p.stderr or "")[-1500:] if p.returncode else "", flush=True)
        write_outputs(spec)
    else:
        set_state(spec, "completato")
    write_outputs(spec)
    print(f"[ciclo] tabella: {out_md}", flush=True)
    # Codice di uscita: 0 solo se il ciclo è completato, così una catena (&&) si ferma al primo problema.
    sys.exit(0 if STATO["testo"] == "completato" else 1)


if __name__ == "__main__":
    main()
