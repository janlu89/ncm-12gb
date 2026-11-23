"""Stadio 2: training del generatore NCM sopra un tokenizer VQ-VAE congelato.

Uso (dalla cartella del progetto):
    .venv\\Scripts\\python.exe -I generator\\train_generator.py --run <nome> [--config configs\\stage2_generator.yaml] [chiave=valore ...]

Esempio di prova breve:
    .venv\\Scripts\\python.exe -I generator\\train_generator.py --run prova train.steps=300 train.accum=8 train.warmup=30 train.eval_every=100

Pausa: pulsante della dashboard, oppure creare runs\\<nome>\\STOP. Ripresa: rilanciare lo stesso comando.
Alla fine scrive runs\\<nome>\\export\\generator.pt (formato compatibile con eval_shard.py) e alcune storie di prova.
La logica dello stadio è in generator/stage2.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import TrainStories  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from generator.stage2 import P, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="nome del run: cartella runs/<nome>; \"auto\" = nome standard dalle impostazioni (common/nomi.py)")
ap.add_argument("--config", default="configs/stage2_generator.yaml")
ap.add_argument("overrides", nargs="*", help="modifiche alla config, es. train.steps=300")
ap.add_argument("--estendi", type=int, default=0,
                help="run esistente con train.schedule=wsd e senza discesa: porta train.steps a questo valore e riprende")
a = ap.parse_args()
if a.run == "auto":
    from common.nomi import nome_standard
    a.run = nome_standard(OmegaConf.merge(OmegaConf.load(P(a.config)), OmegaConf.from_dotlist(a.overrides)))
    print(f"[nome] {a.run}", flush=True)

from common.rundirs import run_dir as prepara_run_dir  # noqa: E402

run_dir = prepara_run_dir(a.run)  # run nuove: sul disco indicato in configs/locale.yaml, collegate in runs/
saved_cfg = os.path.join(run_dir, "config.yaml")
if os.path.exists(saved_cfg):
    # Ripresa: si usa la config salvata alla prima partenza, per non cambiare il run a metà.
    config = OmegaConf.load(saved_cfg)
    if a.estendi:
        # Allungare una run (2026-11-22): solo con lr costante (wsd senza discesa), altrimenti la curva cambierebbe.
        assert str(config.train.get("schedule", "cosine")) == "wsd" and int(config.train.get("decay_steps", 0) or 0) == 0, \
            "--estendi vale solo per run con train.schedule=wsd e train.decay_steps=0"
        assert a.estendi > int(config.train.steps), f"--estendi deve superare i passi attuali ({config.train.steps})"
        print(f"[estendi] train.steps {config.train.steps} -> {a.estendi}", flush=True)
        config.train.steps = int(a.estendi)
        OmegaConf.save(config, saved_cfg)
    if a.overrides:
        print("[config] run esistente: le modifiche da riga di comando sono ignorate, si usa", saved_cfg)
else:
    config = OmegaConf.merge(OmegaConf.load(P(a.config)), OmegaConf.from_dotlist(a.overrides))
    OmegaConf.save(config, saved_cfg)

s = build(config)
codes_file = config.data.get("codes_file", None)
# Con i codici pre-calcolati il batch contiene già i codici VQ: niente encode a ogni passo.
base_loss = s.loss_from_idx if codes_file else s.loss_fn
model_for_loss = s.net
if config.train.get("compile", False):
    import torch
    # Si compila solo il forward del modello. Il trainer continua a usare s.net (stessi parametri) per
    # optimizer, checkpoint ed export, così le chiavi dei pesi non prendono il prefisso "_orig_mod.".
    # La valutazione resta non compilata (batch diverso, FP32).
    model_for_loss = torch.compile(s.net, mode=str(config.train.get("compile_mode", "default")))
_tr = {}


def loss_fn(model, batch):
    if "t" in _tr:                    # generatore che cresce: stato delle ripetizioni dal passo (vale anche alla ripresa)
        s.crescita(_tr["t"].step)
    return base_loss(model_for_loss, batch)


if codes_file:
    from common.data import TrainCodes
    data = TrainCodes(P(codes_file), int(config.data.seed))
    assert data.length == int(config.data.length), "lunghezza dei codici diversa da data.length"
else:
    data = TrainStories(P(config.data.dir), int(config.data.length), int(config.data.seed))
_tr["t"] = Trainer(run_dir, config, s.net, data, loss_fn, s.eval_fn, step_hook=s.crescita)
status = _tr["t"].run()
s.crescita(_tr["t"].step)   # anche se la ripresa trova il training già finito: export e storie con lo stato giusto
if status == "done":
    s.export(s.net, run_dir)
print(status)
