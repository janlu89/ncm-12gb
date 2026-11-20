r"""Stadio 1: training del tokenizer VQ-VAE nostro.

Uso (dalla cartella del progetto):
    .venv\Scripts\python.exe -I tokenizer\train_vqvae.py --run <nome> [--config configs\stage1_tokenizer.yaml] [chiave=valore ...]

Esempio di prova breve:
    .venv\Scripts\python.exe -I tokenizer\train_vqvae.py --run prova_tok train.steps=200 train.accum=1 train.warmup=20 train.eval_every=100

Pausa: pulsante della dashboard, oppure creare runs\<nome>\STOP. Ripresa: rilanciare lo stesso comando.
Alla fine scrive runs\<nome>\export\ (tokenizer.pt + config.yaml nel formato upstream, export.json con le misure).
Per addestrare un generatore sopra: generator\train_generator.py ... base_config=runs/<nome>/export/config.yaml data.codes_file=null
La logica dello stadio è in tokenizer/stage1.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf  # noqa: E402

from common.data import TrainStories  # noqa: E402
from common.rundirs import run_dir as prepara_run_dir  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from tokenizer.stage1 import P, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="nome del run: cartella runs/<nome>; \"auto\" = nome standard dalle impostazioni (common/nomi.py)")
ap.add_argument("--config", default="configs/stage1_tokenizer.yaml")
ap.add_argument("overrides", nargs="*", help="modifiche alla config, es. train.steps=300")
a = ap.parse_args()
if a.run == "auto":
    from common.nomi import nome_standard
    a.run = nome_standard(OmegaConf.merge(OmegaConf.load(P(a.config)), OmegaConf.from_dotlist(a.overrides)))
    print(f"[nome] {a.run}", flush=True)

run_dir = prepara_run_dir(a.run)
saved_cfg = os.path.join(run_dir, "config.yaml")
if os.path.exists(saved_cfg):
    # Ripresa: si usa la config salvata alla prima partenza, per non cambiare il run a metà.
    config = OmegaConf.load(saved_cfg)
    if a.overrides:
        print("[config] run esistente: le modifiche da riga di comando sono ignorate, si usa", saved_cfg)
else:
    config = OmegaConf.merge(OmegaConf.load(P(a.config)), OmegaConf.from_dotlist(a.overrides))
    OmegaConf.save(config, saved_cfg)

s = build(config)
copie = set(int(x) for x in (config.export.get("at_steps") or []))


def dopo_passo(step):
    s.dopo_passo(step)
    if step in copie:
        s.export(s.net, run_dir, sub=f"passo_{step:08d}", passo=step)


data = TrainStories(P(config.data.dir), int(config.data.length), int(config.data.seed))
rim = config.tokenizer.get("rimedi") or {}
if rim.get("init_dati", False) and not os.path.isdir(os.path.join(run_dir, "checkpoints")):
    s.init_dai_dati()   # solo alla prima partenza: alla ripresa i codebook vengono dal checkpoint
status = Trainer(run_dir, config, s.net, data, s.loss_fn, s.eval_fn, step_hook=dopo_passo).run()
if status == "done":
    s.export(s.net, run_dir, passo=int(config.train.steps))
print(status)
# Codice di uscita: 0 solo a training completato, così una catena (&&) si ferma se il tokenizer va in pausa.
sys.exit(0 if status == "done" else 1)
