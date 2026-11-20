"""Checkpoint completi, scritti in modo atomico, con ripresa esatta.

Un checkpoint contiene: pesi e buffer del modello (anche i buffer non persistenti, che lo state_dict
di PyTorch salta), stato dell'optimizer, passo, stato dei dati, stati di tutti i generatori casuali
e configurazione. Si scrive su un file temporaneo e poi lo si rinomina: un'interruzione durante il
salvataggio lascia intatto il checkpoint precedente.
"""
import glob
import os
import random
import re

import numpy as np
import torch

_NAME = re.compile(r"step_(\d{8})\.pt$")


def non_persistent_buffers(model):
    """Buffer registrati con persistent=False (es. contatori interni del VQ upstream)."""
    persistent = set(model.state_dict().keys())
    return {name: buf.detach().clone() for name, buf in model.named_buffers() if name not in persistent}


def load_non_persistent_buffers(model, saved):
    own = dict(model.named_buffers())
    for name, val in saved.items():
        own[name].copy_(val.to(own[name].device))


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def list_checkpoints(ckpt_dir):
    found = []
    for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        m = _NAME.search(p)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def save(ckpt_dir, step, state, keep=2):
    os.makedirs(ckpt_dir, exist_ok=True)
    final = os.path.join(ckpt_dir, f"step_{step:08d}.pt")
    tmp = final + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    for _, old in list_checkpoints(ckpt_dir)[:-keep]:
        os.remove(old)
    return final


def load_latest(ckpt_dir):
    """Carica il checkpoint più recente leggibile; se il più recente è danneggiato prova il precedente."""
    for step, path in reversed(list_checkpoints(ckpt_dir)):
        try:
            return torch.load(path, map_location="cpu", weights_only=False), path
        except Exception as e:  # file danneggiato: si passa al precedente
            print(f"[checkpoint] impossibile leggere {path}: {e!r}; provo il precedente", flush=True)
    return None, None
