"""Ciclo di training generico con accumulo del gradiente, BF16, checkpoint completi e pausa/ripresa.

Lo script di ogni stadio fornisce il modello, i dati e due funzioni:
    loss_fn(model, batch) -> (loss, {metriche})
    eval_fn(model)        -> {metriche}          (opzionale)

Chiavi di configurazione usate (sezione `train`):
    steps, micro_batch, accum, lr, min_lr, warmup, weight_decay, betas, grad_clip,
    bf16, seed, log_every, eval_every, ckpt_every_steps, ckpt_every_minutes, keep_ckpts,
    deterministic, debug_stop_at_step
"""
import math
import os
import random
import sys
import time

import numpy as np
import torch

from . import checkpoint as ck
from .control import StopControl
from .runlog import JsonlLog, gpu_telemetry, write_metadata


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, cfg):
    """Warmup lineare, poi coseno fino a min_lr (la forma del decay non è nel paper: scelta nostra)."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


def make_optimizer(model, cfg):
    """AdamW; weight decay solo sui pesi con almeno 2 dimensioni. Esclusi i parametri senza gradiente
    (i codebook del VQ sono aggiornati via EMA e hanno requires_grad=False)."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), fused=torch.cuda.is_available())


class Trainer:
    def __init__(self, run_dir, config, model, data, loss_fn, eval_fn=None, argv=None, step_hook=None):
        # step_hook(passo): chiamata dopo ogni passo dell'optimizer, prima di log, valutazione e checkpoint
        # (il tokenizer la usa per aggiornare i codebook una volta per passo). None = nessun effetto.
        self.step_hook = step_hook
        self.run_dir = run_dir
        self.config = config
        self.cfg = config.train
        self.model = model
        self.data = data
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.argv = argv if argv is not None else sys.argv
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.log = JsonlLog(run_dir)
        self.opt = make_optimizer(model, self.cfg)
        self.step = 0

    # ---------- checkpoint ----------
    def _state(self):
        from omegaconf import OmegaConf
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "model_non_persistent": ck.non_persistent_buffers(self.model),
            "optimizer": self.opt.state_dict(),
            "data": self.data.state_dict(),
            "rng": ck.rng_state(),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

    def save(self, reason):
        t = time.time()
        path = ck.save(self.ckpt_dir, self.step, self._state(), keep=int(self.cfg.keep_ckpts))
        dt = time.time() - t
        self.log.write({"event": "checkpoint", "step": self.step, "reason": reason, "path": path,
                        "seconds": round(dt, 2)})
        print(f"[checkpoint] passo {self.step} salvato ({reason}, {dt:.1f}s)", flush=True)

    def try_resume(self):
        state, path = ck.load_latest(self.ckpt_dir)
        if state is None:
            return False
        self.model.load_state_dict(state["model"], strict=True)
        ck.load_non_persistent_buffers(self.model, state["model_non_persistent"])
        self.opt.load_state_dict(state["optimizer"])
        self.data.load_state_dict(state["data"])
        ck.set_rng_state(state["rng"])
        self.step = int(state["step"])
        self.log.write({"event": "resume", "step": self.step, "path": path})
        print(f"[ripresa] dal passo {self.step} ({path})", flush=True)
        return True

    # ---------- ciclo ----------
    def _check_finite(self, loss_val):
        if not math.isfinite(loss_val):
            self.log.write({"event": "error", "step": self.step, "error": "loss non finita"})
            raise FloatingPointError(f"loss non finita al passo {self.step}: {loss_val}")

    def _evaluate(self):
        if self.eval_fn is None:
            return
        self.model.eval()
        with torch.no_grad():
            m = self.eval_fn(self.model)
        self.model.train()
        self.log.write({"event": "eval", "step": self.step, **m})
        print(f"[eval] passo {self.step}: " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in m.items()), flush=True)

    def run(self):
        cfg = self.cfg
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "pid.txt"), "w") as f:  # letto dalla dashboard
            f.write(str(os.getpid()))
        if cfg.get("deterministic", False):
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        resumed = self.try_resume()
        if not resumed:
            set_seed(int(cfg.seed))
            write_metadata(self.run_dir, self._state()["config"], self.argv)
            self.log.write({"event": "start", "step": 0, "steps": int(cfg.steps)})

        self.model.train()
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        params = [p for p in self.model.parameters() if p.requires_grad]
        last_ckpt_time = time.time()
        t_log, seqs_log = time.time(), 0
        torch.cuda.reset_peak_memory_stats()
        stop_debug = cfg.get("debug_stop_at_step", None)

        with StopControl(self.run_dir) as stop:
            while self.step < cfg.steps:
                lr = lr_at(self.step, cfg)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                # Loss, metriche e norma del gradiente restano tensori sulla GPU: si leggono (sincronizzando)
                # solo quando si scrive il log o prima di un checkpoint.
                metrics_acc = {}
                loss_acc = torch.zeros((), device="cuda")
                for _ in range(int(cfg.accum)):
                    batch = self.data.next_batch(int(cfg.micro_batch)).cuda(non_blocking=True)
                    with torch.autocast("cuda", dtype=dtype, enabled=cfg.bf16):
                        loss, metrics = self.loss_fn(self.model, batch)
                    (loss / cfg.accum).backward()
                    loss_acc += loss.detach().float() / cfg.accum
                    for k, v in metrics.items():
                        v = v.detach().float() if torch.is_tensor(v) else float(v)
                        metrics_acc[k] = metrics_acc.get(k, 0.0) + v / cfg.accum
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                if self.step_hook is not None:
                    self.step_hook(self.step)
                seqs_log += int(cfg.micro_batch) * int(cfg.accum)

                log_due = self.step % int(cfg.log_every) == 0 or self.step == cfg.steps
                if log_due:
                    loss_val = float(loss_acc)
                    dt = time.time() - t_log
                    rec = {"event": "train", "step": self.step, "loss": loss_val, "lr": lr,
                           "grad_norm": float(gnorm), "seq_per_s": seqs_log / max(dt, 1e-9),
                           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
                           **{k: float(v) for k, v in metrics_acc.items()}, **gpu_telemetry()}
                    self.log.write(rec)
                    print(f"passo {self.step}/{cfg.steps} loss {loss_val:.4f} lr {lr:.2e} "
                          f"gnorm {rec['grad_norm']:.2f} {rec['seq_per_s']:.0f} seq/s "
                          f"GPU {rec.get('gpu_temp_c', '?')}°C {rec.get('gpu_power_w', '?')}W", flush=True)
                    t_log, seqs_log = time.time(), 0
                    self._check_finite(loss_val)

                if cfg.eval_every and self.step % int(cfg.eval_every) == 0:
                    self._evaluate()

                if stop_debug is not None and self.step == int(stop_debug):
                    stop.request("debug_stop_at_step")
                due_steps = cfg.ckpt_every_steps and self.step % int(cfg.ckpt_every_steps) == 0
                due_time = time.time() - last_ckpt_time >= 60 * float(cfg.ckpt_every_minutes)
                if stop.should_stop() or due_steps or due_time:
                    self._check_finite(float(loss_acc))  # mai salvare un checkpoint con loss non finita
                if stop.should_stop():
                    self.save(f"pausa ({stop.reason})")
                    self.log.write({"event": "paused", "step": self.step, "reason": stop.reason})
                    print(f"[pausa] fermato al passo {self.step}. Per riprendere rilancia lo stesso comando.",
                          flush=True)
                    return "paused"
                if due_steps or due_time:
                    self.save("periodico")
                    last_ckpt_time = time.time()

        if self.eval_fn is not None and not (cfg.eval_every and self.step % int(cfg.eval_every) == 0):
            self._evaluate()
        self.save("fine")
        self.log.write({"event": "done", "step": self.step})
        return "done"
