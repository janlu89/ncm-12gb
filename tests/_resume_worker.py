"""Processo di prova per tests/test_resume.py: pochi passi di training della VQ-VAE (pesi nuovi).

Uso: python tests/_resume_worker.py <run_dir> <steps> [stop_at_step]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import TrainStories  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402

run_dir, steps = sys.argv[1], int(sys.argv[2])
stop_at = int(sys.argv[3]) if len(sys.argv) > 3 else None

official = OmegaConf.load(os.path.join(PROJECT_ROOT, "artifacts", "tinystories", "config.yaml"))
config = OmegaConf.create({
    "model": {"vqvae": official.tokenizer.vqvae},
    "data": {"dir": os.path.join(PROJECT_ROOT, "data", "raw", "tinystories"), "length": 128, "seed": 123},
    "train": {"steps": steps, "micro_batch": 16, "accum": 2, "lr": 5e-4, "min_lr": 5e-5, "warmup": 4,
              "weight_decay": 0.01, "betas": [0.9, 0.95], "grad_clip": 1.0, "bf16": True, "seed": 42,
              "log_every": 1, "eval_every": 0, "ckpt_every_steps": 0, "ckpt_every_minutes": 1e9,
              "keep_ckpts": 2, "deterministic": True, "debug_stop_at_step": stop_at},
})
config.model.vqvae.checkpoint_path = None

torch.manual_seed(int(config.train.seed))  # pesi iniziali identici in ogni processo
model = VQVAE(**config.model.vqvae).cuda()
data = TrainStories(config.data.dir, config.data.length, config.data.seed)


def loss_fn(m, x):
    logits, vq_loss, *_ = m(x)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
    return ce + vq_loss, {"ce": ce.item(), "vq_loss": float(vq_loss)}


print(Trainer(run_dir, config, model, data, loss_fn, argv=sys.argv).run())
