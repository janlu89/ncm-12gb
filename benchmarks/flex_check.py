"""T4 punto 6: FlexAttention contro attenzione con maschera, sul generatore ufficiale.

1. Equivalenza (modalità eval, FP32, pesi ufficiali): differenza massima dei logit < 1e-3.
2. Velocità (training, bf16, torch.compile su entrambi): ms per micro-passo, picco di memoria.

Nota: nel codice upstream il percorso flex NON applica il dropout dell'attenzione (il percorso SDPA sì,
p=0,1 in training). In training i due percorsi non sono quindi lo stesso modello.

Uso: .venv\\Scripts\\python.exe -I benchmarks\\flex_check.py [--block 64] [--steps 60]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import TrainStories, load_valid_rows  # noqa: E402
from generator.stage2 import P, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--micro-batch", type=int, default=16)
ap.add_argument("--out", default=os.path.join("benchmarks", time.strftime("%Y-%m-%d"), "flex_check.json"))
a = ap.parse_args()

config = OmegaConf.load(P("configs/stage2_generator.yaml"))
config.eval.n_stories = 64
s = build(config)
net = s.net
g = torch.load(os.path.join(PROJECT_ROOT, "artifacts", "tinystories", "generator.pt"), map_location="cpu",
               weights_only=False)
net.load_state_dict(g["model"], strict=True)
net.config.flex_block_size = a.block
res = {"block": a.block, "torch": torch.__version__}

# ---------- 1. equivalenza in eval, FP32 ----------
net.eval()
rows = load_valid_rows(P(config.data.dir), 128, 64).cuda()
with torch.no_grad():
    x_in = s.inputs_from_idx(s.encode(rows))
    net.use_flex_attention = False
    ref = net(x_in).float()
    net.use_flex_attention = True
    try:
        flex = net(x_in).float()
        res["eq_max_diff_logits_fp32"] = float((flex - ref).abs().max())
        res["eq_argmax_uguali_pct"] = float((flex.argmax(-1) == ref.argmax(-1)).float().mean() * 100)
    except Exception as e:  # conservare l'errore completo
        res["eq_errore"] = repr(e)
print("[equivalenza]", {k: v for k, v in res.items() if k.startswith("eq")}, flush=True)

# ---------- 2. velocità in training, compilato ----------
net.train()
data = TrainStories(P(config.data.dir), 128, 1234)
batches = [data.next_batch(a.micro_batch).cuda() for _ in range(a.steps + 10)]


def bench(use_flex):
    net.use_flex_attention = use_flex
    torch._dynamo.reset()
    net_c = torch.compile(net)
    t0 = time.perf_counter()

    def step(i):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = s.loss_fn(net_c, batches[i])
        loss.backward()

    step(0)
    torch.cuda.synchronize()
    comp = time.perf_counter() - t0
    for i in range(1, 10):
        step(i)
    net.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for i in range(10, 10 + a.steps):
        step(i)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / a.steps
    net.zero_grad(set_to_none=True)
    return {"compilazione_s": round(comp, 1), "ms_micro_passo": round(dt * 1000, 2),
            "seq_s": round(a.micro_batch / dt, 1), "picco_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}


if "eq_errore" not in res:
    for name, f in (("maschera_compile", False), ("flex_compile", True)):
        try:
            res[name] = bench(f)
        except Exception as e:
            res[name] = {"errore": repr(e)[:2000]}
        print(f"[velocità] {name}: {res[name]}", flush=True)

out = a.out if os.path.isabs(a.out) else os.path.join(PROJECT_ROOT, a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
print("salvato:", out)
