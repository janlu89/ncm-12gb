"""Profilo di un micro-passo di training del generatore (T2 di docs/piani/fase1-operativo.md).

Misura il tempo per blocco (encode, input, forward, loss, backward, optimizer) con eventi CUDA,
la tabella dei kernel con torch.profiler e quattro prove A/B da 100 micro-passi:
  base · attenzione senza maschera (solo cronometro, risultato sbagliato) · codici già pronti · senza .item()

Uso: .venv\\Scripts\\python.exe -I benchmarks\\profile_generator.py [--micro-batch 16] [--steps 100] [--out benchmarks\\<data>\\profile_generator.json]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.data import TrainStories  # noqa: E402
from common.trainer import make_optimizer  # noqa: E402
from generator.stage2 import P, build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage2_generator.yaml")
ap.add_argument("--micro-batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=100, help="micro-passi per ogni prova A/B")
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--accum", type=int, default=64, help="per ripartire il costo dell'optimizer")
ap.add_argument("--out", default=os.path.join("benchmarks", time.strftime("%Y-%m-%d"), "profile_generator.json"))
ap.add_argument("--compile", action="store_true", help="aggiunge le prove con torch.compile del forward")
ap.add_argument("--only-ab", action="store_true", help="salta blocchi e profiler, fa solo le prove A/B")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()

config = OmegaConf.merge(OmegaConf.load(P(a.config)), OmegaConf.from_dotlist(a.overrides))
config.eval.n_stories = 64  # la valutazione non serve qui
s = build(config)
net, cfg = s.net, config.train
net.train()
opt = make_optimizer(net, cfg)
params = [p for p in net.parameters() if p.requires_grad]
B = a.micro_batch
n_batches = a.warmup + a.steps
data = TrainStories(P(config.data.dir), int(config.data.length), int(config.data.seed))
batches = [data.next_batch(B).cuda() for _ in range(n_batches)]
codes = [s.encode(x) for x in batches]
L0 = s.levels[0]
bf16 = dict(device_type="cuda", dtype=torch.bfloat16)


def sync_time(fn, n):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(n):
        fn(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


# ---------- micro-passi ----------
def make_step(model, use_codes=False):
    """Micro-passo come nel trainer attuale (dopo T3: nessuna lettura sincrona)."""
    def step(i):
        with torch.autocast(**bf16):
            if use_codes:
                loss, _ = s.loss_from_idx(model, codes[i % n_batches])
            else:
                loss, _ = s.loss_fn(model, batches[i % n_batches])
        (loss / a.accum).backward()
    return step


step_base = make_step(net)
step_codes = make_step(net, use_codes=True)


_sdpa = F.scaled_dot_product_attention


def _sdpa_nomask(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    return _sdpa(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False, **kw)


def run_ab(name, fn, patch_nomask=False):
    if patch_nomask:
        torch.nn.functional.scaled_dot_product_attention = _sdpa_nomask
    try:
        for i in range(a.warmup):
            fn(i)
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        dt = sync_time(lambda i: fn(a.warmup + i), a.steps)
        opt.zero_grad(set_to_none=True)
        return {"ms_micro_passo": round(dt * 1000, 2), "seq_s": round(B / dt, 1),
                "picco_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(f"  {name}: memoria esaurita", flush=True)
        return {"oom": True}
    finally:
        torch.nn.functional.scaled_dot_product_attention = _sdpa


res = {"micro_batch": B, "steps": a.steps, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "data": time.strftime("%Y-%m-%d %H:%M:%S")}

blocks = ["encode", "input", "forward", "loss", "backward"]
tot = None
if not a.only_ab:
    # ---------- 1. tempo per blocco con eventi CUDA ----------
    acc = defaultdict(float)
    for i in range(a.warmup + a.steps):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(blocks) + 1)]
        x = batches[i % n_batches]
        ev[0].record()
        idx = s.encode(x)
        ev[1].record()
        x_in = s.inputs_from_idx(s.corrupt(idx))
        ev[2].record()
        with torch.autocast(**bf16):
            logits = net(x_in).float()
        ev[3].record()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, L0:].reshape(-1))
        ev[4].record()
        (nll / a.accum).backward()
        ev[5].record()
        torch.cuda.synchronize()
        if i >= a.warmup:
            for k, name in enumerate(blocks):
                acc[name] += ev[k].elapsed_time(ev[k + 1])
        del logits, nll, x_in
    opt.zero_grad(set_to_none=True)
    per_block = {k: v / a.steps for k, v in acc.items()}

    # optimizer: ripartito su accum micro-passi
    for p in params:
        p.grad = torch.zeros_like(p)
    t_opt = []
    for _ in range(10):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        e1.record()
        torch.cuda.synchronize()
        t_opt.append(e0.elapsed_time(e1))
    opt.zero_grad(set_to_none=True)
    per_block["optimizer_ripartito"] = sum(t_opt[2:]) / len(t_opt[2:]) / a.accum
    tot = sum(per_block.values())
    res["blocchi_ms"] = {k: round(v, 2) for k, v in per_block.items()}
    res["blocchi_pct"] = {k: round(100 * v / tot, 1) for k, v in per_block.items()}
    res["somma_blocchi_ms"] = round(tot, 2)
    print("\n[blocchi] ms per micro-passo:")
    for k, v in per_block.items():
        print(f"  {k:22s} {v:8.2f} ms  {100 * v / tot:5.1f}%")

    # ---------- 2. kernel con torch.profiler ----------
    from torch.profiler import ProfilerActivity, profile, record_function, schedule  # noqa: E402

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=2, warmup=2, active=5)) as prof:
        for i in range(9):
            x = batches[i]
            with record_function("encode"):
                idx = s.encode(x)
            with record_function("input"):
                x_in = s.inputs_from_idx(s.corrupt(idx))
            with record_function("forward"):
                with torch.autocast(**bf16):
                    logits = net(x_in).float()
            with record_function("loss"):
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, L0:].reshape(-1))
            with record_function("backward"):
                (nll / a.accum).backward()
            prof.step()
    opt.zero_grad(set_to_none=True)


    def dev_time(e):
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return v
        return 0


    def group(name):
        n = name.lower()
        if any(k in n for k in ("fmha", "attention", "mem_eff", "flash", "efficient")):
            return "attenzione"
        if any(k in n for k in ("gemm", "cutlass", "xmma", "matmul", "s16816", "s1688", "cublas", "splitk")):
            return "matmul"
        if any(k in n for k in ("copy", "memcpy", "memset", "cast", "fill", "bfloat16tofloat", "floattobfloat")):
            return "copie/conversioni"
        if any(k in n for k in ("softmax", "nll", "cross_entropy", "log_softmax")):
            return "softmax/loss"
        return "elementwise/norme/altro"


    kern = [e for e in prof.key_averages()
            if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA and dev_time(e) > 0
            and e.key not in blocks and not e.key.startswith("ProfilerStep")]  # esclude gli intervalli di record_function
    groups = defaultdict(float)
    for e in kern:
        groups[group(e.key)] += dev_time(e)
    gt = sum(groups.values())
    res["kernel_gruppi_pct"] = {k: round(100 * v / gt, 1) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])}
    top = sorted(kern, key=dev_time, reverse=True)[:25]
    res["kernel_top25"] = [{"nome": e.key[:160], "gruppo": group(e.key), "ms_totali_5_passi": round(dev_time(e) / 1000, 2),
                            "chiamate": e.count} for e in top]
    print("\n[kernel] tempo GPU per gruppo:")
    for k, v in res["kernel_gruppi_pct"].items():
        print(f"  {k:25s} {v:5.1f}%")
    print("\n[kernel] i 10 più costosi:")
    for r in res["kernel_top25"][:10]:
        print(f"  {r['ms_totali_5_passi']:8.2f} ms  {r['gruppo']:24s} {r['nome'][:90]}")


# ---------- 3. prove A/B ----------
print(f"\n[A/B] {a.steps} micro-passi ciascuna ...", flush=True)
ab = {"base": run_ab("base", step_base)}
if not a.only_ab:
    ab["senza_maschera"] = run_ab("senza_maschera", step_base, patch_nomask=True)
    ab["codici_pronti"] = run_ab("codici_pronti", step_codes)
if a.compile:
    net_c = torch.compile(net)
    t0 = time.perf_counter()
    make_step(net_c)(0)
    torch.cuda.synchronize()
    res["compile_primo_passo_s"] = round(time.perf_counter() - t0, 1)
    opt.zero_grad(set_to_none=True)
    print(f"  [compile] primo passo (compilazione inclusa): {res['compile_primo_passo_s']} s", flush=True)
    ab["compile"] = run_ab("compile", make_step(net_c))
    ab["compile_codici_pronti"] = run_ab("compile_codici_pronti", make_step(net_c, use_codes=True))
b0 = ab["base"].get("ms_micro_passo") or next((v["ms_micro_passo"] for v in ab.values() if "ms_micro_passo" in v), 1.0)
for k, v in ab.items():
    if v.get("oom"):
        continue
    v["guadagno_pct"] = round(100 * (b0 - v["ms_micro_passo"]) / b0, 1)
    v["accelerazione"] = round(b0 / v["ms_micro_passo"], 3)
    print(f"  {k:22s} {v['ms_micro_passo']:8.2f} ms  {v['seq_s']:6.1f} seq/s  picco {v['picco_gb']:.2f} GB  "
          f"x{v['accelerazione']:.3f}", flush=True)
res["ab"] = ab
if tot is not None:
    res["copertura_blocchi_su_base_pct"] = round(100 * tot / b0, 1)
    print(f"\n[copertura] somma dei blocchi = {res['copertura_blocchi_su_base_pct']}% del micro-passo base")

out = a.out if os.path.isabs(a.out) else os.path.join(PROJECT_ROOT, a.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
if not a.only_ab:
    prof.export_chrome_trace(out.replace(".json", "_trace.json"))
print("salvato:", out)
