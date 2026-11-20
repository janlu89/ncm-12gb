"""Cronometra passi di training NCM (VQ-VAE e generatore) con pesi nuovi e dati casuali.

Non carica e non scrive checkpoint. Misura solo velocità e memoria.
Uso: python ncm_train_bench.py <cartella progetto>
"""
import sys, os, time, json
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "third_party", "ncm"))
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

torch.manual_seed(0)
dev = "cuda"
CFG = OmegaConf.load(os.path.join(root, "artifacts", "tinystories", "config.yaml"))
vq_cfg = CFG.tokenizer.vqvae  # DictConfig: VQVAE legge i sottocampi come attributi; checkpoint_path non viene caricato dal costruttore
L = int(vq_cfg["context_length"])
V_in = int(vq_cfg["in_vocab_size"])

results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


def time_steps(step_fn, warmup=3, steps=10):
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps


def try_batches(name, make_step, batches):
    out = []
    for B in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            step = make_step(B)
            dt = time_steps(step)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            out.append({"micro_batch": B, "s_per_step": round(dt, 4),
                        "seq_per_s": round(B / dt, 1), "peak_vram_gb": round(peak, 2)})
            print(f"[{name}] B={B:4d}  {dt*1000:8.1f} ms/step  {B/dt:8.1f} seq/s  picco {peak:.2f} GB", flush=True)
            del step
        except torch.cuda.OutOfMemoryError:
            print(f"[{name}] B={B:4d}  memoria esaurita", flush=True)
            out.append({"micro_batch": B, "oom": True})
            break
        finally:
            torch.cuda.empty_cache()
    return out


# ---------- Stadio 1: VQ-VAE ----------
def make_vq_step(B):
    model = VQVAE(**vq_cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    x = torch.randint(0, V_in, (B, L), device=dev)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, vq_loss, *_ = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1)) + vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return step

print("=== VQ-VAE ===", flush=True)
results["vqvae"] = try_batches("vqvae", make_vq_step, [64, 128, 256, 512])

# ---------- Stadio 2: generatore ----------
frozen_vq = VQVAE(**vq_cfg).to(dev).eval()
for p in frozen_vq.parameters():
    p.requires_grad = False
P = int(CFG.model.prefix_len)
levels = list(frozen_vq.vq.levels)
T = sum(levels) - levels[0]
D = frozen_vq.embed_dim
V_code = int(CFG.model.vocab_size[0])
results["generator_seq_positions"] = P + T


def make_gen_step(B):
    net = NextConceptTransformer(vqvae_local=frozen_vq, config=CFG.model).to(dev).train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    x = torch.randn(B, P + T, D, device=dev)
    y = torch.randint(0, V_code, (B, T), device=dev)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return step

print(f"=== Generatore (sequenza interna di {P + T} posizioni) ===", flush=True)
results["generator"] = try_batches("generatore", make_gen_step, [8, 16, 32, 64, 128])

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncm_train_bench.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("salvato:", out_path)
