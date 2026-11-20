"""Misure di riferimento sui checkpoint NCM ufficiali TinyStories (nessun training, sola lettura).

1. Tokenizer: ricostruzione del testo di validazione usando i primi k livelli (k = 1..12).
2. Generatore: loss in teacher forcing con l'input costruito da vq.idx_to_ncm_input,
   più due controlli (input mescolato, generatore con pesi casuali) per confermare il formato.
3. Livello 0: distribuzione empirica sul validation set contro l'unigramma salvato nel checkpoint.

Uso (dalla cartella del progetto):
    .venv\Scripts\python.exe -I evaluation\reference_official.py --n 4096 --out evaluation\results\<file>.json
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "ncm"))
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4096, help="storie di validazione da usare")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"

# ---------------- modelli ufficiali ----------------
ART = os.path.join(ROOT, "artifacts", "tinystories")
CFG = OmegaConf.load(os.path.join(ART, "config.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ART, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(ART, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(ART, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vqvae = tk.vqvae.to(dev).eval()
vq = vqvae.vq
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
L = vqvae.context_length
levels = list(vq.levels)

net = NextConceptTransformer(vqvae_local=vqvae, config=CFG.model).to(dev)
g = torch.load(os.path.join(ART, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(g["model"], strict=True)
net.eval()
l0_unigram = g["inference"]["level0_unigram"].float()

rand_net = NextConceptTransformer(vqvae_local=vqvae, config=CFG.model).to(dev).eval()  # pesi casuali
P = int(CFG.model.prefix_len)

# ---------------- dati di validazione ----------------
valid = np.fromfile(os.path.join(ROOT, "data", "raw", "tinystories", "valid_bpe_8192.bin"), dtype=np.uint16)
ends = np.flatnonzero(valid == EOS)
starts = np.concatenate([[0], ends[:-1] + 1])
n = min(a.n, len(starts))
rows = np.full((n, L), PAD, dtype=np.int64)
for i in range(n):
    s = valid[starts[i]:ends[i] + 1][:L]
    rows[i, :len(s)] = s
rows = torch.from_numpy(rows)
print(f"storie di validazione usate: {n} (troncate/paddate a {L})", flush=True)

# ---------------- accumulatori ----------------
K = len(levels)
rec_ok = torch.zeros(K, dtype=torch.float64)       # token corretti ricostruendo con i primi k livelli
rec_ok_nopad = torch.zeros(K, dtype=torch.float64)
rec_seq_ok = torch.zeros(K, dtype=torch.float64)
n_tok = n_tok_nopad = 0
lvl_slices = []
off = 0
for l in levels[1:]:
    lvl_slices.append((off, off + l))
    off += l
gen = {name: {"nll": torch.zeros(K - 1, dtype=torch.float64), "ok": torch.zeros(K - 1, dtype=torch.float64)}
       for name in ("ufficiale", "input_mescolato", "pesi_casuali")}
l0_counts = torch.zeros(vq.vqvae_vocab_sizes[0], dtype=torch.float64)

t0 = time.time()
with torch.no_grad():
    for b in range(0, n, a.bs):
        x = rows[b:b + a.bs].to(dev)
        B = x.shape[0]
        idx = vqvae.encode(x)                         # [B, sum(levels)]
        idx = idx.view(B, -1)

        # 1. ricostruzione con i primi k livelli
        # Nota: vq.idx_to_zhats somma in place e restituisce K riferimenti allo stesso tensore,
        # quindi qui si accumula senza modificare i tensori già salvati.
        z_hats = []
        z_acc = torch.zeros(B, levels[-1], vqvae.embed_dim, device=dev)
        o = 0
        for li, l in enumerate(levels):
            emb = vq.embeddings[li](idx[:, o:o + l])
            q = vq._up_to_full(emb, li, levels[-1]) if li != K - 1 else emb
            z_acc = z_acc + vq._apply_quant_resi(q, li)
            z_hats.append(z_acc)
            o += l
        nopad = x != PAD
        for k in range(K):
            pred = vqvae.decoder(z_hats[k]).argmax(-1)
            ok = pred == x
            rec_ok[k] += ok.sum().item()
            rec_ok_nopad[k] += (ok & nopad).sum().item()
            rec_seq_ok[k] += ok.all(dim=1).sum().item()
        n_tok += x.numel()
        n_tok_nopad += nopad.sum().item()

        # 3. livello 0
        l0_counts += torch.bincount(idx[:, 0].cpu(), minlength=l0_counts.numel()).double()

        # 2. generatore in teacher forcing
        z_prefix = vqvae.encoder.wte(torch.full((B, P), EOS, device=dev))
        ncm_in = vq.idx_to_ncm_input(idx)            # [B, sum(levels[1:]), D]
        target = idx[:, levels[0]:]
        variants = {
            "ufficiale": (net, ncm_in),
            "input_mescolato": (net, ncm_in[torch.randperm(B, device=dev)]),
            "pesi_casuali": (rand_net, ncm_in),
        }
        for name, (model, inp) in variants.items():
            logits = model(torch.cat([z_prefix, inp], dim=1)).float()
            for j, (s, e) in enumerate(lvl_slices):
                lg, tg = logits[:, s:e], target[:, s:e]
                gen[name]["nll"][j] += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="sum").item()
                gen[name]["ok"][j] += (lg.argmax(-1) == tg).sum().item()
        print(f"  {b + B}/{n}  ({time.time() - t0:.0f}s)", flush=True)

# ---------------- risultati ----------------
res = {"storie": n, "context_length": L, "levels": levels, "seed": a.seed,
       "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
res["tokenizer"] = [{
    "livelli_usati": k + 1,
    "acc_token": round(rec_ok[k].item() / n_tok, 5),
    "acc_token_senza_pad": round(rec_ok_nopad[k].item() / n_tok_nopad, 5),
    "sequenze_perfette": round(rec_seq_ok[k].item() / n, 5),
} for k in range(K)]

res["generatore"] = {}
for name, d in gen.items():
    per = []
    tot_nll = tot_n = 0.0
    for j, (s, e) in enumerate(lvl_slices):
        cnt = n * (e - s)
        per.append({"livello": j + 1, "posizioni": e - s,
                    "loss": round(d["nll"][j].item() / cnt, 4), "acc": round(d["ok"][j].item() / cnt, 4)})
        tot_nll += d["nll"][j].item()
        tot_n += cnt
    res["generatore"][name] = {"loss_media_per_codice": round(tot_nll / tot_n, 4), "per_livello": per}
res["generatore"]["loss_uniforme_ln4096"] = round(math.log(4096), 4)

emp = l0_counts / l0_counts.sum()
m = 0.5 * (emp + l0_unigram.double())
def _kl(p, q):
    mask = p > 0
    return float((p[mask] * (p[mask] / q[mask]).log()).sum())
res["livello0"] = {
    "codici_usati_validation": int((l0_counts > 0).sum()),
    "codici_nonzero_unigramma_checkpoint": int((l0_unigram > 0).sum()),
    "unigramma_min_nonzero": float(l0_unigram[l0_unigram > 0].min()),
    "unigramma_min_nonzero_x65536": float(l0_unigram[l0_unigram > 0].min() * 65536),
    "jensen_shannon_validation_vs_checkpoint": round(0.5 * _kl(emp, m) + 0.5 * _kl(l0_unigram.double(), m), 5),
}
res["secondi"] = round(time.time() - t0, 1)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
